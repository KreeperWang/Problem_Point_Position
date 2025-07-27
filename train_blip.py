#!/usr/bin/env python
"""
BLIP2 training script for geometry point position detection
Main entry point - optimized for 16GB GPU
"""

import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

# 如果要使用LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("PEFT not available. Install with: pip install peft")

from config import Config, default_config
from prompt import PromptTemplate

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class GeometryPointDataset(Dataset):
    """几何点检测数据集"""
    
    def __init__(
        self, 
        json_path: str, 
        image_dir: str, 
        processor,
        mode: str = 'train'
    ):
        self.image_dir = image_dir
        self.processor = processor
        self.mode = mode
        
        # 加载标注
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 转换为列表格式，验证图像存在
        self.data = []
        missing_count = 0
        for img_id, points in self.annotations.items():
            image_path = os.path.join(image_dir, f"{img_id}.png")
            #if int(img_id)>200: continue
            if os.path.exists(image_path):
                self.data.append({
                    'image_id': img_id,
                    'points': points,
                    'image_path': image_path
                })
            else:
                missing_count += 1
                logger.warning(f"Image not found: {image_path}")
        
        logger.info(f"Loaded {len(self.data)} valid samples from {len(self.annotations)} annotations")
        if missing_count > 0:
            logger.warning(f"Skipped {missing_count} samples due to missing images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 在每次调用时导入必要的库（解决多进程问题）
        import numpy as np
        from PIL import Image
        import torch
        import json
        
        item = self.data[idx]
        
        try:
            # 加载图像
            image = Image.open(item['image_path']).convert('RGB')
            
            # 获取点名
            point_names = list(item['points'].keys())
            
            # 生成提示词
            if self.mode == 'train':
                prompt = PromptTemplate.get_instruction_prompt(point_names)
            else:
                prompt = PromptTemplate.get_point_detection_prompt(point_names)
            
            # 生成目标输出
            target = json.dumps({k: item['points'][k] for k in point_names})
            
            # 分别处理图像和文本（避免使用完整的processor调用）
            # 1. 处理图像
            pixel_values = self.processor.image_processor(
                images=image,
                return_tensors="pt"
            )['pixel_values']
            
            # 2. 处理文本输入
            text_inputs = self.processor.tokenizer(
                text=prompt,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # 3. 处理标签
            labels = self.processor.tokenizer(
                text=target, #'{"A": [76.34, 337.0864661654135], "B": [484.06500000000005, 147.25877192982455], "C": [978.5400000000001, 84.89035087719299]}'
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
                add_special_tokens=False
            )
            
            return {
                'input_ids': text_inputs['input_ids'].squeeze(0),
                'attention_mask': text_inputs['attention_mask'].squeeze(0),
                'pixel_values': pixel_values.squeeze(0),
                'labels': labels['input_ids'].squeeze(0),
                'image_id': item['image_id'],
                'point_names': point_names,
                'ground_truth': item['points']
            }
            
        except Exception as e:
            logger.error(f"Error processing item {idx}, image_id: {item.get('image_id', 'unknown')}: {str(e)}")
            
            # 创建默认样本
            default_image = Image.new('RGB', (384, 384), color='white')
            default_prompt = "Identify points A"
            default_target = '{"A": [192, 192]}'
            
            # 处理默认数据
            pixel_values = self.processor.image_processor(
                images=default_image,
                return_tensors="pt"
            )['pixel_values']
            
            text_inputs = self.processor.tokenizer(
                text=default_prompt,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            labels = self.processor.tokenizer(
                text=default_target,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            
            return {
                'input_ids': text_inputs['input_ids'].squeeze(0),
                'attention_mask': text_inputs['attention_mask'].squeeze(0),
                'pixel_values': pixel_values.squeeze(0),
                'labels': labels['input_ids'].squeeze(0),
                'image_id': f'error_{idx}',
                'point_names': ['A'],
                'ground_truth': {'A': [192, 192]}
            }


def collate_fn(batch):
    """自定义collate函数"""
    # 分离张量和元数据
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # 元数据
    image_ids = [item['image_id'] for item in batch]
    point_names_list = [item['point_names'] for item in batch]
    ground_truths = [item['ground_truth'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels,
        'image_ids': image_ids,
        'point_names_list': point_names_list,
        'ground_truths': ground_truths
    }


class PointDetectionTrainer:
    """点检测训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.training.device)
        
        # 设置随机种子
        self.set_seed(config.training.seed)
        
        # 初始化模型和处理器
        self.setup_model()
        
        # 初始化数据集
        self.setup_datasets()
        
        # 初始化优化器
        self.setup_optimization()
        
        # 初始化混合精度训 list
        if config.training.fp16:
            self.scaler = GradScaler(
                init_scale=2.**16, 
                growth_factor=2.0, 
                backoff_factor=0.5, 
                growth_interval=2000
            )
        
        # 训练状态
        self.global_step = 0
        self.best_metric = float('inf')

        self.last_saved_epoch = 0
        
    def set_seed(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def setup_model(self):
        """初始化模型 - 不使用PEFT以避免兼容性问题"""
        logger.info(f"Loading model from {self.config.model.model_path}")
        
        # Try loading from local path first
        try:
            # 加载处理器
            self.processor = Blip2Processor.from_pretrained(self.config.model.model_path)
                    # 使用更激进的内存优化策略
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.config.model.model_path,
                torch_dtype=torch.float16 if self.config.training.fp16 else torch.float32
            )
            # 加载模型
            # self.model = Blip2ForConditionalGeneration.from_pretrained(
            #     self.config.model.model_path,
            #     torch_dtype=torch.float16 if self.config.training.fp16 else torch.float32
            # )

            logger.info("Successfully loaded model from local path")
            
        except Exception as e:
            logger.warning(f"Failed to load model from local path: {str(e)}")
            logger.info("Attempting to download from Hugging Face...")
            
            # Fallback to Hugging Face model hub
            try:
                hf_model_name = "Salesforce/blip2-opt-2.7b"
                
                logger.info(f"Downloading model: {hf_model_name}")
                self.processor = Blip2Processor.from_pretrained(hf_model_name)
                
                cache_dir = os.path.join(os.path.dirname(self.config.model.model_path), "hf_cache")
                os.makedirs(cache_dir, exist_ok=True)
                
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    hf_model_name,
                    torch_dtype=torch.float16 if self.config.training.fp16 else torch.float32,
                    cache_dir=cache_dir
                )
                
                # Save the model locally for future use
                logger.info(f"Saving model to {self.config.model.model_path}")
                os.makedirs(self.config.model.model_path, exist_ok=True)
                self.processor.save_pretrained(self.config.model.model_path)
                self.model.save_pretrained(self.config.model.model_path)
                
                logger.info("Successfully downloaded and saved model")
                
            except Exception as e2:
                logger.error(f"Failed to download model from Hugging Face: {str(e2)}")
                raise RuntimeError(
                    "Could not load model. Please ensure you have internet connection "
                    "or a valid model at the specified path."
                )
        
        # 自定义参数冻结策略（替代LoRA）
        if self.config.lora.use_lora:
            logger.info("Using custom partial fine-tuning instead of LoRA...")
            logger.info("PEFT/LoRA is not compatible with BLIP2. Using parameter-efficient fine-tuning instead.")
            
            # 首先冻结所有参数
            for param in self.model.parameters():
                param.requires_grad = False
            
            # 策略1: 只微调Q-Former（推荐）
            if hasattr(self.model, 'qformer') and not self.config.model.freeze_qformer:
                logger.info("Unfreezing Q-Former for fine-tuning...")
                for name, param in self.model.qformer.named_parameters():
                    param.requires_grad = True
                logger.info("Q-Former unfrozen")
            
            # 策略2: 微调语言模型的投影层和最后几层
            if hasattr(self.model, 'language_projection'):
                for param in self.model.language_projection.parameters():
                    param.requires_grad = True
                logger.info("Language projection layer unfrozen")
            
            # 策略3: 如果需要，可以微调语言模型的最后几层
            if hasattr(self.model, 'language_model'):
                # 只微调最后的输出层
                if hasattr(self.model.language_model, 'lm_head'):
                    for param in self.model.language_model.lm_head.parameters():
                        param.requires_grad = True
                    logger.info("Language model head unfrozen")
        
        else:
            # 标准的参数冻结（完全微调）
            logger.info("Using full fine-tuning (no LoRA)...")
            
            if self.config.model.freeze_vit:
                logger.info("Freezing vision model...")
                for param in self.model.vision_model.parameters():
                    param.requires_grad = False
            
            if self.config.model.freeze_qformer:
                logger.info("Freezing Q-Former...")
                for param in self.model.qformer.parameters():
                    param.requires_grad = False
        
        # 梯度检查点（节省显存）
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

            
        if self.config.training.fp16:
            self.model.half()  # 转换为半精度       
        # 将模型移到GPU
        self.model.to(self.device)
        
        # 计算可训练参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info(f"\nParameter Statistics:")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logger.info(f"Frozen parameters: {frozen_params:,}")
        
        # 确保有参数可训练
        if trainable_params == 0:
            logger.warning("WARNING: No trainable parameters!")
            logger.warning("Unfreezing Q-Former as fallback...")
            if hasattr(self.model, 'qformer'):
                for param in self.model.qformer.parameters():
                    param.requires_grad = True
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                logger.info(f"Now have {trainable_params:,} trainable parameters")
    
    def setup_datasets(self):
        """初始化数据集"""
        # 训练集
        train_dataset_full = GeometryPointDataset(
            self.config.data.train_json_path,
            self.config.data.image_dir_path,
            self.processor,
            mode='train'
        )
        
        # 划分验证集
        total_size = len(train_dataset_full)
        val_size = int(total_size * self.config.data.val_ratio)
        train_size = total_size - val_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            train_dataset_full, 
            [train_size, val_size]
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.evaluation.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
    
    def setup_optimization(self):
        """设置优化器和调度器"""
        # 优化器
        optimizer_params = {
            'lr': self.config.training.learning_rate,
            'weight_decay': self.config.training.weight_decay,
            'eps': self.config.training.adam_epsilon,
            'betas': (self.config.training.adam_beta1, self.config.training.adam_beta2)
        }
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **optimizer_params
        )
        
        # 学习率调度器
        num_training_steps = len(self.train_loader) * self.config.training.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.training.warmup_ratio)
        
        if self.config.training.lr_scheduler_type == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:  # cosine
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
    
    def train_epoch(self, epoch):
        """训练一个epoch - 修复FP16和学习率问题"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        # 确保优化器梯度清零
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            if self.config.training.fp16:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels
                    )
                    loss = outputs.loss / self.config.training.gradient_accumulation_steps
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels
                )
                loss = outputs.loss / self.config.training.gradient_accumulation_steps
            
            # 反向传播
            if self.config.training.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.config.training.fp16:
                    # 修复FP16梯度问题
                    #self.scaler.unscale_(self.optimizer)
                    
                    # 检查是否有inf/nan梯度
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=1.0,
                        error_if_nonfinite=False  # 不因为inf/nan报错
                    )
                    
                    # 更新参数
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                # 更新学习率调度器
                self.scheduler.step()
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 更新全局步数
                self.global_step += 1
            
            # 更新进度条
            total_loss += loss.item() * self.config.training.gradient_accumulation_steps
            avg_loss = total_loss / (step + 1)
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({'loss': avg_loss, 'lr': current_lr})
            
            # 日志记录
            if self.global_step > 0 and self.global_step % self.config.training.logging_steps == 0:
                logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")



    def evaluate(self):
        """评估模型 - 修复生成参数问题"""
        self.model.eval()
        predictions = {}

        # 单样本评估
        single_sample_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        with torch.no_grad():
            for i, batch in enumerate(tqdm(single_sample_loader, desc="Evaluating")):
                try:
                    # 原始图像路径和元数据
                    img_id = batch['image_ids'][0]
                    point_names = batch['point_names_list'][0]
                    ground_truth = batch['ground_truths'][0]
                    image_path = os.path.join(self.config.data.image_dir_path, f"{img_id}.png")

                    # 重新载入原始 PIL.Image，确保 processor 做全套预处理
                    image = Image.open(image_path).convert("RGB")

                    # 构造 prompt
                    prompt = PromptTemplate.get_point_detection_prompt(point_names)

                    # 一次性由 processor 处理图像和文本
                    inputs = self.processor(
                        images=image,
                        text=prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=256,
                        return_tensors="pt"
                    ).to(self.device)

                    # 如果使用FP16，需要确保输入也是半精度
                    if self.config.training.fp16:
                        # 将pixel_values转换为半精度
                        if 'pixel_values' in inputs:
                            inputs['pixel_values'] = inputs['pixel_values'].half()

                    # 生成时需要正确设置参数
                    try:
                        # 使用max_new_tokens而不是max_length
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=100,  # 改用max_new_tokens
                            min_length=1,        # 确保至少生成一些内容
                            num_beams=3,
                            temperature=0.7,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                        )
                        
                        # 解码生成的文本
                        generated_text = self.processor.decode(
                            generated_ids[0],
                            skip_special_tokens=True
                        )                        
                        
                        # 去掉prompt部分（如果包含的话）
                        # BLIP2有时会重复prompt，需要处理这种情况
                        if prompt in generated_text:
                            generated_text = generated_text.split(prompt)[-1].strip()
                        
                        # 如果生成的文本以"Answer:"或类似标记开头，去掉它
                        if generated_text.startswith("Answer:"):
                            generated_text = generated_text[7:].strip()
                        
                    except Exception as gen_err:
                        logger.warning(f"Generation failed for {img_id}: {gen_err}")
                        generated_text = "{}"

                    # 调试输出
                    if i < 3:
                        logger.info(f"=====================================")
                        logger.info(f"\nSample {img_id}:")
                        #logger.info(f"  Prompt: {prompt}")
                        logger.info(f"  Generated: {generated_text}")
                        logger.info(f"  Expected: {ground_truth}")
                        logger.info(f"=====================================")

                    # 解析模型输出
                    pred_points = PromptTemplate.parse_model_output(generated_text, point_names)
                    predictions[img_id] = {
                        'predictions': pred_points,
                        'ground_truth': ground_truth
                    }

                except Exception as e:
                    logger.error(f"Error evaluating sample {i}: {e}")
                    # 添加默认预测以避免完全失败
                    predictions[img_id] = {
                        'predictions': {},
                        'ground_truth': ground_truth
                    }
                    continue

        # 计算 MAE 和各阈值准确率
        mae = self.calculate_mae(predictions)
        accuracies = self.calculate_accuracy(predictions)

        logger.info(f"Validation MAE: {mae:.2f}")
        for threshold, acc in accuracies.items():
            logger.info(f"Accuracy@{threshold}px: {acc:.2%}")

        return mae, predictions

    
    def calculate_mae(self, predictions):
        """计算平均绝对误差"""
        errors = []
        
        for img_id, data in predictions.items():
            gt_points = data['ground_truth']
            pred_points = data['predictions']
            
            for point_name in gt_points:
                if point_name in pred_points:
                    gt_coord = gt_points[point_name]
                    pred_coord = pred_points[point_name]
                    
                    error = np.sqrt(
                        (gt_coord[0] - pred_coord[0])**2 + 
                        (gt_coord[1] - pred_coord[1])**2
                    )
                    errors.append(error)
        
        if errors:
            return np.mean(errors)
        else:
            return float('inf')
    
    def calculate_accuracy(self, predictions):
        """计算不同阈值下的准确率"""
        thresholds = self.config.evaluation.distance_thresholds
        correct_counts = {t: 0 for t in thresholds}
        total_count = 0
        
        for img_id, data in predictions.items():
            gt_points = data['ground_truth']
            pred_points = data['predictions']
            
            for point_name in gt_points:
                if point_name in pred_points:
                    gt_coord = gt_points[point_name]
                    pred_coord = pred_points[point_name]
                    
                    distance = np.sqrt(
                        (gt_coord[0] - pred_coord[0])**2 + 
                        (gt_coord[1] - pred_coord[1])**2
                    )
                    
                    total_count += 1
                    for threshold in thresholds:
                        if distance <= threshold:
                            correct_counts[threshold] += 1
        
        accuracies = {}
        if total_count > 0:
            for threshold in thresholds:
                accuracies[threshold] = correct_counts[threshold] / total_count
        
        return accuracies
    
    
    
    def save_checkpoint(self, epoch, mae, force_save=False):
        """保存检查点
        
        Args:
            epoch: 当前epoch
            mae: 当前MAE指标
            force_save: 是否强制保存（用于定期保存）
        """
        # 决定保存目录名
        if force_save:
            checkpoint_dir = os.path.join(self.config.training.output_dir, f"checkpoint-epoch-{epoch}")
            logger.info(f"Force saving checkpoint at epoch {epoch} (2 epochs since last save)")
        else:
            checkpoint_dir = os.path.join(self.config.training.output_dir, f"checkpoint-{epoch}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        if self.config.lora.use_lora and PEFT_AVAILABLE:
            # 保存LoRA权重
            self.model.save_pretrained(checkpoint_dir)
        else:
            # 保存完整模型
            self.model.save_pretrained(checkpoint_dir)
        
        # 保存处理器
        self.processor.save_pretrained(checkpoint_dir)
        
        # 保存训练状态
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'mae': mae,
            'config': self.config,
            'is_force_save': force_save
        }, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # 更新上次保存的epoch
        self.last_saved_epoch = epoch
    


    def train(self):
        """主训练循环 - 添加定期保存功能"""
        logger.info("Starting training...")
        logger.info("Will save checkpoint every 2 epochs if no improvement")
        
        for epoch in range(self.config.training.num_epochs):
            current_epoch = epoch + 1
            logger.info(f"Epoch {current_epoch}/{self.config.training.num_epochs}")
            
            # 训练
            self.train_epoch(current_epoch)
            
            # 评估
            mae, predictions = self.evaluate()
            
            # 检查是否需要保存模型
            should_save = False
            is_force_save = False
            
            # 情况1：性能提升，保存最佳模型
            if mae < self.best_metric:
                self.best_metric = mae
                should_save = True
                logger.info(f"New best MAE: {mae:.2f}")
            
            # 情况2：已经2个epoch没有保存了，强制保存
            elif current_epoch - self.last_saved_epoch >= 2:
                should_save = True
                is_force_save = True
                logger.info(f"No improvement for 2 epochs, forcing checkpoint save")
            
            # 执行保存
            if should_save:
                self.save_checkpoint(current_epoch, mae, force_save=is_force_save)
                
                # 保存预测结果
                if self.config.evaluation.save_predictions:
                    pred_path = os.path.join(
                        self.config.training.output_dir, 
                        f"predictions_epoch{current_epoch}.json"
                    )
                    with open(pred_path, 'w') as f:
                        json.dump(predictions, f, indent=2)
            
            logger.info(f"Epoch {current_epoch} - MAE: {mae:.2f} (best: {self.best_metric:.2f})")
            logger.info(f"Last saved: epoch {self.last_saved_epoch}")
        
        logger.info(f"Training completed. Best MAE: {self.best_metric:.2f}")


def check_environment():
    """检查环境配置"""
    logger.info("=== Environment Check ===")
    
    # 检查CUDA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA available: {gpu_count} GPU(s)")
        for i in range(gpu_count):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i} memory: {memory:.1f} GB")
    else:
        logger.error("CUDA not available. Training will be very slow on CPU.")
        return False
    
    # 检查依赖
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("Transformers not installed. Run: pip install transformers")
        return False
    
    # 检查PEFT
    if PEFT_AVAILABLE:
        logger.info("PEFT available for LoRA training")
    else:
        logger.warning("PEFT not available. Install with: pip install peft")
    
    # 检查数据文件
    if not os.path.exists("train.json"):
        logger.error("train.json not found in current directory")
        return False
    
    if not os.path.exists("Images"):
        logger.error("Images directory not found")
        return False
    
    # 检查模型路径（可选，因为我们添加了自动下载功能）
    model_path = "/home/ma-user/work/Problem_Point_Position_train/models/blip2-opt-2.7b"
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}. Will attempt to download from Hugging Face.")
    
    logger.info("All checks passed!")
    return True


def main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Train BLIP2 for geometry point detection")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    # —— 1. 拼 timestamp —— 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # —— 2. 构造最终输出根目录 output/<timestamp> —— 
    final_output = os.path.join(args.output_dir, timestamp)
    os.makedirs(final_output, exist_ok=True)
    logger.info(f"所有结果都将保存在: {final_output}")

    # —— 3. 初始化 config，只做一次 —— 
    config = default_config
    config.experiment_name = f"blip2_point_detection_{timestamp}"
    config.training.output_dir = final_output
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.lora.use_lora = not args.no_lora
    config.training.seed = args.seed

    # 打印配置
    logger.info("\nTraining Configuration:")
    logger.info(f"  - Batch size: {config.training.batch_size}")
    logger.info(f"  - Epochs: {config.training.num_epochs}")
    logger.info(f"  - Learning rate: {config.training.learning_rate}")
    logger.info(f"  - Use LoRA: {config.lora.use_lora}")
    logger.info(f"  - Output dir: {config.training.output_dir}")

    # 保存 config.json
    config_path = os.path.join(final_output, "config.json")
    with open(config_path, 'w') as f:
        json.dump({
            'model': config.model.__dict__,
            'data': config.data.__dict__,
            'training': config.training.__dict__,
            'lora': config.lora.__dict__,
            'evaluation': config.evaluation.__dict__,
            'experiment_name': config.experiment_name
        }, f, indent=2)
    logger.info(f"Configuration saved to: {config_path}")
    
    # 创建训练器
    try:
        trainer = PointDetectionTrainer(config)
        
        if args.eval_only:
            # 仅评估模式
            logger.info("Running evaluation only...")
            mae, predictions = trainer.evaluate()
            logger.info(f"Evaluation MAE: {mae:.2f}")
            
            # 保存预测结果
            pred_path = os.path.join(config.training.output_dir, "eval_predictions.json")
            with open(pred_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"Predictions saved to: {pred_path}")
        else:
            # 训练模式
            trainer.train()
            
        logger.info("\nTraining/Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    main()
    

# 使用说明:
# 1. 基础训练（推荐配置）:
#    python train_blip.py
#
# 2. 自定义参数:
#    python train_blip.py --batch_size 1 --epochs 20 --lr 5e-6
#
# 3. 不使用LoRA（需要更多显存）:
#    python train_blip.py --no_lora --batch_size 1
#
# 4. 仅评估:
#    python train_blip.py --eval_only --checkpoint ./output/checkpoint-best
#
# 5. 从检查点恢复训练:
#    python train_blip.py --checkpoint ./output/checkpoint-5