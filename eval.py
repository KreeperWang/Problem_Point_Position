#!/usr/bin/env python
"""
BLIP2-based Geometry Point Detection Evaluation Script
使用训练好的BLIP2模型进行几何点检测
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
from typing import Optional
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')

from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration,
)

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class CoordinateEncoder(nn.Module):
    """增强版坐标编码器 - 使用正弦位置编码"""
    
    def __init__(self, hidden_size, max_resolution=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_resolution = max_resolution
        
        # 使用正弦位置编码而不是可学习的embedding
        self.temperature = 10000
        
    def forward(self, features, height, width):
            """
            添加2D正弦位置编码到特征
            features: [batch_size, seq_len, hidden_size]
            """
            batch_size = features.size(0)
            device = features.device
            
            # 创建归一化的坐标网格 [0, 1]
            y_embed = torch.arange(height, device=device).float() / height
            x_embed = torch.arange(width, device=device).float() / width
            
            y_embed = y_embed.unsqueeze(1).repeat(1, width)
            x_embed = x_embed.unsqueeze(0).repeat(height, 1)
            
            # 展平并扩展维度
            y_embed = y_embed.reshape(-1, 1)  # [H*W, 1]
            x_embed = x_embed.reshape(-1, 1)  # [H*W, 1]
            
            # 计算正弦位置编码
            # 为了处理任意的hidden_size，我们需要更灵活的方法
            half_dim = self.hidden_size // 2
            dim_t = torch.arange(half_dim, device=device).float()
            dim_t = self.temperature ** (2 * (dim_t // 2) / half_dim)
            
            # 扩展坐标以匹配编码维度
            pos_x = x_embed / dim_t[:half_dim//2]  # 只取前half_dim//2个维度
            pos_y = y_embed / dim_t[:half_dim//2]
            
            # 应用sin和cos
            if half_dim % 2 == 0:
                # 偶数情况
                pos_x_encoded = torch.cat([
                    pos_x.sin(), 
                    pos_x.cos()
                ], dim=1)  # [H*W, half_dim]
                pos_y_encoded = torch.cat([
                    pos_y.sin(), 
                    pos_y.cos()
                ], dim=1)  # [H*W, half_dim]
            else:
                # 奇数情况 - 需要特殊处理
                pos_x_encoded = torch.cat([
                    pos_x.sin(), 
                    pos_x.cos(),
                    torch.zeros(pos_x.size(0), 1, device=device)  # 填充一个0
                ], dim=1)[:, :half_dim]  # 截取到正确的维度
                pos_y_encoded = torch.cat([
                    pos_y.sin(), 
                    pos_y.cos(),
                    torch.zeros(pos_y.size(0), 1, device=device)
                ], dim=1)[:, :half_dim]
            
            # 合并x和y编码
            pos_emb = torch.cat([pos_y_encoded, pos_x_encoded], dim=1)  # [H*W, hidden_size]
            
            # 如果维度仍然不匹配（奇数hidden_size的情况），进行填充
            if pos_emb.size(1) < self.hidden_size:
                padding = torch.zeros(pos_emb.size(0), self.hidden_size - pos_emb.size(1), device=device)
                pos_emb = torch.cat([pos_emb, padding], dim=1)
            elif pos_emb.size(1) > self.hidden_size:
                pos_emb = pos_emb[:, :self.hidden_size]
            
            # 扩展到batch维度
            pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # 如果序列长度不匹配，进行插值
            if features.size(1) != pos_emb.size(1):
                pos_emb = F.interpolate(
                    pos_emb.transpose(1, 2),
                    size=features.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            return features + pos_emb


class GeometricRelationEncoder(nn.Module):
    """几何关系编码器 - 编码点之间的空间关系"""
    
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 关系类型embedding
        self.relation_embeddings = nn.Embedding(4, hidden_size // 4)  # 4种基本关系
        
        # 距离编码
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 4)
        )
        
        # 角度编码
        self.angle_encoder = nn.Sequential(
            nn.Linear(2, hidden_size // 4),  # sin和cos
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 4)
        )
        
        # 相对位置编码
        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 4)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True
        )
        
    def compute_geometric_features(self, points: torch.Tensor):
        """
        计算点之间的几何特征
        points: [batch_size, num_points, 2]
        """
        batch_size, num_points, _ = points.shape
        device = points.device
        
        # 扩展维度用于计算成对关系
        points_i = points.unsqueeze(2)  # [B, N, 1, 2]
        points_j = points.unsqueeze(1)  # [B, 1, N, 2]
        
        # 计算距离矩阵
        distances = torch.norm(points_i - points_j, dim=-1)  # [B, N, N]
        
        # 归一化距离（避免过大的值）
        max_dist = distances.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        normalized_distances = distances / (max_dist + 1e-6)
        
        # 计算相对位置
        relative_pos = points_i - points_j  # [B, N, N, 2]
        
        # 计算角度（使用atan2）
        angles = torch.atan2(relative_pos[..., 1], relative_pos[..., 0])  # [B, N, N]
        
        # 将角度转换为sin和cos（更稳定的表示）
        angle_sin = torch.sin(angles)
        angle_cos = torch.cos(angles)
        
        return normalized_distances, relative_pos, angle_sin, angle_cos
    
    def forward(self, features: torch.Tensor, points: Optional[torch.Tensor] = None):
        """
        增强特征with几何关系
        features: [batch_size, num_points, hidden_size]
        points: [batch_size, num_points, 2] - 如果提供，用于计算真实的几何关系
        """
        batch_size, num_points, _ = features.shape
        device = features.device
        
        if points is not None:
            # 使用真实的点坐标计算几何关系
            distances, relative_pos, angle_sin, angle_cos = self.compute_geometric_features(points)
            
            # 编码各种几何特征
            dist_features = self.distance_encoder(distances.unsqueeze(-1))  # [B, N, N, H/4]
            angle_features = self.angle_encoder(torch.stack([angle_sin, angle_cos], dim=-1))  # [B, N, N, H/4]
            rel_pos_features = self.relative_pos_encoder(relative_pos)  # [B, N, N, H/4]
            
            # 关系类型（这里使用简单的规则）
            relation_types = torch.zeros(batch_size, num_points, num_points, dtype=torch.long, device=device)
            # 0: self, 1: near, 2: medium, 3: far
            relation_types[distances < 0.2] = 1
            relation_types[(distances >= 0.2) & (distances < 0.5)] = 2
            relation_types[distances >= 0.5] = 3
            # 对角线设为0（自身关系）
            for i in range(num_points):
                relation_types[:, i, i] = 0
            
            rel_embeddings = self.relation_embeddings(relation_types)  # [B, N, N, H/4]
            
            # 组合所有几何特征
            geometric_features = torch.cat([
                dist_features,
                angle_features,
                rel_pos_features,
                rel_embeddings
            ], dim=-1)  # [B, N, N, H]
            
            # 聚合来自所有其他点的几何信息
            aggregated_features = geometric_features.mean(dim=2)  # [B, N, H]
            
        else:
            # 如果没有提供点坐标，使用可学习的关系
            aggregated_features = torch.zeros_like(features)
        
        # 融合几何特征和原始特征
        enhanced_features = self.fusion(features + aggregated_features)
        
        # 应用自注意力进一步提炼特征
        attn_output, _ = self.self_attention(enhanced_features, enhanced_features, enhanced_features)
        
        return enhanced_features + attn_output

class PointDetectionHead(nn.Module):
    """点检测头"""
    def __init__(self, hidden_size, num_points_max=26):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_points_max = num_points_max
        self.point_embeddings = nn.Embedding(num_points_max, hidden_size)
        self.coord_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
            nn.Sigmoid()
        )
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, visual_features, point_indices):
        batch_size, num_points = point_indices.shape
        point_embeds = self.point_embeddings(point_indices)
        visual_features_expanded = visual_features.unsqueeze(1).expand(-1, num_points, -1)
        combined_features = torch.cat([visual_features_expanded, point_embeds], dim=-1)
        coords = self.coord_predictor(combined_features)
        confidence_logits = self.confidence_predictor(combined_features)
        return coords, confidence_logits.squeeze(-1)


class GeometricBlip2Model(nn.Module):
    """增强的BLIP2模型"""
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        
        # 几何模块
        self.coordinate_encoder = CoordinateEncoder(hidden_size)
        self.geometric_encoder = GeometricRelationEncoder(hidden_size, num_heads=4)
        self.point_detection_head = PointDetectionHead(hidden_size)
        self.visual_projection = nn.Linear(
            base_model.vision_model.config.hidden_size,
            hidden_size
        )



# 先load原模型，再load参数！

class BLIP2GeometryDetector:
    """BLIP2几何点检测器"""
    
    def __init__(self, checkpoint_path, json_path,device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        
        logger.info("=" * 80)
        logger.info("Initializing BLIP2 Geometry Point Detection Model")
        logger.info("=" * 80)
        
        # 检查checkpoint路径
        logger.info(f"Checkpoint path: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            logger.info(f"✓ Checkpoint directory exists")
            checkpoint_files = os.listdir(checkpoint_path)
            logger.info(f"  Files in checkpoint: {checkpoint_files[:5]}...")
        else:
            logger.error(f"✗ Checkpoint directory not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
        
        # 首先尝试从checkpoint目录加载processor，如果失败则使用默认的
        processor_path = checkpoint_path
        try:
            # 检查是否有processor的配置文件
            if not os.path.exists(os.path.join(processor_path, 'preprocessor_config.json')):
                # 尝试父目录
                parent_dir = os.path.dirname(checkpoint_path)
                if os.path.exists(os.path.join(parent_dir, 'preprocessor_config.json')):
                    processor_path = parent_dir
                else:
                    raise FileNotFoundError("Processor config not found")
            
            self.processor = Blip2Processor.from_pretrained(processor_path)
            logger.info(f"✓ Loaded BLIP2 Processor from {processor_path}")
        except Exception as e:
            # 使用默认的BLIP2 processor
            default_model_name = "Salesforce/blip2-opt-2.7b"
            self.processor = Blip2Processor.from_pretrained(default_model_name)
            logger.info(f"✓ Loaded BLIP2 Processor from {default_model_name} (default)")
        
        default_model_name = "Salesforce/blip2-opt-2.7b"
        logger.info(f"Loading base BLIP2 model from {default_model_name}")
        self.base_model = Blip2ForConditionalGeneration.from_pretrained(
            "/home/ma-user/work/Problem_Point_Position_train/models/blip2-opt-2.7b", #self.config.model.model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        )
        logger.info("✓ Loaded BLIP2 Base Model")
        
        # 创建增强模型
        hidden_size = self.base_model.config.qformer_config.hidden_size
        self.model = GeometricBlip2Model(self.base_model, hidden_size)
        
        # 尝试加载checkpoint中的权重
        checkpoint_file = os.path.join(checkpoint_path, 'checkpoint.pt')
        if os.path.exists(checkpoint_file):
            try:
                logger.info(f"Loading trained weights from {checkpoint_file}")
                checkpoint = torch.load(checkpoint_file, map_location=self.device)
                
                # 加载base_model的权重
                if 'base_model_state_dict' in checkpoint:
                    self.base_model.load_state_dict(checkpoint['base_model_state_dict'], strict=False)
                    logger.info("✓ Loaded trained base model weights")
                
                # 加载训练的模型权重
                if 'model_state_dict' in checkpoint:
                    # 这些权重包含了几何模块的参数
                    for name, param in checkpoint['model_state_dict'].items():
                        if 'base_model' in name:
                            # 移除'base_model.'前缀并加载到base_model
                            param_name = name.replace('base_model.', '')
                            try:
                                self.base_model.state_dict()[param_name].copy_(param)
                            except:
                                pass
                    logger.info("✓ Loaded trained model weights")
                
                # 显示checkpoint信息
                if 'epoch' in checkpoint:
                    logger.info(f"  - Checkpoint epoch: {checkpoint['epoch']}")
                if 'best_metric' in checkpoint:
                    logger.info(f"  - Best metric: {checkpoint['best_metric']:.4f}")
                    
            except Exception as e:
                logger.warning(f"Could not load checkpoint weights: {e}")
                logger.info("Continuing with base model...")
        else:
            logger.warning(f"Checkpoint file not found: {checkpoint_file}")
            logger.info("Using base BLIP2 model without fine-tuned weights")
        

        self.model.to(self.device)
        self.model.eval()

        with open(json_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"\nModel Configuration:")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Model dtype: {next(self.model.parameters()).dtype}")
        
        self.template_scales = [0.6, 0.8, 1.0, 1.2, 1.4]
        self.search_radius = 50
        self.point_offset_range = 30
        
        logger.info("\nModel loaded successfully! Ready for inference.")
        logger.info("=" * 80)
        
    def generate_prompt(self, point_names):
        """生成检测prompt"""
        points_str = ', '.join(point_names)
        prompt = f"""<image>
Task: Detect the coordinates of geometric points.
Points to detect: {points_str}
Output the coordinates for each point in JSON format.

Instruction: Analyze the geometry diagram and identify the exact pixel coordinates of the labeled points."""
        return prompt
    
    def _create_letter_template(self, letter, scale=1.0):
        base_size = int(40 * scale)
        template = np.ones((base_size, base_size), dtype=np.uint8) * 255
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9 * scale
        thickness = max(1, int(2 * scale))
        
        (text_width, text_height), _ = cv2.getTextSize(letter, font, font_scale, thickness)
        x = (base_size - text_width) // 2
        y = (base_size + text_height) // 2
        
        cv2.putText(template, letter, (x, y), font, font_scale, 0, thickness)
        
        return template
    
    def _find_letter_position(self, img, letter):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        best_match = None
        best_score = -1
        
        for scale in self.template_scales:
            template = self._create_letter_template(letter, scale)
            
            try:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score and max_val > 0.5:
                    best_score = max_val
                    best_match = (
                        max_loc[0] + template.shape[1] // 2,
                        max_loc[1] + template.shape[0] // 2
                    )
            except:
                continue
        
        return best_match
    
    def _find_point_near_letter(self, img, letter_pos):
        x, y = letter_pos
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        search_size = self.search_radius
        x_start = max(0, x - search_size)
        x_end = min(w, x + search_size)
        y_start = max(0, y - search_size)
        y_end = min(h, y + search_size)
        
        search_region = gray[y_start:y_end, x_start:x_end]
        
        _, binary = cv2.threshold(search_region, 50, 255, cv2.THRESH_BINARY_INV)
        
        circles = cv2.HoughCircles(
            search_region,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=20,
            minRadius=2,
            maxRadius=15
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            min_dist = float('inf')
            best_circle = None
            
            for circle in circles[0, :]:
                cx, cy = circle[0], circle[1]
                actual_x = x_start + cx
                actual_y = y_start + cy
                
                dist = np.sqrt((actual_x - x)**2 + (actual_y - y)**2)
                
                if dist < min_dist and dist < self.point_offset_range:
                    min_dist = dist
                    best_circle = (actual_x, actual_y)
            
            if best_circle:
                return best_circle
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_dist = float('inf')
        best_point = None
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if 10 < area < 200:
                cx = centroids[i][0] + x_start
                cy = centroids[i][1] + y_start
                
                dist = np.sqrt((cx - x)**2 + (cy - y)**2)
                
                if dist < min_dist and dist < self.point_offset_range:
                    min_dist = dist
                    best_point = (cx, cy)
        
        return best_point
    
    def detect_single_image(self, image_path,  verbose=True):
        """检测单张图片"""
        #expected_points = list(self.test_data[image_id].keys())
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        expected_points = list(self.test_data[image_id].keys())
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return {}
        
        pil_img = Image.open(image_path).convert('RGB')
        
        # 生成prompt
        prompt = self.generate_prompt(expected_points)
        if verbose:
            logger.info("\nGenerated Prompt:")
            logger.info("-" * 50)
            logger.info(prompt)
            logger.info("-" * 50)

        print("prompt:"+prompt)
        
        logger.info("\nRunning BLIP2 inference...")
        
        with torch.no_grad():
            inputs = self.processor(
                images=pil_img,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            generated_ids = self.model.base_model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=3,
                temperature=0.7,
                do_sample=False
            )
            
            # 解码
            generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            if verbose:
                logger.info(f"Model raw output: {generated_text[:200]}...")
        
        final_points = {}
        for letter in expected_points:
            letter_pos = self._find_letter_position(img, letter)
            
            if letter_pos is not None:
                point_pos = self._find_point_near_letter(img, letter_pos)
                if point_pos is not None:
                    final_points[letter] = [float(point_pos[0]), float(point_pos[1])]
        
        if verbose:
            logger.info("\nFinal Detection Results:")
            logger.info("-" * 50)
            output_text = json.dumps(final_points, indent=2)
            logger.info(output_text)
            logger.info("-" * 50)
            
            logger.info(f"\nDetection Statistics:")
            logger.info(f"  - Expected points: {len(expected_points)}")
            logger.info(f"  - Detected points: {len(final_points)}")
            logger.info(f"  - Detection rate: {len(final_points)/len(expected_points)*100:.1f}%")
            
            if len(final_points) < len(expected_points):
                missing = set(expected_points) - set(final_points.keys())
                logger.warning(f"  - Missing points: {', '.join(missing)}")
        
        return final_points
    
    def evaluate_dataset(self, test_json_path, image_dir, output_file="predictions.json"):
        """评估整个数据集"""
        with open(test_json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        logger.info(f"\nEvaluating on {len(test_data)} test images...")
        logger.info("=" * 80)
        
        results = {}
        
        # 处理每张图像
        for idx, image_id in enumerate(tqdm(test_data.keys(), desc="Processing images")):
            image_path = os.path.join(image_dir, f"{image_id}.png")
            expected_points = list(test_data[image_id].keys())
            
            # 第一张图片显示详细信息
            verbose = (idx == 0)
            
            try:
                points = self.detect_single_image(image_path, expected_points, verbose=verbose)
                results[image_id] = points
            except Exception as e:
                logger.error(f"Error processing {image_id}: {str(e)}")
                results[image_id] = {}
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResults saved to: {output_file}")
        
        # 计算统计
        total_expected = sum(len(points) for points in test_data.values())
        total_detected = sum(len(points) for points in results.values())
        
        logger.info("\nOverall Statistics:")
        logger.info(f"  - Total expected points: {total_expected}")
        logger.info(f"  - Total detected points: {total_detected}")
        logger.info(f"  - Overall detection rate: {total_detected/total_expected*100:.2f}%")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="BLIP2-based Geometry Point Detection Evaluation"
    )
    
    # 模型参数
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/home/ma-user/work/Problem_Point_Position_train/output_memory_opt/20250725_033022/checkpoint-epoch-12',
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--test_json',
        type=str,
        default='/home/ma-user/work/Problem_Point_Position_train/test.json',
        help='Path to test.json file (batch mode)'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='/home/ma-user/work/Problem_Point_Position_train/Images',
        help='Directory containing images (batch mode)'
    )
    
    parser.add_argument(
        '--single_image',
        type=str,
        default='/home/ma-user/work/Problem_Point_Position_train/Images/4231.png',
        help='Path to single image for testing'
    )
    # parser.add_argument(
    #     '--points',
    #     type=str,
    #     help='Points to detect in single image mode (e.g., "A,B,C")'
    # )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Activate batch evaluation mode'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.json',
        help='Output file path'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting BLIP2 Geometry Point Detection System")
    detector = BLIP2GeometryDetector(
        args.checkpoint,
        args.test_json,
        args.device
    )

    if not args.batch:
        #points_list = [p.strip() for p in args.points.split(',')]
        #logger.info(f"Detecting points {points_list} in image: {args.single_image}")

        results = detector.detect_single_image(args.single_image, verbose=True)
        # 保存结果
        image_id = os.path.basename(args.single_image).split('.')[0]
        output_data = {image_id: results}
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {args.output}")

    else:
        results = detector.evaluate_dataset(args.test_json, args.image_dir, args.output)
        # 如果存在评估脚本，运行评估
        # if os.path.exists("cal_acc.py"):
        #     logger.info("\nRunning accuracy evaluation...")
        #     try:
        #         from cal_acc import PointAccuracyCalculator
        #         calculator = PointAccuracyCalculator(
        #             gt_file=args.test_json,
        #             pred_file=args.output
        #         )
        #         calculator.calculate_all_metrics()
        #         calculator.print_results()
        #     except Exception as e:
        #         logger.error(f"Error running evaluation: {str(e)}")


if __name__ == "__main__":
    main()