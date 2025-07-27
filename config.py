"""
Configuration file for BLIP2 training on geometry point detection
Optimized for 16GB GPU (Pnt1)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """模型配置"""
    model_path: str = "/home/ma-user/work/Problem_Point_Position_train/models/blip2-opt-2.7b"
    model_name: str = "blip2-opt-2.7b"
    
    freeze_vit: bool = True  # 冻结视觉编码器以节省显存
    freeze_qformer: bool = False  # Q-Former参与训练
    num_query_token: int = 32  # Query tokens数量
    
    # 生成配置
    max_length: int = 256 #128
    min_length: int = 10
    num_beams: int = 3
    temperature: float = 0.7
    do_sample: bool = False  # 对于坐标任务，使用确定性生成


@dataclass
class DataConfig:
    """数据配置"""
    data_root: str = "/home/ma-user/work/Problem_Point_Position_train"
    train_json: str = "train.json"
    test_json: str = "test.json"
    image_dir: str = "Images"
    
    # 数据处理
    image_size: int = 224 #384  # BLIP2默认输入尺寸
    augmentation: bool = False  # 几何任务不使用数据增强
    
    # 数据集划分
    train_ratio: float = 0.99
    val_ratio: float = 0.01
    
    @property
    def train_json_path(self):
        return os.path.join(self.data_root, self.train_json)
    
    @property
    def test_json_path(self):
        return os.path.join(self.data_root, self.test_json)
    
    @property
    def image_dir_path(self):
        return os.path.join(self.data_root, self.image_dir)


@dataclass
class TrainingConfig:
    # 基础训练参数
    batch_size: int = 1 
    gradient_accumulation_steps: int = 8 
    num_epochs: int = 20
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.05
    weight_decay: float = 0.05
    
    # 优化器
    optimizer: str = "adamw"
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    # 学习率调度
    lr_scheduler_type: str = "cosine"
    
    # 训练策略
    fp16: bool = False 
    gradient_checkpointing: bool = True  # 开启梯度检查点
    
    # 日志和保存
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_mae"
    greater_is_better: bool = False
    
    # 输出目录
    output_dir: str = "./results"
    logging_dir: str = "./logs"
    
    # 设备
    device: str = "cuda"
    n_gpu: int = 1
    
    # 早停
    early_stopping_patience: int = 3
    
    # 种子
    seed: int = 42


@dataclass
class LoRAConfig:
    use_lora: bool = True  # 16GB显卡推荐使用LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",  # 注意力层
        "gate_proj", "up_proj", "down_proj",  # MLP层
    ])


@dataclass
class EvaluationConfig:
    """评估配置"""
    eval_batch_size: int = 1  # 评估时可以使用稍大的batch size
    distance_thresholds: List[float] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    save_predictions: bool = True
    visualize_results: bool = True
    num_visualize_samples: int = 20


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # 实验配置
    experiment_name: str = "blip2_geometry_point_detection"
    wandb_project: Optional[str] = None  # 如果使用wandb
    
    def __post_init__(self):
        """后处理：创建必要的目录"""
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs(self.training.logging_dir, exist_ok=True)
        
        # 更新输出目录
        self.training.output_dir = os.path.join(
            self.training.output_dir, 
            self.experiment_name
        )
        os.makedirs(self.training.output_dir, exist_ok=True)


# 创建16GB GPU优化的默认配置
default_config = Config()

# 快速配置函数
def get_quick_config(use_lora=True, batch_size=2):
    """快速获取配置"""
    config = Config()
    config.lora.use_lora = use_lora
    config.training.batch_size = batch_size
    
    # 调整梯度累积步数以保持有效batch size
    if batch_size == 1:
        config.training.gradient_accumulation_steps = 16
    elif batch_size == 2:
        config.training.gradient_accumulation_steps = 8
    elif batch_size == 4:
        config.training.gradient_accumulation_steps = 4
    
    return config