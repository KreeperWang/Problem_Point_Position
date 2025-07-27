"""
Enhanced utility functions with geometric encoding modules
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import logging
import math

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
    """专门的点检测头 - 直接预测坐标"""
    
    def __init__(self, hidden_size, num_points_max=26):  # 最多26个点(A-Z)
        super().__init__()
        self.hidden_size = hidden_size
        self.num_points_max = num_points_max
        
        # 点名称embedding
        self.point_embeddings = nn.Embedding(num_points_max, hidden_size)
        
        # 坐标预测网络
        self.coord_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),  # 输出x, y坐标
            nn.Sigmoid()  # 归一化到[0, 1]
        )
        
        # 置信度预测 - 注意：不使用Sigmoid，因为将使用BCEWithLogitsLoss
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
            # 移除了Sigmoid层
        )
        
    def forward(self, visual_features, point_indices):
        """
        预测点的坐标
        visual_features: [batch_size, hidden_size] - 汇聚的视觉特征
        point_indices: [batch_size, num_points] - 要预测的点的索引
        """
        batch_size, num_points = point_indices.shape
        
        # 获取点embedding
        point_embeds = self.point_embeddings(point_indices)  # [B, N, H]
        
        # 扩展视觉特征
        visual_features_expanded = visual_features.unsqueeze(1).expand(-1, num_points, -1)
        
        # 组合特征
        combined_features = torch.cat([visual_features_expanded, point_embeds], dim=-1)
        
        # 预测坐标和置信度logits
        coords = self.coord_predictor(combined_features)  # [B, N, 2]
        confidence_logits = self.confidence_predictor(combined_features)  # [B, N, 1]
        
        return coords, confidence_logits.squeeze(-1)


class GeometricAwareLoss(nn.Module):
    """几何感知损失函数"""
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # 坐标损失权重
        self.beta = beta    # 关系损失权重
        self.gamma = gamma  # 置信度损失权重
        
    def forward(self, pred_coords, gt_coords, pred_confidence_logits, valid_mask):
        """
        计算损失
        pred_coords: [batch_size, num_points, 2] - 预测的归一化坐标
        gt_coords: [batch_size, num_points, 2] - 真实的归一化坐标
        pred_confidence_logits: [batch_size, num_points] - 预测的置信度logits（未经过sigmoid）
        valid_mask: [batch_size, num_points] - 有效点的mask
        """
        # 坐标损失（L1 + L2）
        coord_l1_loss = F.l1_loss(pred_coords * valid_mask.unsqueeze(-1), 
                                  gt_coords * valid_mask.unsqueeze(-1), 
                                  reduction='sum') / valid_mask.sum()
        
        coord_l2_loss = F.mse_loss(pred_coords * valid_mask.unsqueeze(-1), 
                                   gt_coords * valid_mask.unsqueeze(-1), 
                                   reduction='sum') / valid_mask.sum()
        
        coord_loss = coord_l1_loss + 0.5 * coord_l2_loss
        
        # 关系损失（保持点之间的相对距离）
        pred_distances = torch.cdist(pred_coords, pred_coords)
        gt_distances = torch.cdist(gt_coords, gt_coords)
        
        # 只计算有效点之间的距离
        valid_pairs = valid_mask.unsqueeze(-1) * valid_mask.unsqueeze(-2)
        relation_loss = F.l1_loss(pred_distances * valid_pairs, 
                                  gt_distances * valid_pairs,
                                  reduction='sum') / valid_pairs.sum()
        
        # 置信度损失（使用BCEWithLogitsLoss，它内部会应用sigmoid）
        confidence_loss = F.binary_cross_entropy_with_logits(pred_confidence_logits, valid_mask.float())
        
        # 总损失
        total_loss = self.alpha * coord_loss + self.beta * relation_loss + self.gamma * confidence_loss
        
        return total_loss, {
            'coord_loss': coord_loss.item(),
            'relation_loss': relation_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'total_loss': total_loss.item()
        }


# 保留原有的辅助函数...
def calculate_point_relationships(points: Dict[str, List[float]]) -> Dict[str, any]:
    """计算点之间的几何关系"""
    if not points or len(points) < 2:
        return {}
    
    point_names = list(points.keys())
    coords = np.array([points[name] for name in point_names])
    n_points = len(point_names)
    
    # 计算距离矩阵
    distances = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(i+1, n_points):
            dist = np.linalg.norm(coords[i] - coords[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    # 识别可能的形状
    shape_type = identify_shape(coords)
    
    # 计算重心
    centroid = coords.mean(axis=0)
    
    # 计算包围盒
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    bbox = {
        'min': min_coords.tolist(),
        'max': max_coords.tolist(),
        'width': float(max_coords[0] - min_coords[0]),
        'height': float(max_coords[1] - min_coords[1])
    }
    
    return {
        'n_points': n_points,
        'distances': distances.tolist(),
        'shape_type': shape_type,
        'centroid': centroid.tolist(),
        'bbox': bbox,
        'point_names': point_names
    }


def identify_shape(coords: np.ndarray) -> str:
    """识别可能的几何形状"""
    n = len(coords)
    
    if n < 3:
        return "line" if n == 2 else "point"
    elif n == 3:
        return "triangle"
    elif n == 4:
        # 检查是否为矩形/正方形
        distances = []
        for i in range(4):
            for j in range(i+1, 4):
                distances.append(np.linalg.norm(coords[i] - coords[j]))
        distances.sort()
        
        # 如果有两对相等的边，可能是矩形
        if abs(distances[0] - distances[1]) < 1e-2 and abs(distances[4] - distances[5]) < 1e-2:
            if abs(distances[0] - distances[4]) < 1e-2:
                return "square"
            else:
                return "rectangle"
        else:
            return "quadrilateral"
    else:
        return "polygon"


# 保留其他辅助函数...

def visualize_predictions(
    predictions: Dict[str, Dict],
    image_dir: str,
    output_dir: str,
    num_samples: int = 20
):
    """可视化预测结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 随机选择样本
    sample_ids = list(predictions.keys())
    if len(sample_ids) > num_samples:
        sample_ids = random.sample(sample_ids, num_samples)
    
    for img_id in sample_ids:
        try:
            # 加载图像
            image_path = os.path.join(image_dir, f"{img_id}.png")
            if not os.path.exists(image_path):
                continue
            
            image = Image.open(image_path).convert('RGB')
            
            # 获取预测和真实坐标
            pred_data = predictions[img_id]
            pred_points = pred_data['predictions']
            gt_points = pred_data['ground_truth']
            
            # 创建可视化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # 显示原图
            ax1.imshow(image)
            ax1.set_title('Ground Truth', fontsize=14)
            ax1.axis('off')
            
            # 绘制真实点
            for name, (x, y) in gt_points.items():
                circle = Circle((x, y), 8, color='green', fill=True, alpha=0.7)
                ax1.add_patch(circle)
                ax1.text(x+10, y-10, name, color='green', fontsize=12, weight='bold')
            
            # 显示预测图
            ax2.imshow(image)
            ax2.set_title('Predictions', fontsize=14)
            ax2.axis('off')
            
            # 绘制预测点
            for name, (x, y) in pred_points.items():
                circle = Circle((x, y), 8, color='red', fill=True, alpha=0.7)
                ax2.add_patch(circle)
                ax2.text(x+10, y-10, name, color='red', fontsize=12, weight='bold')
                
                # 如果有对应的真实点，画连线
                if name in gt_points:
                    gt_x, gt_y = gt_points[name]
                    ax2.plot([x, gt_x], [y, gt_y], 'b--', alpha=0.5, linewidth=2)
                    
                    # 计算误差
                    error = np.sqrt((x - gt_x)**2 + (y - gt_y)**2)
                    ax2.text((x + gt_x)/2, (y + gt_y)/2 - 5, f'{error:.1f}px', 
                            color='blue', fontsize=10, ha='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # 计算总体MAE
            errors = []
            for name in gt_points:
                if name in pred_points:
                    gt = gt_points[name]
                    pred = pred_points[name]
                    error = np.sqrt((gt[0] - pred[0])**2 + (gt[1] - pred[1])**2)
                    errors.append(error)
            
            mae = np.mean(errors) if errors else 0
            fig.suptitle(f'Image {img_id} - MAE: {mae:.2f}px', fontsize=16)
            
            # 保存图像
            output_path = os.path.join(output_dir, f'vis_{img_id}.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved visualization: {output_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing {img_id}: {str(e)}")
            continue


def analyze_errors(predictions: Dict[str, Dict], output_path: str):
    """分析预测误差并保存报告"""
    all_errors = []
    point_errors = {}
    
    for img_id, data in predictions.items():
        gt_points = data['ground_truth']
        pred_points = data['predictions']
        
        for name in gt_points:
            if name in pred_points:
                gt = gt_points[name]
                pred = pred_points[name]
                error = np.sqrt((gt[0] - pred[0])**2 + (gt[1] - pred[1])**2)
                
                all_errors.append(error)
                if name not in point_errors:
                    point_errors[name] = []
                point_errors[name].append(error)
    
    if not all_errors:
        logger.warning("No valid predictions for error analysis")
        return
    
    # 计算统计信息
    stats = {
        'total_predictions': len(all_errors),
        'mean_error': np.mean(all_errors),
        'std_error': np.std(all_errors),
        'median_error': np.median(all_errors),
        'min_error': np.min(all_errors),
        'max_error': np.max(all_errors),
        'percentiles': {
            '25': np.percentile(all_errors, 25),
            '50': np.percentile(all_errors, 50),
            '75': np.percentile(all_errors, 75),
            '90': np.percentile(all_errors, 90),
            '95': np.percentile(all_errors, 95),
        }
    }
    
    # 写入报告
    with open(output_path, 'w') as f:
        f.write("Geometry Point Detection Error Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Overall Statistics:\n")
        f.write(f"Total Predictions: {stats['total_predictions']}\n")
        f.write(f"Mean Error: {stats['mean_error']:.2f} pixels\n")
        f.write(f"Std Error: {stats['std_error']:.2f} pixels\n")
        f.write(f"Median Error: {stats['median_error']:.2f} pixels\n")
        f.write(f"Min Error: {stats['min_error']:.2f} pixels\n")
        f.write(f"Max Error: {stats['max_error']:.2f} pixels\n\n")
        
        f.write("Error Percentiles:\n")
        for p, value in stats['percentiles'].items():
            f.write(f"  {p}th percentile: {value:.2f} pixels\n")
        f.write("\n")
        
        f.write("Per-Point Statistics:\n")
        for name, errors in sorted(point_errors.items()):
            f.write(f"\nPoint {name}:\n")
            f.write(f"  Count: {len(errors)}\n")
            f.write(f"  Mean: {np.mean(errors):.2f} pixels\n")
            f.write(f"  Std: {np.std(errors):.2f} pixels\n")
            f.write(f"  Median: {np.median(errors):.2f} pixels\n")
    
    logger.info(f"Error report saved to: {output_path}")