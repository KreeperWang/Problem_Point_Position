import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os

class PointAccuracyCalculator:
    """
    点坐标准确率计算器
    支持多种主流评测指标
    """
    
    def __init__(self, gt_file: str, pred_file: str, image_size: Optional[Tuple[int, int]] = None):
        """
        初始化计算器
        
        Args:
            gt_file: 真实标注文件路径
            pred_file: 预测结果文件路径  
            image_size: 图像尺寸(width, height)，用于归一化计算
        """
        self.gt_file = gt_file
        self.pred_file = pred_file
        self.image_size = image_size
        
        # 加载数据
        self.gt_data = self._load_json(gt_file)
        self.pred_data = self._load_json(pred_file) if os.path.exists(pred_file) else {}
        
        # 计算结果存储
        self.results = {}
        
    def _load_json(self, file_path: str) -> Dict:
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def euclidean_distance(self, point1: List[float], point2: List[float]) -> float:
        """计算两点间的欧几里得距离"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_distances(self) -> List[float]:
        """
        计算所有匹配点对的距离
        
        Returns:
            距离列表
        """
        distances = []
        matched_points = 0
        total_gt_points = 0
        
        for diagram_id, gt_points in self.gt_data.items():
            total_gt_points += len(gt_points)
            
            if diagram_id not in self.pred_data:
                # 如果预测中没有这个图表，所有点都算作最大误差
                distances.extend([float('inf')] * len(gt_points))
                continue
                
            pred_points = self.pred_data[diagram_id]
            
            for point_name, gt_coord in gt_points.items():
                if point_name in pred_points:
                    pred_coord = pred_points[point_name]
                    dist = self.euclidean_distance(gt_coord, pred_coord)
                    distances.append(dist)
                    matched_points += 1
                else:
                    # 预测中缺失该点
                    distances.append(float('inf'))
        
        print(f"总GT点数: {total_gt_points}")
        print(f"匹配点数: {matched_points}")
        print(f"匹配率: {matched_points/total_gt_points:.2%}")
        
        return distances
    
    def calculate_mae(self, distances: List[float]) -> float:
        """计算平均绝对误差(忽略inf值)"""
        finite_distances = [d for d in distances if d != float('inf')]
        if not finite_distances:
            return float('inf')
        return np.mean(finite_distances)
    
    def calculate_accuracy_at_threshold(self, distances: List[float], threshold: float) -> float:
        """计算在给定阈值下的准确率"""
        correct_predictions = sum(1 for d in distances if d <= threshold)
        return correct_predictions / len(distances) if distances else 0.0
    
    def calculate_normalized_error(self, distances: List[float]) -> List[float]:
        """计算归一化点误差（距离除以图像对角线）"""
        if not self.image_size:
            raise ValueError("image_size未指定，无法归一化")
        width, height = self.image_size
        diag_len = np.sqrt(width ** 2 + height ** 2)
        return [d / diag_len if np.isfinite(d) else 1.0 for d in distances]
    
    def calculate_relative_point_distance_error(self) -> float:
        """
        计算所有点对之间的预测距离与真实距离的平均误差
        
        Returns:
            所有图像的平均相对点间距离误差
        """
        total_error = 0.0
        count = 0
        for diagram_id, gt_points in self.gt_data.items():
            if diagram_id not in self.pred_data:
                continue
            pred_points = self.pred_data[diagram_id]
            gt_names = list(gt_points.keys())
            # 只考虑预测和gt都存在的点名
            common_names = [n for n in gt_names if n in pred_points]
            for i in range(len(common_names)):
                for j in range(i + 1, len(common_names)):
                    name_i, name_j = common_names[i], common_names[j]
                    gt_dist = self.euclidean_distance(gt_points[name_i], gt_points[name_j])
                    pred_dist = self.euclidean_distance(pred_points[name_i], pred_points[name_j])
                    total_error += abs(gt_dist - pred_dist)
                    count += 1
        return total_error / count if count > 0 else float('inf')
    
    def calculate_all_metrics(self, thresholds: List[float] = [5, 10, 15, 20]) -> Dict:
        """
        计算所有评测指标
        
        Args:
            thresholds: 准确率计算的阈值列表
            
        Returns:
            包含所有指标的字典
        """
        print("正在计算点坐标准确率指标...")
        
        # 计算距离
        distances = self.calculate_distances()
        
        # 基本统计
        finite_distances = [d for d in distances if d != float('inf')]
        
        results = {
            'total_points': len(distances),
            'matched_points': len(finite_distances),
            'match_rate': len(finite_distances) / len(distances) if distances else 0,
        }
        
        if finite_distances:
            # 距离统计
            results.update({
                'mean_distance': np.mean(finite_distances),
                'median_distance': np.median(finite_distances),
                'std_distance': np.std(finite_distances),
                'min_distance': np.min(finite_distances),
                'max_distance': np.max(finite_distances),
            })
            
            # MAE
            results['mae'] = self.calculate_mae(distances)
            
            # 不同阈值下的准确率
            for threshold in thresholds:
                acc = self.calculate_accuracy_at_threshold(distances, threshold)
                results[f'accuracy@{threshold}px'] = acc
            
            # 归一化误差
            if self.image_size:
                normalized_errors = self.calculate_normalized_error(distances)
                finite_normalized = [e for e in normalized_errors if e < 1.0]
                if finite_normalized:
                    results['mean_normalized_error'] = np.mean(finite_normalized)
                    results['median_normalized_error'] = np.median(finite_normalized)
        
        # 相对点间距离误差
        results['relative_point_distance_error'] = self.calculate_relative_point_distance_error()
        
        self.results = results
        return results
    
    def print_results(self):
        """打印计算结果"""
        if not self.results:
            print("请先运行calculate_all_metrics()方法")
            return
        
        print("\n" + "="*60)
        print("           点坐标准确率评测结果")
        print("="*60)
        
        print(f"\n📊 基本统计:")
        print(f"  总点数: {self.results['total_points']}")
        print(f"  匹配点数: {self.results['matched_points']}")
        print(f"  匹配率: {self.results['match_rate']:.2%}")
        
        if 'mean_distance' in self.results:
            print(f"\n📏 距离误差统计:")
            print(f"  平均距离误差: {self.results['mean_distance']:.2f} 像素")
            print(f"  中位数距离误差: {self.results['median_distance']:.2f} 像素")
            print(f"  标准差: {self.results['std_distance']:.2f} 像素")
            print(f"  最小误差: {self.results['min_distance']:.2f} 像素")
            print(f"  最大误差: {self.results['max_distance']:.2f} 像素")
            
            print(f"\n🎯 平均绝对误差(MAE): {self.results['mae']:.2f} 像素")
            
            print(f"\n✅ 不同阈值下的准确率:")
            for key, value in self.results.items():
                if key.startswith('accuracy@'):
                    threshold = key.split('@')[1]
                    print(f"  {threshold}: {value:.2%}")
        
        if 'mean_normalized_error' in self.results:
            print(f"\n📐 归一化误差 (0~1, 越小越好):")
            print(f"  平均归一化误差: {self.results['mean_normalized_error']:.4f}")
            print(f"  中位数归一化误差: {self.results['median_normalized_error']:.4f}")
        
        if 'relative_point_distance_error' in self.results:
            print(f"\n🔗 相对点间距离误差:")
            print(f"  平均相对点间距离误差: {self.results['relative_point_distance_error']:.2f} 像素")
    
    def calculate_pdr_curve(self, distances: List[float], max_threshold: float = 50, num_points: int = 100):
        """
        计算点检测率曲线（PDR Curve）
        Args:
            distances: 所有点的距离误差列表
            max_threshold: 最大阈值
            num_points: 曲线采样点数
        Returns:
            thresholds: 阈值列表
            pdrs: 各阈值下的准确率
        """
        thresholds = np.linspace(0, max_threshold, num_points)
        pdrs = [self.calculate_accuracy_at_threshold(distances, t) for t in thresholds]
        return thresholds, pdrs
    
    def plot_pdr_curve(self, save_path: str = "pdr_curve.png"):
        """绘制点检测率曲线"""
        distances = self.calculate_distances()
        thresholds, pdrs = self.calculate_pdr_curve(distances)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, pdrs, 'b-', linewidth=2, label='PDR Curve')
        plt.xlabel('Distance Threshold (pixels)')
        plt.ylabel('Point Detection Rate')
        plt.title('Point Detection Rate vs Distance Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, max(thresholds))
        plt.ylim(0, 1)
        
        # 添加关键点标注
        key_thresholds = [5, 10, 15, 20]
        for threshold in key_thresholds:
            if threshold <= max(thresholds):
                idx = np.argmin(np.abs(np.array(thresholds) - threshold))
                plt.plot(threshold, pdrs[idx], 'ro', markersize=6)
                plt.annotate(f'{pdrs[idx]:.2%}', 
                           (threshold, pdrs[idx]), 
                           xytext=(5, 5), 
                           textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"PDR曲线已保存到: {save_path}")
    
    def save_results(self, output_file: str = "accuracy_results.json"):
        """保存结果到JSON文件"""
        if not self.results:
            print("没有结果可保存，请先运行calculate_all_metrics()")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_file}")

def demo_usage():
    """演示用法"""
    print("点坐标准确率计算器演示")
    print("="*50)
    
    # 示例：创建一些虚拟的预测数据用于演示
    gt_file = "test.json"
    pred_file = "test.json"
    
    # 如果预测文件不存在，创建一个示例文件
    if not os.path.exists(pred_file):
        print(f"创建示例预测文件: {pred_file}")
        
        # 读取GT数据
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        # 创建带有随机噪声的预测数据
        pred_data = {}
        for diagram_id, points in gt_data.items():
            pred_data[diagram_id] = {}
            for point_name, coord in points.items():
                # 添加随机噪声 (-10到10像素)
                noise_x = np.random.uniform(-10, 10)
                noise_y = np.random.uniform(-10, 10)
                pred_data[diagram_id][point_name] = [
                    coord[0] + noise_x,
                    coord[1] + noise_y
                ]
        
        with open(pred_file, 'w', encoding='utf-8') as f:
            json.dump(pred_data, f, indent=2, ensure_ascii=False)
    
    # 创建计算器实例
    calculator = PointAccuracyCalculator(
        gt_file=gt_file,
        pred_file=pred_file,
        image_size=(800, 600)  # 假设图像尺寸
    )
    
    # 计算所有指标
    results = calculator.calculate_all_metrics(thresholds=[5, 10, 15, 20, 25])
    
    # 打印结果
    calculator.print_results()
    
    # 绘制PDR曲线
    calculator.plot_pdr_curve()
    
    # 保存结果
    calculator.save_results()

if __name__ == "__main__":
    demo_usage()
