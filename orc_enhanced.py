import cv2
import numpy as np
import json
import os
from tqdm import tqdm

class GeometryPointExtractor:
    def __init__(self, test_json_path="test.json"):
        """
        初始化点提取器
        """
        # 加载test.json获取每张图片包含的点
        with open(test_json_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        # 参数设置
        self.template_scales = [0.6, 0.8, 1.0, 1.2, 1.4]  # 模板缩放比例
        self.search_radius = 50  # 搜索点的半径
        self.point_offset_range = 30  # 点相对于字母的最大偏移距离
        
    def extract_points(self, image_path, image_id):
        """
        提取指定图片中的点坐标
        """
        img = cv2.imread(image_path)
        if img is None:
            return {}
        
        # 获取该图片应该包含的点
        if image_id not in self.test_data:
            return {}
        
        expected_points = list(self.test_data[image_id].keys())
        
        # 对每个期望的点进行查找
        final_points = {}
        for letter in expected_points:
            # 先找到字母位置
            letter_pos = self._find_letter_position(img, letter)
            
            if letter_pos is not None:
                # 找到字母后，寻找附近的实际点位置
                point_pos = self._find_point_near_letter(img, letter_pos)
                if point_pos is not None:
                    final_points[letter] = [float(point_pos[0]), float(point_pos[1])]
        
        return final_points
    
    def _find_letter_position(self, img, letter):
        """
        使用模板匹配找到字母位置
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        best_match = None
        best_score = -1
        
        for scale in self.template_scales:
            # 创建字母模板
            template = self._create_letter_template(letter, scale)
            
            # 模板匹配
            try:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score and max_val > 0.5:  # 阈值
                    best_score = max_val
                    best_match = (
                        max_loc[0] + template.shape[1] // 2,
                        max_loc[1] + template.shape[0] // 2
                    )
            except:
                continue
        
        return best_match
    
    def _create_letter_template(self, letter, scale=1.0):
        """
        创建字母模板
        """
        base_size = int(40 * scale)
        template = np.ones((base_size, base_size), dtype=np.uint8) * 255
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9 * scale
        thickness = max(1, int(2 * scale))
        
        # 获取文字大小并居中
        (text_width, text_height), _ = cv2.getTextSize(letter, font, font_scale, thickness)
        x = (base_size - text_width) // 2
        y = (base_size + text_height) // 2
        
        cv2.putText(template, letter, (x, y), font, font_scale, 0, thickness)
        
        return template
    
    def _find_point_near_letter(self, img, letter_pos):
        """
        在字母附近寻找实际的点位置
        通过分析周围的黑色圆点来确定
        """
        x, y = letter_pos
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 定义搜索区域
        search_size = self.search_radius
        x_start = max(0, x - search_size)
        x_end = min(w, x + search_size)
        y_start = max(0, y - search_size)
        y_end = min(h, y + search_size)
        
        # 获取搜索区域
        search_region = gray[y_start:y_end, x_start:x_end]
        
        # 寻找黑色圆点（点通常是黑色的）
        # 使用阈值处理找到黑色区域
        _, binary = cv2.threshold(search_region, 50, 255, cv2.THRESH_BINARY_INV)
        
        # 使用霍夫圆检测找圆点
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
            # 选择最近的圆
            min_dist = float('inf')
            best_circle = None
            
            for circle in circles[0, :]:
                cx, cy = circle[0], circle[1]
                # 转换回原图坐标
                actual_x = x_start + cx
                actual_y = y_start + cy
                
                # 计算到字母中心的距离
                dist = np.sqrt((actual_x - x)**2 + (actual_y - y)**2)
                
                if dist < min_dist and dist < self.point_offset_range:
                    min_dist = dist
                    best_circle = (actual_x, actual_y)
            
            if best_circle:
                return best_circle
        
        # 如果霍夫圆检测失败，使用连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        min_dist = float('inf')
        best_point = None
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 点的面积应该在合理范围内
            if 10 < area < 200:
                cx = centroids[i][0] + x_start
                cy = centroids[i][1] + y_start
                
                dist = np.sqrt((cx - x)**2 + (cy - y)**2)
                
                if dist < min_dist and dist < self.point_offset_range:
                    min_dist = dist
                    best_point = (cx, cy)
        
        return best_point

def process_test_dataset(image_dir="Images", test_json="test.json", output_file="predictions.json"):
    """
    处理测试数据集
    """
    extractor = GeometryPointExtractor(test_json)
    results = {}
    
    # 获取test.json中的所有图像ID
    image_ids = list(extractor.test_data.keys())
    print(f"需要处理 {len(image_ids)} 张测试图像")
    
    # 处理每张图像
    for image_id in tqdm(image_ids, desc="处理图像"):
        image_path = os.path.join(image_dir, f"{image_id}.png")
        
        if not os.path.exists(image_path):
            print(f"警告: 图像 {image_path} 不存在")
            results[image_id] = {}
            continue
        
        try:
            points = extractor.extract_points(image_path, image_id)
            results[image_id] = points
            
            # 输出识别结果
            expected = set(extractor.test_data[image_id].keys())
            found = set(points.keys())
            missing = expected - found
            
            if missing:
                print(f"\n图像 {image_id}: 识别 {len(found)}/{len(expected)} 个点，缺失: {missing}")
            
        except Exception as e:
            print(f"\n处理图像 {image_id} 时出错: {str(e)}")
            results[image_id] = {}
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n处理完成！结果已保存到 {output_file}")
    
    return results

if __name__ == "__main__":
    # 运行主程序
    results = process_test_dataset(
        image_dir="Images",
        test_json="test.json",
        output_file="predictions.json"
    )
    
    # 如果有评估脚本，进行评估
    if os.path.exists("cal_acc.py"):
        from cal_acc import PointAccuracyCalculator
        calculator = PointAccuracyCalculator(
            gt_file="test.json",
            pred_file="predictions2.json"
        )
        calculator.calculate_all_metrics()
        calculator.print_results()