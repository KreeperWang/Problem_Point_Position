import cv2
import numpy as np
import json
import os
from collections import defaultdict
import easyocr
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import time
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

class ImprovedGeometryPointExtractor:
    def __init__(self, use_gpu=True):
        """
        改进的点提取器，专门针对几何图形优化
        """
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 初始化多个OCR引擎
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        
        # 参数设置（针对几何图形优化）
        self.min_text_confidence = 0.3  # 降低置信度阈值
        self.point_merge_distance = 50  # 增加合并距离
        self.valid_labels = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # 字母检测的特定参数
        self.letter_min_area = 20  # 更小的最小面积
        self.letter_max_area = 5000  # 更大的最大面积
        
    def extract_points(self, image_path):
        """
        主要的点提取方法
        """
        img = cv2.imread(image_path)
        if img is None:
            return {}
        
        # 使用多种策略提取点
        strategies = [
            #self._strategy_1_enhanced_ocr,
            self._strategy_2_template_matching,
            #self._strategy_3_contour_based,
            #self._strategy_4_connected_components,
            ##self._strategy_5_tesseract_ocr
        ]
        
        all_candidates = []
        
        for strategy in strategies:
            try:
                candidates = strategy(img)
                all_candidates.extend(candidates)
            except Exception as e:
                print(f"策略 {strategy.__name__} 出错: {str(e)}")
                continue
        
        # 合并所有候选点
        final_points = self._merge_all_candidates(all_candidates, img)
        
        return final_points
    
    def _strategy_1_enhanced_ocr(self, img):
        """
        策略1：增强的OCR方法
        """
        candidates = []
        
        # 多种预处理方式
        preprocessed_images = [
            img,  # 原图
            self._preprocess_for_text(img),  # 文本优化
            self._preprocess_high_contrast(img),  # 高对比度
            self._preprocess_remove_colors(img),  # 去除彩色
        ]
        
        for processed_img in preprocessed_images:
            # 使用不同的参数进行OCR
            for width_ths in [0.5, 0.7, 0.9]:
                for height_ths in [0.5, 0.7, 0.9]:
                    try:
                        results = self.reader.readtext(
                            processed_img, 
                            width_ths=width_ths, 
                            height_ths=height_ths,
                            text_threshold=0.3,
                            low_text=0.3,
                            link_threshold=0.3
                        )
                        
                        for (bbox, text, confidence) in results:
                            text = text.strip().upper()
                            
                            # 放宽匹配条件
                            if len(text) == 1 and text in self.valid_labels:
                                bbox = np.array(bbox)
                                center_x = np.mean(bbox[:, 0])
                                center_y = np.mean(bbox[:, 1])
                                
                                candidates.append({
                                    'label': text,
                                    'x': float(center_x),
                                    'y': float(center_y),
                                    'confidence': confidence,
                                    'method': 'ocr_enhanced'
                                })
                    except:
                        continue
        
        return candidates
    
    def _strategy_2_template_matching(self, img):
        """
        策略2：模板匹配方法
        """
        candidates = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 为每个字母创建多个尺度的模板
        scales = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
        
        for letter in self.valid_labels:
            for scale in scales:
                # 创建字母模板
                template = self._create_letter_template(letter, scale)
                
                # 模板匹配
                try:
                    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                    threshold = 0.6  # 降低阈值
                    locations = np.where(result >= threshold)
                    
                    for pt in zip(*locations[::-1]):
                        center_x = pt[0] + template.shape[1] // 2
                        center_y = pt[1] + template.shape[0] // 2
                        
                        candidates.append({
                            'label': letter,
                            'x': float(center_x),
                            'y': float(center_y),
                            'confidence': result[pt[1], pt[0]],
                            'method': 'template_matching'
                        })
                except:
                    continue
        
        return candidates
    
    def _strategy_3_contour_based(self, img):
        """
        策略3：改进的轮廓检测方法
        """
        candidates = []
        
        # 多种二值化方法
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        binary_images = [
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2),
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)
        ]
        
        for binary in binary_images:
            # 使用不同的形态学操作
            kernels = [
                cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            ]
            
            for kernel in kernels:
                processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
                
                contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if self.letter_min_area < area < self.letter_max_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        
                        # 字母的宽高比通常接近1
                        if 0.3 < aspect_ratio < 3:
                            roi = img[y:y+h, x:x+w]
                            
                            # 对ROI进行OCR
                            try:
                                roi_results = self.reader.readtext(roi, width_ths=0.5, height_ths=0.5)
                                
                                for (_, text, confidence) in roi_results:
                                    text = text.strip().upper()
                                    if len(text) == 1 and text in self.valid_labels:
                                        candidates.append({
                                            'label': text,
                                            'x': float(x + w/2),
                                            'y': float(y + h/2),
                                            'confidence': confidence * 0.8,
                                            'method': 'contour'
                                        })
                                        break
                            except:
                                continue
        
        return candidates
    
    def _strategy_4_connected_components(self, img):
        """
        策略4：连通组件分析
        """
        candidates = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用连通组件分析
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if self.letter_min_area < area < self.letter_max_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                aspect_ratio = w / h if h > 0 else 0
                
                if 0.3 < aspect_ratio < 3:
                    roi = img[y:y+h, x:x+w]
                    
                    try:
                        roi_results = self.reader.readtext(roi, width_ths=0.5, height_ths=0.5)
                        
                        for (_, text, confidence) in roi_results:
                            text = text.strip().upper()
                            if len(text) == 1 and text in self.valid_labels:
                                candidates.append({
                                    'label': text,
                                    'x': float(centroids[i][0]),
                                    'y': float(centroids[i][1]),
                                    'confidence': confidence * 0.7,
                                    'method': 'connected_components'
                                })
                                break
                    except:
                        continue
        
        return candidates
    
    def _strategy_5_tesseract_ocr(self, img):
        """
        策略5：使用Tesseract OCR作为补充
        """
        candidates = []
        
        try:
            # 转换为PIL图像
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # 增强图像
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(2.0)
            
            # 使用Tesseract进行OCR
            custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            data = pytesseract.image_to_data(pil_img, config=custom_config, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip().upper()
                conf = int(data['conf'][i])
                
                if len(text) == 1 and text in self.valid_labels and conf > 30:
                    x = data['left'][i] + data['width'][i] // 2
                    y = data['top'][i] + data['height'][i] // 2
                    
                    candidates.append({
                        'label': text,
                        'x': float(x),
                        'y': float(y),
                        'confidence': conf / 100.0,
                        'method': 'tesseract'
                    })
        except:
            pass
        
        return candidates
    
    def _preprocess_for_text(self, img):
        """
        专门针对文本识别的预处理
        """
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 去噪
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # 锐化
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_high_contrast(self, img):
        """
        高对比度预处理
        """
        # 转换为LAB颜色空间
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 对L通道进行CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # 合并通道
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _preprocess_remove_colors(self, img):
        """
        去除彩色，保留黑色文字
        """
        # 转换为HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 创建黑色掩码
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # 创建白色背景
        result = np.ones_like(img) * 255
        result[mask > 0] = img[mask > 0]
        
        return result
    
    def _create_letter_template(self, letter, scale=1.0):
        """
        创建字母模板用于模板匹配
        """
        # 基础大小
        base_size = int(30 * scale)
        
        # 创建空白图像
        template = np.ones((base_size, base_size), dtype=np.uint8) * 255
        
        # 添加文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8 * scale
        thickness = max(1, int(2 * scale))
        
        # 获取文字大小
        (text_width, text_height), _ = cv2.getTextSize(letter, font, font_scale, thickness)
        
        # 计算文字位置（居中）
        x = (base_size - text_width) // 2
        y = (base_size + text_height) // 2
        
        # 绘制文字
        cv2.putText(template, letter, (x, y), font, font_scale, 0, thickness)
        
        return template
    
    def _merge_all_candidates(self, candidates, img):
        """
        合并所有候选点，使用更智能的策略
        """
        if not candidates:
            return {}
        
        # 按标签分组
        label_groups = defaultdict(list)
        for cand in candidates:
            label_groups[cand['label']].append(cand)
        
        final_points = {}
        h, w = img.shape[:2]
        
        for label, group in label_groups.items():
            if len(group) == 1:
                point = group[0]
                final_points[label] = [point['x'], point['y']]
            else:
                # 使用DBSCAN聚类
                positions = np.array([[p['x'], p['y']] for p in group])
                
                clustering = DBSCAN(eps=self.point_merge_distance, min_samples=1).fit(positions)
                
                # 找到最大的聚类
                unique_labels, counts = np.unique(clustering.labels_, return_counts=True)
                
                if len(unique_labels) > 0:
                    # 选择点数最多的聚类
                    best_cluster = unique_labels[np.argmax(counts)]
                    cluster_points = [group[i] for i in range(len(group)) if clustering.labels_[i] == best_cluster]
                    
                    # 在聚类中选择置信度最高的点或计算加权平均
                    if len(cluster_points) == 1:
                        best_point = cluster_points[0]
                        final_points[label] = [best_point['x'], best_point['y']]
                    else:
                        # 加权平均
                        total_conf = sum(p['confidence'] for p in cluster_points)
                        if total_conf > 0:
                            avg_x = sum(p['x'] * p['confidence'] for p in cluster_points) / total_conf
                            avg_y = sum(p['y'] * p['confidence'] for p in cluster_points) / total_conf
                        else:
                            avg_x = np.mean([p['x'] for p in cluster_points])
                            avg_y = np.mean([p['y'] for p in cluster_points])
                        
                        # 边界检查
                        avg_x = max(0, min(avg_x, w-1))
                        avg_y = max(0, min(avg_y, h-1))
                        
                        final_points[label] = [float(avg_x), float(avg_y)]
        
        return final_points

def process_dataset_improved(image_dir="Images", output_file="predictions34.json"):
    """
    使用改进的方法处理数据集
    """
    extractor = ImprovedGeometryPointExtractor(use_gpu=True)
    results = {}
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    image_files.sort(key=lambda x: int(x.split('.')[0]))
    
    print(f"找到 {len(image_files)} 张图像")
    
    start_time = time.time()
    
    # 处理每张图像
    for image_file in tqdm(image_files, desc="处理图像"):
        image_path = os.path.join(image_dir, image_file)
        image_id = image_file.split('.')[0]
        
        try:
            points = extractor.extract_points(image_path)
            results[image_id] = points
            
            # 输出当前图像的识别结果
            if points:
                print(f"\n图像 {image_id}: 识别到 {len(points)} 个点: {list(points.keys())}")
            else:
                print(f"\n图像 {image_id}: 未识别到任何点")
                
        except Exception as e:
            print(f"\n处理图像 {image_id} 时出错: {str(e)}")
            results[image_id] = {}
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    elapsed_time = time.time() - start_time
    print(f"\n处理完成！总耗时: {elapsed_time:.2f} 秒")
    print(f"结果已保存到 {output_file}")
    
    return results

if __name__ == "__main__":
    # 请确保安装了 pytesseract
    # pip install pytesseract
    # 并且安装了 Tesseract-OCR: https://github.com/tesseract-ocr/tesseract
    
    results = process_dataset_improved(
        image_dir="Images_part", 
        output_file="predictions_improved.json"
    )
    
    # 评估结果
    if os.path.exists("test.json"):
        from cal_acc import PointAccuracyCalculator
        calculator = PointAccuracyCalculator(
            gt_file="test.json",
            pred_file="predictions_improved.json"
        )
        calculator.calculate_all_metrics()
        calculator.print_results()