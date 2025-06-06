# recognize.py
import cv2
import torch
import numpy as np
from torchvision import transforms
from models import EnhancedCNN

class DigitRecognizer:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = EnhancedCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def find_and_recognize(self, image_path):
        """主识别流程"""
        # 1. 图像读取和预处理
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"图像读取失败: {image_path}")

        processed = self._advanced_preprocess(img)
        digits_info = self._improved_digit_detection(processed, img)
        
        # 2. 数字识别
        results = []
        for i, (digit_img, rect) in enumerate(digits_info):
            standardized = self._enhanced_standardization(digit_img)
            
            # 有效性检查
            fill_ratio = np.count_nonzero(standardized) / standardized.size
            if fill_ratio < 0.03:  # 忽略填充率过低的区域
                continue
                
            # 动态置信度阈值
            tensor = self.transform(standardized).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)
                
                min_conf = 0.6 if standardized.sum() > 800 else 0.4
                if conf.item() > min_conf:
                    results.append((pred.item(), rect, conf.item()))
        
        # 3. 后处理和结果返回
        return self._sort_results(results), img

    def _advanced_preprocess(self, img):
        """增强的预处理流程"""
        # 动态参数计算
        img_std = np.std(img)
        clip_limit = max(1.0, min(3.0, img_std/25))
        block_size = max(11, min(31, int(img.shape[1]/20)*2+1))
        
        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        
        # 动态阈值处理
        binary = cv2.adaptiveThreshold(enhanced, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block_size, 3)
        
        # 噪声感知的形态学操作
        noise_ratio = np.count_nonzero(binary) / binary.size
        kernel_size = 1 if noise_ratio < 0.05 else 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    def _improved_digit_detection(self, processed_img, original_img):
        """改进的数字检测方法"""
        contours, _ = cv2.findContours(processed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidate_rects = []
        img_area = processed_img.shape[0] * processed_img.shape[1]
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            contour_area = cv2.contourArea(cnt)
            density = contour_area / (w * h)
            
            # 动态过滤条件
            min_area = max(10, img_area * 0.0002)
            max_area = img_area * 0.2
            if (min_area < contour_area < max_area and 
                0.05 < w/h < 10 and 
                0.15 < density < 0.95):
                candidate_rects.append((x, y, w, h))
        
        # 重叠矩形过滤
        filtered_rects = []
        for i, rect_i in enumerate(candidate_rects):
            x_i, y_i, w_i, h_i = rect_i
            is_inside = False
            
            for j, rect_j in enumerate(candidate_rects):
                if i == j: continue
                x_j, y_j, w_j, h_j = rect_j
                if (x_j <= x_i and y_j <= y_i and 
                    (x_j + w_j) >= (x_i + w_i) and 
                    (y_j + h_j) >= (y_i + h_i)):
                    is_inside = True
                    break
                    
            if not is_inside:
                filtered_rects.append(rect_i)
        
        # ROI提取
        digits = []
        for (x, y, w, h) in filtered_rects:
            pad = int(max(w, h) * 0.15)
            roi = original_img[
                max(0, y-pad):min(original_img.shape[0], y+h+pad),
                max(0, x-pad):min(original_img.shape[1], x+w+pad)
            ]
            digits.append((roi, (x, y, w, h)))
        
        return digits

    def _enhanced_standardization(self, digit_img):
        """增强的标准化方法"""
        if len(digit_img.shape) == 3:
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        
        # 多尺度处理
        scales = [0.8, 1.0, 1.2]
        processed = []
        
        for scale in scales:
            h, w = digit_img.shape
            scaled = cv2.resize(digit_img, (int(w*scale), int(h*scale)))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            norm = clahe.apply(scaled)
            _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            processed.append(binary)
        
        # 多尺度融合
        canvas = np.zeros((28,28), dtype=np.uint8)
        for p in processed:
            resized = cv2.resize(p, (28,28))
            canvas = cv2.bitwise_or(canvas, resized)
        
        return canvas

    def _sort_results(self, results):
        """改进的结果排序"""
        if not results:
            return []
            
        # 按行分组排序
        rows = {}
        for item in results:
            _, (x, y, w, h), _ = item
            row_key = round(y / (h * 1.2))
            
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append(item)
        
        # 每行内按x坐标排序
        sorted_results = []
        for row in sorted(rows.keys()):
            sorted_results.extend(sorted(rows[row], key=lambda x: x[1][0]))
            
        return sorted_results

    def _resize_and_center(self, img):
        """辅助缩放居中函数"""
        target_size = 20
        h, w = img.shape
        scale = min(target_size / w, target_size / h)
        resized = cv2.resize(img, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_AREA)

        canvas = np.zeros((28,28), dtype=np.uint8)
        dx = (28 - resized.shape[1]) // 2
        dy = (28 - resized.shape[0]) // 2
        canvas[dy:dy+resized.shape[0], dx:dx+resized.shape[1]] = resized
        return canvas