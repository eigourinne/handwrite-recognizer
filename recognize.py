# recognize.py
import cv2
import torch
import numpy as np
from torchvision import transforms
from models import EnhancedCNN

# 识别器
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
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"图像读取失败: {image_path}")

        processed = self._advanced_preprocess(img)
        digits_info = self._improved_digit_detection(processed, img)
        
        results = []
        for i, (digit_img, rect) in enumerate(digits_info):
            standardized = self._enhanced_standardization(digit_img)
            
            # 计算数字区域的填充率（过滤过于稀疏的检测）
            fill_ratio = np.count_nonzero(standardized) / standardized.size
            if fill_ratio < 0.03:  # 忽略填充率过低的区域
                continue
                
            tensor = self.transform(standardized).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)
                
                # 对长数字'1'的特殊处理
                if pred.item() == 1:
                    # 计算长宽比
                    h, w = digit_img.shape
                    aspect_ratio = max(h, w) / min(h, w) if min(h, w) > 0 else 1
                    
                    # 如果是特别长的'1'，提高置信度阈值
                    min_conf = 0.7 if aspect_ratio > 5 else 0.5
                else:
                    # 动态置信度阈值（对小数字更宽松）
                    min_conf = 0.6 if standardized.sum() > 800 else 0.4
                    
                if conf.item() > min_conf:
                    results.append((pred.item(), rect, conf.item()))
        
        # 添加后处理：修正可能的识别错误
        results = self._postprocess_results(results)
        return self._sort_results(results), img
    
    def _postprocess_results(self, results):
        """后处理：修正可能的识别错误，特别是长数字'1'"""
        if not results:
            return results
            
        corrected_results = []
        prev_pred = None
        
        for i, (pred, rect, conf) in enumerate(results):
            # 检查是否是特别长的数字
            x, y, w, h = rect
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
            
            # 如果是特别长的垂直形状，且识别为其他数字，很可能应该是'1'
            if aspect_ratio > 4 and h > w and pred != 1:
                # 检查周围数字
                if prev_pred == 1 or (i < len(results)-1 and results[i+1][0] == 1):
                    corrected_results.append((1, rect, min(1.0, conf + 0.2)))  # 提高置信度
                    continue
            
            # 如果是特别长的水平形状，且识别为其他数字，很可能应该是'-'或'1'的误识别
            elif aspect_ratio > 4 and w > h and pred != 1:
                # 检查是否可能是'-'，但这里我们只处理数字
                # 保持原预测
                pass
                
            corrected_results.append((pred, rect, conf))
            prev_pred = pred
            
        return corrected_results

    def _advanced_preprocess(self, img):
        """自适应的通用预处理"""
        # 1. 基于图像特性的动态参数计算
        img_std = np.std(img)
        clip_limit = max(1.0, min(3.0, img_std/25))  # 动态CLAHE参数
        block_size = max(11, min(31, int(img.shape[1]/20)*2+1))  # 动态块大小
        
        # 2. 自适应对比度增强
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        
        # 3. 动态阈值处理
        binary = cv2.adaptiveThreshold(enhanced, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block_size, 3)
        
        # 4. 基于图像噪声水平的形态学操作
        noise_ratio = np.count_nonzero(binary) / binary.size
        kernel_size = 1 if noise_ratio < 0.05 else 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 5. 增加闭操作连接不连续笔画
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    
    def _improved_digit_detection(self, processed_img, original_img):
        """改进的数字检测：支持倾斜数字和不连续数字"""
        # 1. 使用更鲁棒的轮廓分析方法
        contours, _ = cv2.findContours(processed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        candidate_rects = []  # 存储所有候选矩形
        img_area = processed_img.shape[0] * processed_img.shape[1]
        
        for cnt in contours:
            # 使用最小外接矩形处理倾斜数字
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # 计算矩形参数
            x, y, w, h = cv2.boundingRect(cnt)
            contour_area = cv2.contourArea(cnt)
            
            # 动态面积阈值
            min_area = max(10, img_area * 0.0002)
            max_area = img_area * 0.2
            
            # 基于密度和形状的动态判断
            density = contour_area / (w * h)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
            
            if (min_area < contour_area < max_area and 
                0.05 < aspect_ratio < 10 and 
                0.15 < density < 0.95):
                candidate_rects.append((box, (x, y, w, h)))
        
        # 过滤完全包含在其他矩形内部的矩形
        filtered_rects = []
        for i, (box_i, hor_rect_i) in enumerate(candidate_rects):
            x_i, y_i, w_i, h_i = hor_rect_i
            is_inside = False
            
            for j, (box_j, hor_rect_j) in enumerate(candidate_rects):
                if i == j:
                    continue
                    
                x_j, y_j, w_j, h_j = hor_rect_j
                # 检查 rect_i 是否完全在 rect_j 内部
                if (x_j <= x_i and 
                    y_j <= y_i and 
                    (x_j + w_j) >= (x_i + w_i) and 
                    (y_j + h_j) >= (y_i + h_i)):
                    is_inside = True
                    break
                    
            if not is_inside:
                filtered_rects.append((box_i, hor_rect_i))
        
        # 提取过滤后的ROI区域并进行几何校正
        digits = []
        for (rotated_box, hor_rect) in filtered_rects:
            # 提取旋转矩形区域
            x, y, w, h = hor_rect
            
            # 几何校正：透视变换
            corrected = self._perspective_transform(original_img, rotated_box)
            if corrected is None:
                # 如果校正失败，使用水平矩形
                pad = int(max(w, h) * 0.15)
                roi = original_img[
                    max(0, y-pad):min(original_img.shape[0], y+h+pad),
                    max(0, x-pad):min(original_img.shape[1], x+w+pad)
                ]
                digits.append((roi, (x, y, w, h)))
            else:
                # 使用校正后的图像
                digits.append((corrected, (x, y, w, h)))
        
        return digits

    def _perspective_transform(self, img, box):
        """几何校正：透视变换处理倾斜数字"""
        try:
            # 将点排序为 [左上, 右上, 右下, 左下] 顺序
            rect = self._order_points(box)
            (tl, tr, br, bl) = rect
            
            # 计算新图像的宽度（取两对边的最大距离）
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            # 计算新图像的高度（取两对边的最大距离）
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            # 构造目标点坐标
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            
            # 计算透视变换矩阵并应用
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            return warped
        except:
            return None

    def _order_points(self, pts):
        """将点排序为 [左上, 右上, 右下, 左下] 顺序"""
        # 按x坐标排序
        xSorted = pts[np.argsort(pts[:, 0]), :]
        
        # 获取左侧的两个点和右侧的两个点
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        
        # 按y坐标排序
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        
        # 计算右侧点的重心
        D = np.linalg.norm(rightMost - np.mean(rightMost, axis=0), axis=1)
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        
        return np.array([tl, tr, br, bl], dtype="float32")

    def _enhanced_standardization(self, digit_img):
        """改进的标准化：处理不连续数字"""
        if len(digit_img.shape) == 3:
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        
        # 1. 使用形态学操作连接不连续笔画
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        closed = cv2.morphologyEx(digit_img, cv2.MORPH_CLOSE, kernel)
        
        # 2. 自适应二值化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        norm = clahe.apply(closed)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # 3. 调整大小并居中
        return self._resize_and_center(binary)

    def _sort_results(self, results):
        """改进的结果排序逻辑"""
        if not results:
            return []
            
        # 先按y坐标分组（行）
        rows = {}
        for item in results:
            _, (x, y, w, h), _ = item
            row_key = round(y / (h * 1.2))  # 动态行高分组
            
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