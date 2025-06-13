# recognize.py
import cv2
import torch
import numpy as np
from torchvision import transforms
from models import EnhancedCNN

class DigitRecognizer:
    def __init__(self, model_path, device="cuda", debug=False):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = EnhancedCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.debug = debug

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
        for i, (digit_img, box, angle) in enumerate(digits_info):
            # 使用智能旋转校正方法
            best_angle, best_conf, best_pred = self._correct_and_recognize(digit_img, angle)
            
            if best_conf >= 0.4:  # 只保留置信度足够的数字
                results.append((best_pred, box, best_conf))
        
        # 3. 后处理和结果返回
        img = 255 - img  # 反色操作（白底黑字 -> 黑底白字）
        return self._sort_results(results), img

    def _correct_and_recognize(self, digit_img, rect_angle):
        """根据旋转矩形的角度智能校正并识别"""
        # 确定初始旋转角度
        initial_rotation = 0
        
        # 分析旋转矩形角度（在[-90, 0)范围内）
        if rect_angle < -75:  # 接近水平方向，需要大幅校正
            initial_rotation = -rect_angle - 90
        elif rect_angle < -15:  # 明显倾斜
            initial_rotation = -rect_angle
        # 轻微倾斜（-15°到0°）不校正
        
        # 尝试初始校正
        rotated_img = self._rotate_image(digit_img, initial_rotation)
        best_angle, best_conf, best_pred = self._find_best_rotation(rotated_img, initial_rotation)
        
        # 如果置信度不高，尝试额外的180°旋转
        if best_conf < 0.7:
            rotated_180 = self._rotate_image(rotated_img, 180)
            _, conf_180, pred_180 = self._find_best_rotation(rotated_180, initial_rotation + 180)
            
            # 如果180°旋转后置信度更高，则采用
            if conf_180 > best_conf:
                best_angle = initial_rotation + 180
                best_conf = conf_180
                best_pred = pred_180
            # 否则保持原结果
        
        return best_angle, best_conf, best_pred

    # 保留了对旋转矩阵的提取
    def _find_best_rotation(self, digit_img, base_angle=0):
        """在有限角度范围内寻找最佳旋转"""
        best_angle = 0
        best_conf = -1
        best_pred = -1
        
        # 只需要尝试0°和可能的±5°微调
        angles = [0]
        
        # 如果需要微调，添加±5°选项
        if abs(base_angle) > 2:
            angles.extend([5, -5])
        
        for angle in angles:
            # 应用微调旋转
            rotated = self._rotate_image(digit_img, angle) if angle != 0 else digit_img
            standardized = self._enhanced_standardization(rotated)
            
            if standardized is None or standardized.size == 0:
                continue
                
            fill_ratio = np.count_nonzero(standardized) / standardized.size
            if fill_ratio < 0.03 or fill_ratio > 0.95:
                continue
                
            tensor = self.transform(standardized).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                
                # ========== 新增：对 '8' 和 '0' 等特殊数字的特殊处理 ==========
                # 获取数字的原始置信度
                conf_0 = probs[0, 0].item()
                conf_8 = probs[0, 8].item()
                conf_3 = probs[0, 3].item()
                conf_9 = probs[0, 9].item()
                
                # 特殊处理
                if conf_3 > 0.5 and conf_3 > conf_9 and conf_9 > 0.1:
                    probs[0, 9] = conf_3 + 0.5
                elif conf_0 > 0.5 and conf_0 > conf_8 and conf_0 < 0.95:
                    probs[0, 8] = conf_8 + 1
                    # 重新归一化概率分布
                    probs = probs / probs.sum(dim=1, keepdim=True)
                    # ========== 结束新增 ==========
                
                conf, pred = torch.max(probs, dim=1)
                confidence = conf.item()
                
                if confidence > best_conf:
                    best_angle = angle
                    best_conf = confidence
                    best_pred = pred.item()
        
        # 返回绝对角度（基础角度+微调角度）和结果
        return base_angle + best_angle, best_conf, best_pred

    def _rotate_image(self, img, angle):
        """旋转图像，保持原始大小"""
        if angle == 0:
            return img.copy()
            
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    def _improved_digit_detection(self, processed_img, original_img):
        """改进的数字检测方法（使用旋转矩形）"""
        contours, _ = cv2.findContours(processed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidate_rects = []
        img_area = processed_img.shape[0] * processed_img.shape[1]
        
        # 增加最小面积约束 (绝对像素值)
        MIN_AREA_PIXELS = 80  # x 像素以下的区域直接忽略
        MIN_DIMENSION = 5     # 最小宽度/高度要求
        SAFETY_MARGIN = 0     # 安全边界阈值（像素）
        
        for cnt in contours:
            # 使用旋转矩形代替水平矩形
            rect = cv2.minAreaRect(cnt)
            center, (width, height), angle = rect
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # 计算旋转矩形的面积
            rect_area = width * height
            if rect_area == 0:  # 防止除零错误
                continue
                
            contour_area = cv2.contourArea(cnt)
            density = contour_area / rect_area
            
            # 动态过滤条件 - 添加绝对面积和尺寸约束
            min_area = max(MIN_AREA_PIXELS, img_area * 0.0002)  # 结合相对和绝对阈值
            max_area = img_area * 0.3
            
            # 检查尺寸是否过小
            min_dimension = min(width, height)
            if min_dimension < MIN_DIMENSION:
                continue
                
            # 应用所有过滤条件
            if (min_area < contour_area < max_area and 
                0.02 < min(width, height)/max(width, height) < 40.0 and 
                0.05 < density < 0.95):
                candidate_rects.append((rect, box))
        
        # 重叠矩形过滤
        filtered_rects = []
        for i, (rect_i, box_i) in enumerate(candidate_rects):
            is_inside = False
            
            for j, (rect_j, box_j) in enumerate(candidate_rects):
                if i == j: continue
                
                # 检查旋转矩形是否完全包含在另一个旋转矩形内
                inside_points = 0
                # 修正点格式问题：确保点是整数元组
                for point in box_i:
                    pt = tuple(map(int, point))  # 转换为整数元组
                    if cv2.pointPolygonTest(box_j, pt, False) >= 0:
                        inside_points += 1
                    
                if inside_points == 4:  # 所有点都在另一个矩形内
                    is_inside = True
                    break
                    
            if not is_inside:
                filtered_rects.append((rect_i, box_i))
        
        # ROI提取（使用旋转矩形的最小外接矩形）
        digits = []
        img_h, img_w = original_img.shape[:2]  # 获取图像尺寸
        
        for (rect, box) in filtered_rects:
            # 获取旋转矩形的参数
            center, (width, height), angle = rect
            
            # 确保宽度和高度有效
            if width == 0 or height == 0:
                continue
                
            # 获取最小外接矩形
            x, y, w, h = cv2.boundingRect(box)
            
            # 检查是否超出边界 - 新增安全边界处理
            out_of_bounds = False
            if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
                out_of_bounds = True
                
            # 应用安全边界：如果超出边界，则向内收缩SAFETY_MARGIN像素
            if out_of_bounds:
                # 计算向内收缩的矩形
                safe_x = max(0, x) + SAFETY_MARGIN
                safe_y = max(0, y) + SAFETY_MARGIN
                safe_w = max(0, w - 2 * SAFETY_MARGIN)
                safe_h = max(0, h - 2 * SAFETY_MARGIN)
                
                # 确保收缩后的矩形有效
                if safe_w > MIN_DIMENSION and safe_h > MIN_DIMENSION:
                    # 使用收缩后的矩形
                    x, y, w, h = safe_x, safe_y, safe_w, safe_h
            
            # 裁剪数字区域（确保在图像范围内）
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)
            
            # 检查有效区域
            if x2 > x1 and y2 > y1 and (x2 - x1) > MIN_DIMENSION and (y2 - y1) > MIN_DIMENSION:
                digit_img = original_img[y1:y2, x1:x2]
                
                if digit_img.size > 0:
                    # 返回数字图像、边界框和旋转角度
                    digits.append((digit_img, box, angle))
        
        return digits

    def _enhanced_standardization(self, digit_img):
        """增强的标准化方法（简化版）"""
        if len(digit_img.shape) == 3:
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        
        # 二值化处理
        _, binary = cv2.threshold(digit_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 确保数字图像不会被过度压缩
        max_dim = max(binary.shape[:2])
        if max_dim > 0:
            scale = 20 / max_dim  # 稍微缩小，保留边界
            scaled = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            scaled = binary.copy()
        
        # 创建28x28画布，将数字置于中心
        canvas = np.zeros((28, 28), dtype=np.uint8)
        y_offset = (28 - scaled.shape[0]) // 2
        x_offset = (28 - scaled.shape[1]) // 2
        if y_offset >= 0 and x_offset >= 0:
            canvas[y_offset:y_offset+scaled.shape[0], 
                  x_offset:x_offset+scaled.shape[1]] = scaled
        else:
            # 如果数字太大，直接缩放
            canvas = cv2.resize(scaled, (28, 28))
        
        return canvas

    def _sort_results(self, results):
        """改进的结果排序（基于旋转矩形中心点）"""
        if not results:
            return []
            
        # 计算每个旋转矩形的中心点
        results_with_center = []
        for (pred, box, conf) in results:
            center_x = np.mean(box[:, 0])
            center_y = np.mean(box[:, 1])
            results_with_center.append((pred, box, conf, center_x, center_y))
        
        # 按行分组排序
        rows = {}
        for item in results_with_center:
            _, _, _, _, y = item
            row_key = round(y / 20)  # 行高阈值设为20像素
            
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append(item)
        
        # 每行内按x坐标排序
        sorted_results = []
        for row in sorted(rows.keys()):
            sorted_in_row = sorted(rows[row], key=lambda x: x[3])  # 按中心点x坐标排序
            for (pred, box, conf, _, _) in sorted_in_row:
                sorted_results.append((pred, box, conf))
            
        return sorted_results