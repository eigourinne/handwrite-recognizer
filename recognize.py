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
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"图像读取失败: {image_path}")

        # 自动判断是否需要矫正
        corrected = self._auto_correct_image(img)

        processed = self._advanced_preprocess(corrected)
        digits_info = self._improved_digit_detection(processed, corrected)

        results = []
        for i, (digit_img, rect) in enumerate(digits_info):
            standardized = self._enhanced_standardization(digit_img)
            fill_ratio = np.count_nonzero(standardized) / standardized.size
            if fill_ratio < 0.005:
                continue

            tensor = self.transform(standardized).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)

                min_conf = 0.6 if standardized.sum() > 800 else 0.4
                if conf.item() > min_conf:
                    results.append((pred.item(), rect, conf.item()))

        return self._sort_results(results), img

    def _auto_correct_image(self, img):
        """检测倾斜图像并进行整体几何矫正（保留正常图）"""
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is None:
            return img

        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.rad2deg(theta)
            angle = angle if angle <= 90 else 180 - angle
            angles.append(angle)

        avg_angle = np.mean(angles)
        if 80 <= avg_angle <= 100 or -10 <= avg_angle <= 10:
            return img  # 正常角度，跳过校正

        # 尝试执行透视矫正
        corrected = self._perspective_correction(img)
        return corrected if corrected is not None else img

    def _perspective_correction(self, img):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break
        else:
            return None

        pts = doc_cnt.reshape(4, 2)
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        if maxWidth < 10 or maxHeight < 10:
            return None

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        return warped

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _advanced_preprocess(self, img):
        img_std = np.std(img)
        clip_limit = max(1.0, min(3.0, img_std/25))
        block_size = max(11, min(31, int(img.shape[1]/20)*2+1))

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        enhanced = clahe.apply(img)

        binary = cv2.adaptiveThreshold(enhanced, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block_size, 3)

        noise_ratio = np.count_nonzero(binary) / binary.size
        kernel_size = 1 if noise_ratio < 0.05 else 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def _improved_digit_detection(self, processed_img, original_img):
        contours, _ = cv2.findContours(processed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        candidate_rects = []
        img_area = processed_img.shape[0] * processed_img.shape[1]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            contour_area = cv2.contourArea(cnt)

            min_area = max(10, img_area * 0.0002)
            max_area = img_area * 0.2

            density = contour_area / (w * h)
            if (min_area < contour_area < max_area and 
                0.05 < w/h < 10 and 
                0.15 < density < 0.95):
                candidate_rects.append((x, y, w, h))

        filtered_rects = []
        for i, rect_i in enumerate(candidate_rects):
            x_i, y_i, w_i, h_i = rect_i
            is_inside = False
            for j, rect_j in enumerate(candidate_rects):
                if i == j:
                    continue
                x_j, y_j, w_j, h_j = rect_j
                if (x_j <= x_i and 
                    y_j <= y_i and 
                    (x_j + w_j) >= (x_i + w_i) and 
                    (y_j + h_j) >= (y_i + h_i)):
                    is_inside = True
                    break
            if not is_inside:
                filtered_rects.append(rect_i)

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
        if len(digit_img.shape) == 3:
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

        scales = [0.8, 1.0, 1.2]
        processed = []

        for scale in scales:
            try:
                h, w = digit_img.shape
                if h < 2 or w < 2:
                    continue
                scaled = cv2.resize(digit_img, (int(w*scale), int(h*scale)))
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
                norm = clahe.apply(scaled)
                _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                processed.append(binary)
            except Exception as e:
                continue

        canvas = np.zeros((28,28), dtype=np.uint8)
        for p in processed:
            try:
                resized = cv2.resize(p, (28,28))
                canvas = cv2.bitwise_or(canvas, resized)
            except:
                continue

        return canvas

    def _sort_results(self, results):
        if not results:
            return []

        rows = {}
        for item in results:
            _, (x, y, w, h), _ = item
            row_key = round(y / (h * 1.2))
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append(item)

        sorted_results = []
        for row in sorted(rows.keys()):
            sorted_results.extend(sorted(rows[row], key=lambda x: x[1][0]))
        return sorted_results
