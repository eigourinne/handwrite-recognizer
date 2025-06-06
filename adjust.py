# adjust.py
import cv2
import numpy as np
import os

# 几何矫正
class GeometricAdjuster:
    def __init__(self):
        self.points = []  # 存储选择的四个点
        self.original_img = None
        self.window_name = "Geometric Correction"
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，用于选择校正点"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                cv2.circle(self.original_img, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(self.window_name, self.original_img)
                
                # 显示已选择的点数量
                if len(self.points) > 0:
                    cv2.putText(self.original_img, f"Selected {len(self.points)}/4 points", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow(self.window_name, self.original_img)
    
    def _order_points(self, pts):
        """将点排序为: 左上, 右上, 右下, 左下"""
        rect = np.zeros((4, 2), dtype="float32")
        
        # 左上点将具有最小的x+y和，右下点将具有最大的x+y和
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        
        # 右上点将具有最小的x-y差，左下点将具有最大的x-y差
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下
        
        return rect
    
    def _four_point_transform(self, image, pts):
        """执行四点透视变换"""
        # 获取有序点并计算目标矩形尺寸
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        
        # 计算新图像的宽度（取最大宽度）
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # 计算新图像的高度（取最大高度）
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # 定义目标点
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # 计算透视变换矩阵并应用
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    def adjust_image(self, image_path, output_path=None):
        """
        对图像进行几何校正
        :param image_path: 输入图像路径
        :param output_path: 输出图像路径，如果为None则不保存
        :return: 校正后的图像
        """
        # 读取图像
        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 创建窗口并设置鼠标回调
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # 显示原始图像和操作说明
        display_img = self.original_img.copy()
        cv2.putText(display_img, "Select 4 corners (clockwise or counter-clockwise)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_img, "Press 'r' to reset, 'c' to confirm, 'q' to quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(self.window_name, display_img)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # 重置选择
            if key == ord('r'):
                self.points = []
                display_img = self.original_img.copy()
                cv2.putText(display_img, "Select 4 corners (clockwise or counter-clockwise)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_img, "Press 'r' to reset, 'c' to confirm, 'q' to quit", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(self.window_name, display_img)
            
            # 确认选择并执行变换
            elif key == ord('c'):
                if len(self.points) == 4:
                    # 执行透视变换
                    corrected_img = self._four_point_transform(self.original_img, np.array(self.points, dtype="float32"))
                    
                    # 显示结果
                    cv2.imshow("Corrected Image", corrected_img)
                    cv2.waitKey(0)
                    
                    # 保存结果
                    if output_path is not None:
                        cv2.imwrite(output_path, corrected_img)
                        print(f"校正后的图像已保存至: {output_path}")
                    
                    cv2.destroyAllWindows()
                    return corrected_img
                else:
                    print("请选择4个点")
            
            # 退出
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None

def main():
    print("=== 几何校正工具 ===")
    print("操作说明:")
    print("1. 在图像上点击选择4个角点（顺时针或逆时针）")
    print("2. 按 'c' 确认并执行校正")
    print("3. 按 'r' 重置选择")
    print("4. 按 'q' 退出")
    
    adjuster = GeometricAdjuster()
    
    while True:
        input_path = input("\n输入图像路径(或q退出): ").strip()
        if input_path.lower() == 'q':
            break
        
        if not os.path.exists(input_path):
            print(f"文件不存在: {input_path}")
            continue
        
        # 设置输出路径
        dirname, filename = os.path.split(input_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(dirname, f"{name}_corrected{ext}")
        
        try:
            print("正在处理图像...")
            adjuster.adjust_image(input_path, output_path)
            print("处理完成!")
        except Exception as e:
            print(f"处理图像时出错: {e}")

if __name__ == '__main__':
    main()