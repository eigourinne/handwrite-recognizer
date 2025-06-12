# divide.py
import cv2
import numpy as np
from tkinter import Tk, filedialog

# 作弊方法:手动对图像进行仿射变换,要怪就怪手写数据集太少吧...能yolo的话也不会有人用传统 ml 方法坐牢

# 全局变量
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
selected_region = None
original_image = None
transform_mode = False
points = []
region_x1, region_y1, region_x2, region_y2 = -1, -1, -1, -1  # 存储选择区域的坐标
transformed_image = None  # 存储变换后的图像

def select_image():
    """打开文件对话框选择图像"""
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="选择图像文件",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return cv2.imread(file_path) if file_path else None

def mouse_callback(event, x, y, flags, param):
    """鼠标事件回调函数"""
    global ix, iy, fx, fy, drawing, selected_region, original_image, transform_mode, points
    global region_x1, region_y1, region_x2, region_y2, transformed_image
    
    if transform_mode:
        # 仿射变换模式
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            # 将点转换到原始图像坐标
            abs_x = x + region_x1
            abs_y = y + region_y1
            points.append((abs_x, abs_y))
            # 在selected_region图像上显示点
            cv2.circle(selected_region, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Selected Region", selected_region)
            
            if len(points) == 3:
                apply_affine_transform()
        return
    
    # 区域选择模式
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = original_image.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", img_copy)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        cv2.rectangle(original_image, (ix, iy), (fx, fy), (0, 255, 0), 2)
        cv2.imshow("Image", original_image)
        
        # 确保坐标正确
        x1, y1 = min(ix, fx), min(iy, fy)
        x2, y2 = max(ix, fx), max(iy, fy)
        
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # 防止误选
            region_x1, region_y1 = x1, y1
            region_x2, region_y2 = x2, y2
            selected_region = original_image[y1:y2, x1:x2].copy()
            transformed_image = None  # 重置变换结果
            cv2.imshow("Selected Region", selected_region)
            print("区域已选择! 按 't' 进入变换模式")

def apply_affine_transform():
    """应用逆仿射变换校正扭曲区域"""
    global selected_region, points, original_image, transformed_image
    
    if len(points) != 3:
        print("需要选择3个点")
        return
    
    # 获取选择区域的尺寸
    height, width = selected_region.shape[:2]
    
    # 目标点 (校正后的矩形区域)
    dst_points = np.float32([[0, 0], [width, 0], [0, height]])
    
    # 源点 (用户选择的扭曲区域)
    src_points = np.float32(points)
    
    # 计算逆仿射变换矩阵
    M = cv2.getAffineTransform(src_points, dst_points)
    
    # 应用逆仿射变换校正图像
    transformed_image = cv2.warpAffine(
        original_image,  # 使用整个原始图像
        M, 
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(127, 127, 127)  # 灰色边框
    )
    
    cv2.imshow("Transformed Region", transformed_image)
    print("变换完成! 按 's' 保存结果")

# 主程序
def main():
    """主函数"""
    global original_image, transform_mode, points, selected_region, transformed_image
    
    # 选择并加载图像
    image = select_image()
    if image is None:
        print("未选择图像或加载失败!")
        return
    
    original_image = image.copy()
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)
    
    print("请选择区域: 鼠标拖拽绘制矩形")
    cv2.imshow("Image", original_image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC键退出
            break
        
        elif key == ord('t') and selected_region is not None:
            transform_mode = True
            points = []
            print("变换模式: 点击选择3个点 (左上, 右上, 左下)")
            cv2.setMouseCallback("Selected Region", mouse_callback)
        
        elif key == ord('s') and transformed_image is not None:
            # 保存校正后的结果
            root = Tk()
            root.withdraw()
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            root.destroy()
            if save_path:
                cv2.imwrite(save_path, transformed_image)
                print(f"结果已保存至: {save_path}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()