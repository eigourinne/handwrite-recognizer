# enhance.py
import cv2
import numpy as np
import argparse
import os

def enhance_black_lines(image_path, output_path=None, intensity=1.5, thickness=1.0):
    """
    增强白纸黑字图像中的黑色字符线条
    
    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径(可选)
        intensity: 线条增强强度(1.0-3.0)
        thickness: 线条加粗程度(1.0-2.0)
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return None
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 自适应阈值处理 - 更好地分离字符和背景
    binary = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        101, 15
    )
    
    # 形态学操作 - 连接断裂的笔画
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 笔画细化 - 获得更精确的线条位置
    thinned = cv2.ximgproc.thinning(
        connected, 
        thinningType=cv2.ximgproc.THINNING_GUOHALL
    )
    
    # 创建笔画增强掩码
    mask = np.zeros_like(gray, dtype=np.float32)
    mask[thinned > 0] = 1.0  # 笔画位置设为1
    
    # 应用高斯模糊使边缘平滑
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.5)
    
    # 增强笔画
    enhanced = img.astype(np.float32)
    
    # 在笔画区域应用颜色增强
    for c in range(3):
        # 在笔画区域加深颜色（使黑色更黑）
        enhanced[:, :, c] = np.where(
            mask > 0,
            np.clip(enhanced[:, :, c] * (1 - mask * intensity * 0.5), 0, 255),
            enhanced[:, :, c]
        )
    
        # 在非笔画区域提亮背景（使白色更白）
        enhanced[:, :, c] = np.where(
            mask < 0.1,
            np.clip(enhanced[:, :, c] * (1 + (1 - mask) * intensity * 0.2), 0, 255),
            enhanced[:, :, c]
        )
    
    # 加粗线条（可选）
    if thickness > 1.0:
        dilation_size = int(1 * (thickness - 1.0))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (2 * dilation_size + 1, 2 * dilation_size + 1)
        )
        mask_dilated = cv2.dilate(mask, kernel)
        
        for c in range(3):
            enhanced[:, :, c] = np.where(
                mask_dilated > 0,
                np.clip(enhanced[:, :, c] * (1 - mask_dilated * intensity * 0.3), 0, 255),
                enhanced[:, :, c]
            )
    
    # 转换为8位图像
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    # 保存结果
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_enhanced{ext}"
    
    cv2.imwrite(output_path, enhanced)
    print(f"增强后的图像已保存至: {output_path}")
    
    # 显示结果（可选）
    cv2.imshow("raw", img)
    cv2.imshow("after", enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return enhanced

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='白纸黑字图像线条增强器 - 增强黑色字符线条'
    )
    parser.add_argument(
        'image_path', 
        type=str, 
        help='输入图像路径'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None, 
        help='输出图像路径(可选)'
    )
    parser.add_argument(
        '--intensity', 
        type=float, 
        default=1.5, 
        help='线条增强强度(1.0-3.0)'
    )
    parser.add_argument(
        '--thickness', 
        type=float, 
        default=1.5, 
        help='线条加粗程度(1.0-2.0)'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 执行增强
    enhanced_image = enhance_black_lines(
        args.image_path,
        args.output,
        args.intensity,
        args.thickness
    )
    
    if enhanced_image is not None:
        print("图像增强完成！")

if __name__ == '__main__':
    main()