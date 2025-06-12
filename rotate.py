# rotate.py
import cv2
import numpy as np
import os
import sys

# 之前尝试在 recognize.py 中使用二分法从 -45 到 45 度中自动迭代,然而没有合适的量度,效果简直依托,学习不了一点...固使用传统手工业,毕竟特征提取属于人类智慧

def rotate_and_crop_image(image_path, angle):
    """
    旋转图像并裁剪成矩形
    :param image_path: 图像路径
    :param angle: 旋转角度（正数表示逆时针旋转）
    :return: 旋转并裁剪后的图像
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return None
    
    # 获取图像尺寸
    (h, w) = img.shape[:2]
    
    # 计算旋转中心
    center = (w // 2, h // 2)
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后图像的边界
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # 计算新图像的宽高
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # 调整旋转矩阵的平移部分
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    
    # 执行旋转
    rotated = cv2.warpAffine(img, M, (nW, nH))
    
    # 裁剪旋转后的图像为矩形
    # 计算裁剪边界（保留最大内接矩形）
    crop_h = int(h * (cos + sin))
    crop_w = int(w * (cos + sin))
    
    # 确保裁剪尺寸不超过旋转后图像尺寸
    crop_h = min(crop_h, nH)
    crop_w = min(crop_w, nW)
    
    # 计算裁剪起点
    start_x = (nW - crop_w) // 2
    start_y = (nH - crop_h) // 2
    
    # 执行裁剪
    cropped = rotated[start_y:start_y+crop_h, start_x:start_x+crop_w]
    
    return cropped

def main():
    if len(sys.argv) < 2:
        print("用法: python adjust.py <图像路径> [旋转角度]")
        print("示例: python adjust.py input.jpg 5")
        return
    
    image_path = sys.argv[1]
    angle = 0  # 默认不旋转
    
    if len(sys.argv) >= 3:
        try:
            angle = float(sys.argv[2])
        except ValueError:
            print("错误: 旋转角度必须是数字")
            return
    
    # 旋转并裁剪图像
    result = rotate_and_crop_image(image_path, angle)
    
    if result is not None:
        # 生成输出文件名
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = f"{name}_adjusted{ext}"
        
        # 保存结果
        cv2.imwrite(output_path, result)
        print(f"已保存矫正后的图像: {output_path}")

if __name__ == "__main__":
    main()