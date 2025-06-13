# main.py
import cv2
import numpy as np
from recognize import DigitRecognizer
import sys

def visualize(image, predictions, output_path="result.jpg"):
    """可视化函数 - 解决置信度显示不完整问题"""
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colors = [(0,255,0), (0,0,255), (255,0,0), 
              (255,255,0), (0,255,255), (255,0,255),
              (128,255,0), (0,128,255), (255,0,128)]
    
    recognized_digits = []
    for i, (pred, box, conf) in enumerate(predictions):
        color = colors[i % len(colors)]
        
        # 绘制旋转矩形边界框
        cv2.drawContours(img, [box.astype(int)], 0, color, 2)

        # 计算中心点用于标签放置
        center_x = int(np.mean(box[:, 0]))
        center_y = int(np.mean(box[:, 1]))
        
        # 简化标签显示
        label = f"{pred}({conf:.2f})"
        font_scale = max(0.5, min(1.0, min(img.shape[:2]) / 600))  # 自适应字体大小
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        
        # 标签背景位置
        label_x = max(0, center_x - tw // 2)
        label_y = max(th + 5, center_y - 10)
        
        # 确保标签在图像范围内
        label_x = min(label_x, img.shape[1] - tw - 5)
        label_y = min(label_y, img.shape[0] - 5)
        
        cv2.rectangle(img, (label_x, label_y - th - 5), 
                     (label_x + tw, label_y), color, -1)
        cv2.putText(img, label, (label_x, label_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 2)
        
        recognized_digits.append((pred, conf))
    
    # 添加汇总信息（仅显示识别结果）
    if recognized_digits:
        summary = " ".join([f"{p}" for p, c in recognized_digits])
        cv2.putText(img, f"识别结果: {summary}", (10, img.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imwrite(output_path, img)
    print(f"结果图已保存至: {output_path}")
    return recognized_digits

def main():
        
    recognizer = DigitRecognizer("best_model.pth")
    print("模型加载成功！")

    print("\n--- 增强版手写数字识别系统 ---")
    print("输入图像路径进行识别，输入 'q' 退出。")
        
    while True:
        path = input("\n输入图像路径(或q退出): ").strip()
        if path.lower() == 'q':
            break

        try:
            print(f"正在处理图像: {path}...")
            results, img = recognizer.find_and_recognize(path)

            if results:
                recognized_digits = visualize(img, results)
                print(f"最终识别结果: {' '.join(str(p) for p, c in recognized_digits)}")
            else:
                print("未检测到有效数字。请检查图像质量。")

        except Exception as e:
            print(f"处理图像失败: {e}")

    print("\n程序结束。")

if __name__ == '__main__':
    main()