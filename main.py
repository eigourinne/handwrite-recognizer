# main.py
import cv2
import numpy as np
from recognize import DigitRecognizer
import torch

def visualize(image, predictions, output_path="result.jpg"):
    """改进的可视化函数：显示长宽比信息"""
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colors = [(0,255,0), (0,0,255), (255,0,0), 
              (255,255,0), (0,255,255), (255,0,255),
              (128,255,0), (0,128,255), (255,0,128)]
    
    recognized_digits = []
    for i, (pred, rect, conf) in enumerate(predictions):
        # 检查是否为旋转矩形（4个点）还是水平矩形（4个值）
        if isinstance(rect, tuple) and len(rect) == 4:
            # 水平矩形
            x, y, w, h = rect
            points = np.array([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
        elif len(rect) == 4 and isinstance(rect, np.ndarray):
            # 旋转矩形（4个点）
            points = rect
            # 计算最小外接矩形的宽高
            rect = cv2.minAreaRect(points)
            w, h = rect[1]
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
        else:
            continue
            
        color = colors[i % len(colors)]
        
        # 绘制边界框（多边形）
        cv2.polylines(img, [points], True, color, 2)
        
        # 计算边界框中心点
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
        
        # 绘制标签和置信度
        conf_float = float(conf) if isinstance(conf, (str, torch.Tensor)) else conf
        label = f"{pred}({conf_float:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # 文本背景
        text_bg_top_left = (center_x - tw//2, center_y - th - 10)
        text_bg_bottom_right = (center_x + tw//2, center_y)
        cv2.rectangle(img, text_bg_top_left, text_bg_bottom_right, color, -1)
        
        # 文本
        cv2.putText(img, label, (center_x - tw//2, center_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # 显示长宽比信息（调试用）
        # aspect_text = f"AR:{aspect_ratio:.1f}"
        # cv2.putText(img, aspect_text, (center_x - 15, center_y + 15), 
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        recognized_digits.append((pred, conf))
    
    # 添加汇总信息
    summary = " ".join([f"{p}({c:.2f})" for p, c in recognized_digits])
    cv2.putText(img, f"Recognized: {summary}", (10, img.shape[0]-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imwrite(output_path, img)
    print("识别结果:", summary)
    print(f"结果图已保存至: {output_path}")

def main():
    try:
        recognizer = DigitRecognizer("best_model.pth")
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

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
                visualize(img, results)
            else:
                print("未检测到有效数字。请检查图像质量。")

        except Exception as e:
            print(f"处理图像失败: {e}")

    print("\n程序结束。")

if __name__ == '__main__':
    main()