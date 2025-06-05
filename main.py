# main.py
import cv2
import numpy as np
from recognize import DigitRecognizer
import torch

def visualize(image, predictions, output_path="result.jpg"):
    """改进的可视化函数"""
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colors = [(0,255,0), (0,0,255), (255,0,0), 
              (255,255,0), (0,255,255), (255,0,255),
              (128,255,0), (0,128,255), (255,0,128)]
    
    recognized_digits = []
    for i, (pred, rect, conf) in enumerate(predictions):
        x, y, w, h = rect
        color = colors[i % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        
        # 绘制标签和置信度
        conf_float = float(conf) if isinstance(conf, (str, torch.Tensor)) else conf
        label = f"{pred}({float(conf_float):.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x, y-th-10), (x+tw, y), color, -1)
        cv2.putText(img, label, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
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