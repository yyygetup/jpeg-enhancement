import os
import cv2
import numpy as np

def generate_standard_test_set(hq_dir, base_lq_dir, q_list=[10, 20, 30, 40]):
    # 防迷路检测
    if not os.path.exists(hq_dir):
        print(f"请把新的高清测试截图放进 {hq_dir} 然后重试！")
        return

    files = os.listdir(hq_dir)
    if len(files) == 0:
        print("文件夹是空的！")
        return

    # 遍历你要测试的每一个 JPEG 质量档位
    for q in q_list:
        # 为每个质量档位建一个单独的文件夹
        lq_dir_q = os.path.join(base_lq_dir, f"q_{q}")
        os.makedirs(lq_dir_q, exist_ok=True)
        print(f"\n开始生成固定质量 Q={q} 的测试集...")

        for filename in files:
            if not filename.lower().endswith(('.png', '.bmp', '.tiff', '.jpg')):
                continue
                
            hq_path = os.path.join(hq_dir, filename)
            lq_path = os.path.join(lq_dir_q, filename) 
            
            img_bgr = cv2.imread(hq_path)
            if img_bgr is None:
                continue

            # 【核心修改】：不再随机！使用固定的 q 值进行压缩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            success, encimg = cv2.imencode('.jpg', img_bgr, encode_param)

            if success:
                decoded_bgr = cv2.imdecode(encimg, 1)
                cv2.imwrite(lq_path, decoded_bgr)
            else:
                print(f"处理失败: {filename}")

if __name__ == "__main__":
    # 指向你准备好的几十张“全新、没参与过训练”的高清图
    HQ_FOLDER = r"./data/test_hq"
    # 生成的测试集存放目录
    LQ_FOLDER = r"./data/test_lq"
    
    print("开始生成固定质量测试集...")
    generate_standard_test_set(HQ_FOLDER, LQ_FOLDER)
    print("代码执行结束！")