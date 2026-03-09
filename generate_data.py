# generate_data.py
import os
import random
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

def create_synthetic_dataset(num_images=3000, size=(256, 256)):
    hq_dir = "./data/train_hq"
    lq_dir = "./data/train_lq"
    os.makedirs(hq_dir, exist_ok=True)
    os.makedirs(lq_dir, exist_ok=True)

    print(f" 开始生成 {num_images} 张合成训练数据...")

    # 尝试加载系统默认字体，如果找不到就用默认的
    try:
        font = ImageFont.truetype("arial.ttf", size=random.randint(12, 36))
    except:
        font = ImageFont.load_default()

    for i in range(num_images):
        # 1. 生成随机纯色背景
        bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        img = Image.new('RGB', size, color=bg_color)
        draw = ImageDraw.Draw(img)

        # 2. 随机画一些粗细不同的矩形和线条（模拟表格和网页边框）
        for _ in range(random.randint(2, 6)):
            x0, y0 = random.randint(0, size[0]//2), random.randint(0, size[1]//2)
            x1, y1 = random.randint(size[0]//2, size[0]), random.randint(size[1]//2, size[1])
            color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            draw.rectangle([x0, y0, x1, y1], outline=color, width=random.randint(1, 4))
            
        # 3. 随机写一些文字（模拟屏幕内容）
        for _ in range(random.randint(3, 8)):
            tx, ty = random.randint(0, size[0]-50), random.randint(0, size[1]-20)
            text_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
            # 随机生成一些字母和数字
            text = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=random.randint(3, 15)))
            draw.text((tx, ty), text, fill=text_color, font=font)

        # 4. 保存高清原图 (HQ) - 无损 PNG
        img_name = f"synth_{i:05d}.png"
        hq_path = os.path.join(hq_dir, img_name)
        img.save(hq_path, format="PNG")

        # 5. 在内存中进行极低质量的 JPEG 压缩 (模拟严重的马赛克和振铃)
        buffer = BytesIO()
        jpeg_quality = random.randint(10, 30) # 质量越低，马赛克越重
        img.save(buffer, format="JPEG", quality=jpeg_quality)
        
        # 6. 读取劣化的 JPEG 并保存为 LQ 图 (依然存成 PNG 避免二次压缩)
        buffer.seek(0)
        lq_img = Image.open(buffer).convert('RGB')
        lq_path = os.path.join(lq_dir, img_name)
        lq_img.save(lq_path, format="PNG")

        if (i + 1) % 500 == 0:
            print(f"已生成 {i + 1} / {num_images} 张...")

    print("数据集生成完毕！请重新运行 train.py 开启真正的炼丹！")

if __name__ == "__main__":
    create_synthetic_dataset(num_images=3000)