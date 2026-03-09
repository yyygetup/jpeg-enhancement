import os
import random
import shutil
from tqdm import tqdm

def split_dataset(source_dir, train_dir, test_dir, split_ratio=0.8):
    """
    将一个文件夹里的图片按比例随机划分为训练集和测试集
    source_dir: 原始图片总文件夹
    train_dir: 输出的训练集文件夹
    test_dir: 输出的测试集文件夹
    split_ratio: 训练集所占比例 (默认 0.8，即 80% 训练，20% 测试)
    """
    # 1. 安全检查与创建文件夹
    if not os.path.exists(source_dir):
        print(f"找不到原始文件夹: {source_dir}")
        return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 2. 获取所有图片文件
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    all_files = [f for f in os.listdir(source_dir) if f.lower().endswith(supported_formats)]
    
    total_imgs = len(all_files)
    if total_imgs == 0:
        print("原始文件夹里没有图片！")
        return

    print(f"📦 找到总计 {total_imgs} 张图片。")

    # 3. 【核心】打乱顺序，保证随机性
    random.seed(42)  # 固定随机种子，保证每次运行划分结果一样
    random.shuffle(all_files)

    # 4. 计算分割点
    train_count = int(total_imgs * split_ratio)
    train_files = all_files[:train_count]
    test_files = all_files[train_count:]

    print(f"🔪 按照 {split_ratio*100}% : {(1-split_ratio)*100}% 比例划分...")
    print(f"   -> 训练集将分配: {len(train_files)} 张")
    print(f"   -> 测试集将分配: {len(test_files)} 张")

    # 5. 复制文件到对应目录
    print("\n开始复制文件到训练集 (train_hq)...")
    for f in tqdm(train_files):
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))

    print("开始复制文件到测试集 (test_hq)...")
    for f in tqdm(test_files):
        shutil.copy(os.path.join(source_dir, f), os.path.join(test_dir, f))

    print("\n数据集划分完成！")

if __name__ == "__main__":
    # 假设你把刚下载或截图的几百张原始高清图放在了这个文件夹：
    SOURCE_FOLDER = r"./data/sdu_raw_images" 
    
    # 脚本会自动帮你生成这两个文件夹：
    TRAIN_HQ = r"./data/train_hq_SDU"  # 给 train.py 用
    TEST_HQ = r"./data/test_hq"    # 给上一回合的测试集生成脚本用
    
    # 0.8 表示 80% 的图片去训练，20% 的图片留作测试
    split_dataset(SOURCE_FOLDER, TRAIN_HQ, TEST_HQ, split_ratio=0.8)