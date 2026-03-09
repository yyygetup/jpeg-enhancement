import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 导入权威的指标计算函数
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

from sci_enhancer import SCIEnhancementNet

# 开启 CUDNN 加速
torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True

def tensor_to_uint8_numpy(tensor):
    """学术界标准：将 Tensor (0~1) 转换为严格的 uint8 (0~255) 格式再算指标"""
    img_np = tensor.squeeze(0).clamp(0, 1).cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0)) # [C, H, W] -> [H, W, C]
    img_np = np.round(img_np * 255.0).astype(np.uint8)
    return img_np

def evaluate_dataset(lq_dir, hq_dir, output_dir, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 正在使用 {device} 进行批量科学评估...")

    # 1. 加载模型
    model = SCIEnhancementNet().to(device)
    if not os.path.exists(weight_path):
        print(f" 找不到权重文件: {weight_path}")
        return
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval() 
    print(f" 成功加载 Mamba 权重: {weight_path}")

    os.makedirs(output_dir, exist_ok=True)
    
    image_names = [f for f in os.listdir(hq_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if len(image_names) == 0:
        print(" 测试集为空，请检查路径！")
        return

    # 初始化指标累加器
    total_psnr_baseline, total_ssim_baseline = 0.0, 0.0
    total_psnr_mamba, total_ssim_mamba = 0.0, 0.0
    
    transform = transforms.ToTensor()

    print(f" 开始评估，共 {len(image_names)} 张测试图...")
    pbar = tqdm(image_names)
    
    with torch.no_grad():
        for img_name in pbar:
            lq_path = os.path.join(lq_dir, img_name)
            hq_path = os.path.join(hq_dir, img_name)
            out_path = os.path.join(output_dir, img_name)
            
            # 读取图片
            lq_img = Image.open(lq_path).convert('RGB')
            hq_img = Image.open(hq_path).convert('RGB')
            
            lq_tensor = transform(lq_img).unsqueeze(0).to(device)
            
            # Mamba 推理
            enhanced_tensor, _ = model(lq_tensor, temperature=0.01)
            
            # 转换为严格的 NumPy 数组计算客观指标
            lq_np = np.array(lq_img)
            hq_np = np.array(hq_img)
            enhanced_np = tensor_to_uint8_numpy(enhanced_tensor)
            
            # ---------------------------------------------------------
            # 指标计算 1：Baseline (原始 JPEG 到底有多差)
            # ---------------------------------------------------------
            psnr_base = calculate_psnr(hq_np, lq_np)
            ssim_base = calculate_ssim(hq_np, lq_np, channel_axis=2)
            
            # ---------------------------------------------------------
            # 指标计算 2：Ours (Mamba 到底修复了多少)
            # ---------------------------------------------------------
            psnr_mamba = calculate_psnr(hq_np, enhanced_np)
            ssim_mamba = calculate_ssim(hq_np, enhanced_np, channel_axis=2)
            
            # 累加分数
            total_psnr_baseline += psnr_base
            total_ssim_baseline += ssim_base
            total_psnr_mamba += psnr_mamba
            total_ssim_mamba += ssim_mamba
            
            # 保存修复后的图片用于肉眼观察
            Image.fromarray(enhanced_np).save(out_path)
            
            # 更新进度条显示当前图的信息
            pbar.set_postfix({'PSNR_Boost': f"+{psnr_mamba - psnr_base:.2f}dB"})

    # 计算平均分
    avg_psnr_base = total_psnr_baseline / len(image_names)
    avg_ssim_base = total_ssim_baseline / len(image_names)
    avg_psnr_mamba = total_psnr_mamba / len(image_names)
    avg_ssim_mamba = total_ssim_mamba / len(image_names)

    # 打印最终霸气的对比报告
    print("\n" + "="*50)
    print(" 测试集评估报告 (Objective Evaluation)")
    print("="*50)
    print(f" 测试图片数量: {len(image_names)}")
    print("-" * 50)
    print(" 基线 (未经处理的 JPEG 失真图):")
    print(f"   - Average PSNR : {avg_psnr_base:.2f} dB")
    print(f"   - Average SSIM : {avg_ssim_base:.4f}")
    print("-" * 50)
    print(" Ours (真 Mamba 增强后的图像):")
    print(f"   - Average PSNR : {avg_psnr_mamba:.2f} dB  ( 提升: {avg_psnr_mamba - avg_psnr_base:+.2f} dB)")
    print(f"   - Average SSIM : {avg_ssim_mamba:.4f}  ( 提升: {avg_ssim_mamba - avg_ssim_base:+.4f})")
    print("="*50)
    print(f" 修复后的高清图片已批量存入: {output_dir}")

if __name__ == "__main__":
    #  测试集文件夹！
    TEST_HQ_DIR = "./data/test_hq"  
    TEST_LQ_DIR = "./data/test_lq"  
    OUTPUT_DIR = "./results_test"
    
    # 指向跑出来的 100 轮最好权重
    WEIGHT_FILE = "./checkpoints/sci_model_epoch_99.pth" 
    
    evaluate_dataset(TEST_LQ_DIR, TEST_HQ_DIR, OUTPUT_DIR, WEIGHT_FILE)