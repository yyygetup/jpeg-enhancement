import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 导入学术界最权威的指标计算工具
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

# 导入你的真 Mamba 网络
from sci_enhancer import SCIEnhancementNet

# 开启 CUDNN 极致加速
torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True

def tensor_to_uint8_numpy(tensor):
    """学术界严谨标准：将 Tensor 的 0~1 转换为严格的 uint8 0~255 格式再算指标"""
    img_np = tensor.squeeze(0).clamp(0, 1).cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0)) # [C, H, W] -> [H, W, C]
    img_np = np.round(img_np * 255.0).astype(np.uint8)
    return img_np

def evaluate_all_qualities(hq_dir, lq_base_dir, output_base_dir, weight_path, q_list=[10, 20, 30, 40]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 召唤你的满级 Mamba
    model = SCIEnhancementNet().to(device)
    if not os.path.exists(weight_path):
        print(f" 找不到权重文件: {weight_path}")
        return
        
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval() 
    print(f" 成功加载 Mamba 权重: {weight_path}\n")

    # 找到所有的原图（标准答案）
    hq_files = [f for f in os.listdir(hq_dir) if f.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg', '.tiff'))]
    if len(hq_files) == 0:
        print(" HQ测试集为空，请检查路径！")
        return

    transform = transforms.ToTensor()
    
    # 用来存储最终报告的字典
    report_dict = {}

    # 2. 开始逐个攻破各个难度的考卷
    with torch.no_grad():
        for q in q_list:
            lq_dir_q = os.path.join(lq_base_dir, f"q_{q}")
            out_dir_q = os.path.join(output_base_dir, f"q_{q}")
            
            if not os.path.exists(lq_dir_q):
                print(f" 找不到文件夹 {lq_dir_q}，跳过 Q={q} 的测试。")
                continue
                
            os.makedirs(out_dir_q, exist_ok=True)
            print(f"正在测试 Q={q} 的 ({len(hq_files)} 张图片)...")
            
            total_psnr_base, total_ssim_base = 0.0, 0.0
            total_psnr_ours, total_ssim_ours = 0.0, 0.0
            valid_imgs = 0
            
            # 使用 tqdm 加上炫酷进度条
            for hq_filename in tqdm(hq_files, desc=f"Testing Q={q}"):
                hq_path = os.path.join(hq_dir, hq_filename)
                
                # 注意：因为我们为了防二次压缩，生成的 LQ 统一保存为了 .png
                name, _ = os.path.splitext(hq_filename)
                lq_path = os.path.join(lq_dir_q, f"{name}.png")
                out_path = os.path.join(out_dir_q, f"{name}.png")
                
                if not os.path.exists(lq_path):
                    continue
                
                # 加载图片
                hq_img = Image.open(hq_path).convert('RGB')
                lq_img = Image.open(lq_path).convert('RGB')
                lq_tensor = transform(lq_img).unsqueeze(0).to(device)
                
                # enhanced_tensor, _ = model(lq_tensor, temperature=0.01)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    enhanced_tensor, _ = model(lq_tensor, temperature=0.01)
                
                # 转换格式准备给裁判打分
                hq_np = np.array(hq_img)
                lq_np = np.array(lq_img)
                enhanced_np = tensor_to_uint8_numpy(enhanced_tensor)
                
                # 算分：烂图 vs 原图 ( Baseline )
                total_psnr_base += calculate_psnr(hq_np, lq_np)
                # 注: 新版 skimage 使用 channel_axis=2 替代 multichannel=True
                total_ssim_base += calculate_ssim(hq_np, lq_np, channel_axis=2)
                
                # 算分：Mamba图 vs 原图 ( Ours )
                total_psnr_ours += calculate_psnr(hq_np, enhanced_np)
                total_ssim_ours += calculate_ssim(hq_np, enhanced_np, channel_axis=2)
                
                # 保存高清修复结果供肉眼查看
                Image.fromarray(enhanced_np).save(out_path)
                valid_imgs += 1
                
                del lq_tensor, enhanced_tensor
                torch.cuda.empty_cache()
                
            # 计算当前 Q 值的平均分
            if valid_imgs > 0:
                report_dict[q] = {
                    'psnr_base': total_psnr_base / valid_imgs,
                    'ssim_base': total_ssim_base / valid_imgs,
                    'psnr_ours': total_psnr_ours / valid_imgs,
                    'ssim_ours': total_ssim_ours / valid_imgs
                }

    # 3. 打印终极对比表格
    print("\n" + "="*70)
    print(f" Mamba JPEG Artifact Removal Benchmark ")
    print("="*70)
    print(f"| Quality | Metric | JPEG (Baseline) | Ours (Mamba) |   Boost   |")
    print("-" * 70)
    
    for q in q_list:
        if q not in report_dict: continue
        res = report_dict[q]
        psnr_diff = res['psnr_ours'] - res['psnr_base']
        ssim_diff = res['ssim_ours'] - res['ssim_base']
        
        print(f"|  Q={q:<2}   |  PSNR  |     {res['psnr_base']:<6.2f} dB   |    {res['psnr_ours']:<6.2f} dB  |  +{psnr_diff:<5.2f} dB |")
        print(f"|         |  SSIM  |     {res['ssim_base']:<8.4f}  |    {res['ssim_ours']:<8.4f} |  +{ssim_diff:<7.4f} |")
        print("-" * 70)
        
    print("="*70)
    print(f"所有高清修复结果已保存在: {output_base_dir} 文件夹下")

if __name__ == "__main__":
    # 路径配置 (确保你在 enhancement 文件夹下运行)
    TEST_HQ = r"./data/test_hq"              # 高清原图标准答案
    TEST_LQ_BASE = r"./data/test_lq"         # 刚刚生成的带 Q 档位马赛克的考卷
    OUTPUT_BASE = r"./results_test"          # Mamba 修复后输出的高清图
    
    # 换成你跑得最好的一轮权重 (比如 epoch_99 或者是你保存的最好权重)
    WEIGHT_FILE = r"./checkpoints/sci_model_epoch_99.pth" 
    
    # 开始测试
    evaluate_all_qualities(TEST_HQ, TEST_LQ_BASE, OUTPUT_BASE, WEIGHT_FILE, q_list=[10, 20, 30, 40])