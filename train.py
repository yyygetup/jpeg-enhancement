# enhancement/train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torch import amp
from tqdm import tqdm
import torchvision

from sci_enhancer import SCIEnhancementNet, SCILoss,BaselineNet
from dataset import SCIDataset

# 真 Mamba 极度依赖 CUDNN 加速，绝对要开启！
torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True 

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    
    # 【服务器专属配置】
    batch_size = 8       # 显存大可以开到 16 或 32
    patch_size = 256     # 序列变长，真 Mamba 轻松拿捏
    num_workers = 8      # Linux 下开启多线程加载，告别 GPU 等待 CPU
    
    learning_rate = 2e-4
    
    hq_folder = "./data/train_hq_SCID" 
    lq_folder = "./data/train_lq_SCID" 
    
    dataset = SCIDataset(hq_dir=hq_folder, patch_size=patch_size)
    # pin_memory=True 开启锁页内存，数据向 GPU 传输更快
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    
    #--------消融实验-------------
    
    is_baseline = True  # 【改成 True 跑 Model A，改成 False 跑完全体】
    
    if is_baseline:
        print(" 正在训练: BaselineNet (纯原生 Mamba, 仅 MSE Loss)")
        model = BaselineNet().to(device)
        criterion = torch.nn.MSELoss().to(device) # Baseline 只能用最普通的 MSE
    else:
        print(" 正在训练: SCIEnhancementNet (完全体, 含混合 Loss)")
        model = SCIEnhancementNet().to(device)
        criterion = SCILoss().to(device)
    
    #---------------------------------
    
    # criterion = SCILoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    scaler = amp.GradScaler('cuda')

    print(f" | 设备: {device} | 训练集数量: {len(dataset)} 张")

    for epoch in range(num_epochs):
        model.train()
        current_temp = 1.0 if epoch < 20 else max(0.01, 1.0 * (0.9 ** (epoch - 20)))
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}] Temp:{current_temp:.2f}")

        for batch_idx, (lq_img, hq_img, gt_mask) in enumerate(pbar):
            lq_img, hq_img, gt_mask = lq_img.to(device), hq_img.to(device), gt_mask.to(device)

            optimizer.zero_grad()
            
            with amp.autocast('cuda'):
                # pred_img, mask_pred = model(lq_img, temperature=current_temp)
                # total_loss, loss_mse, loss_polar, loss_trans = criterion(
                #     pred_img, hq_img, mask_pred, gt_mask
                # )
                if is_baseline:
                    # Baseline 专属前向传播：单输入，单输出，普通 Loss
                    pred_img = model(lq_img)
                    total_loss = criterion(pred_img, hq_img)
                    loss_mse = total_loss # 为了能在终端打印，把 total_loss 赋给 loss_mse
                else:
                    # 完全体专属前向传播：带温度参数，双输出，混合 Loss
                    pred_img, mask_pred = model(lq_img, temperature=current_temp)
                    total_loss, loss_mse, loss_polar, loss_trans = criterion(
                        pred_img, hq_img, mask_pred, gt_mask
                    )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            pbar.set_postfix({'Loss': f"{total_loss.item():.4f}", 'MSE': f"{loss_mse.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f" Epoch {epoch} 结束 | 平均 Loss: {avg_loss:.4f}\n")
        
        os.makedirs("results", exist_ok=True)
        
        # 把低质图、网络预测图、高清原图 拼成一排保存下来
        compare_img = torch.cat([lq_img[:4], pred_img[:4], hq_img[:4]], dim=0) 
        torchvision.utils.save_image(compare_img, f"results/epoch_{epoch}.png", nrow=4)
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/sci_model_epoch_{epoch}.pth")
        
if __name__ == "__main__":
    train()