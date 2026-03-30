import torch
import torch.nn as nn
import torch.nn.functional as F

# 引入官方真 Mamba 算子
from mamba_ssm import Mamba

# ==========================================
# 1. 可微二值化边缘掩码分支 
# ==========================================
class BinarizedEdgeBranch(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        
    def forward(self, x, temperature=1.0):
        feat = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(feat))
        logits = self.conv_out(feat)
        mask_pred = torch.sigmoid(logits / temperature)
        return mask_pred, feat

# ==========================================
# 2. 空间特征调制模块 
# ==========================================
class SpatialFeatureModulation(nn.Module):
    def __init__(self, mask_channels=1, feat_channels=64):
        super().__init__()
        self.conv_gamma = nn.Conv2d(mask_channels, feat_channels, 3, padding=1)
        self.conv_beta = nn.Conv2d(mask_channels, feat_channels, 3, padding=1)
        
    def forward(self, x, mask):
        gamma = self.conv_gamma(mask)
        beta = self.conv_beta(mask)
        return x * (1 + gamma) + beta

# ==========================================
# 3. 基于官方引擎的·2D Mamba 模块
# ==========================================
class Real2DMambaBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        
        # 调用底层 CUDA 编写的 Mamba！无梯度消失，极速扫描！
        self.mamba = Mamba(
            d_model=channels, 
            d_state=16,       
            d_conv=4,         
            expand=2,         
        )
        self.linear_fuse = nn.Linear(channels * 4, channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2) 
        x_norm = self.norm(x_flat)
        
        # 4向交叉扫描
        scan_tl_br = x_norm
        scan_br_tl = torch.flip(x_norm, dims=[1])
        x_norm_t = x_norm.view(B, H, W, C).transpose(1, 2).reshape(B, H*W, C)
        scan_tr_bl = x_norm_t
        scan_bl_tr = torch.flip(x_norm_t, dims=[1])
        
        # 硬件级并行扫描，速度极快
        out_tl_br = self.mamba(scan_tl_br)
        out_br_tl = torch.flip(self.mamba(scan_br_tl), dims=[1]) 
        out_tr_bl = self.mamba(scan_tr_bl).view(B, W, H, C).transpose(1, 2).reshape(B, H*W, C)
        out_bl_tr = torch.flip(self.mamba(scan_bl_tr), dims=[1]).view(B, W, H, C).transpose(1, 2).reshape(B, H*W, C)
        
        fused = torch.cat([out_tl_br, out_br_tl, out_tr_bl, out_bl_tr], dim=-1)
        fused = self.linear_fuse(fused)
        
        out = fused.transpose(1, 2).view(B, C, H, W)
        return x + out

# ==========================================
# 4. 完整的屏幕内容增强网络
# ==========================================
class SCIEnhancementNet(nn.Module):
    def __init__(self, in_channels=3, feat_channels=64):
        super().__init__()
        self.edge_branch = BinarizedEdgeBranch(in_channels=in_channels)
        self.conv_in = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
        
        # 连续堆叠2个 Mamba Block 构成一个 Group
        self.mamba_group1 = nn.Sequential(*[Real2DMambaBlock(feat_channels) for _ in range(2)])
        self.sft1 = SpatialFeatureModulation(mask_channels=1, feat_channels=feat_channels)
        
        self.mamba_group2 = nn.Sequential(*[Real2DMambaBlock(feat_channels) for _ in range(2)])
        self.sft2 = SpatialFeatureModulation(mask_channels=1, feat_channels=feat_channels)
        
        self.conv_out = nn.Conv2d(feat_channels, in_channels, 3, padding=1)

    def forward(self, x, temperature=1.0):
        mask_pred, _ = self.edge_branch(x, temperature)
        feat = F.relu(self.conv_in(x))
        
        feat = self.mamba_group1(feat)
        feat = self.sft1(feat, mask_pred)
        
        feat = self.mamba_group2(feat)
        feat = self.sft2(feat, mask_pred)
        
        out = self.conv_out(feat) + x
        out = torch.clamp(out, min=0.0, max=1.0)
        return out, mask_pred

# ==========================================
# 5. 自定义损失函数 
# ==========================================
class SCILoss(nn.Module):
    # ... (此处保留你原有的 SCILoss 代码，完全不需要改动) ...
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def polarization_loss(self, mask_pred):
        return torch.mean(mask_pred * (1.0 - mask_pred))

    def transition_penalty_loss(self, pred_img, gt_img, gt_mask):
        local_max = F.max_pool2d(gt_img, kernel_size=3, stride=1, padding=1)
        local_min = -F.max_pool2d(-gt_img, kernel_size=3, stride=1, padding=1)
        dist_to_max = torch.abs(pred_img - local_max)
        dist_to_min = torch.abs(pred_img - local_min)
        min_dist = torch.min(dist_to_max, dist_to_min)
        transition_loss = torch.sum(min_dist * gt_mask) / (torch.sum(gt_mask) + 1e-6)
        return transition_loss

    def forward(self, pred_img, gt_img, mask_pred, gt_mask):
        loss_mse = self.mse(pred_img, gt_img)
        loss_polar = self.polarization_loss(mask_pred)
        loss_trans = self.transition_penalty_loss(pred_img, gt_img, gt_mask)
        total_loss = loss_mse + 0.001 * loss_polar + 0.005 * loss_trans
        return total_loss, loss_mse, loss_polar, loss_trans
    
    
    
# baseline model

class Baseline2DMambaBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        
        # 纯 Mamba 核心
        self.mamba = Mamba(
            d_model=channels, 
            d_state=16,       
            d_conv=4,         
            expand=2,         
        )
        #  删除了 self.linear_fuse (Concat+Linear 的核心)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2) 
        x_norm = self.norm(x_flat)
        
        scan_tl_br = x_norm
        scan_br_tl = torch.flip(x_norm, dims=[1])
        x_norm_t = x_norm.view(B, H, W, C).transpose(1, 2).reshape(B, H*W, C)
        scan_tr_bl = x_norm_t
        scan_bl_tr = torch.flip(x_norm_t, dims=[1])
        
        out_tl_br = self.mamba(scan_tl_br)
        out_br_tl = torch.flip(self.mamba(scan_br_tl), dims=[1]) 
        out_tr_bl = self.mamba(scan_tr_bl).view(B, W, H, C).transpose(1, 2).reshape(B, H*W, C)
        out_bl_tr = torch.flip(self.mamba(scan_bl_tr), dims=[1]).view(B, W, H, C).transpose(1, 2).reshape(B, H*W, C)
        
        #  核心改变：纯原生做法是直接将四向特征相加 (Summation)
        fused = out_tl_br + out_br_tl + out_tr_bl + out_bl_tr
        
        out = fused.transpose(1, 2).view(B, C, H, W)
        return x + out
    
    
class BaselineNet(nn.Module):
    def __init__(self, in_channels=3, feat_channels=64):
        super().__init__()
        #  删除了 self.edge_branch
        self.conv_in = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
        
        # 替换为刚刚写好的 Baseline Block
        self.mamba_group1 = nn.Sequential(*[Baseline2DMambaBlock(feat_channels) for _ in range(2)])
        #  删除了 self.sft1
        
        self.mamba_group2 = nn.Sequential(*[Baseline2DMambaBlock(feat_channels) for _ in range(2)])
        #  删除了 self.sft2
        
        self.conv_out = nn.Conv2d(feat_channels, in_channels, 3, padding=1)

    def forward(self, x):
        #  没有 mask_pred 的生成了
        feat = F.relu(self.conv_in(x))
        
        # 纯粹的串联前向传播
        feat = self.mamba_group1(feat)
        feat = self.mamba_group2(feat)
        
        out = self.conv_out(feat) + x
        out = torch.clamp(out, min=0.0, max=1.0)
        
        #  仅返回图像，不返回掩码
        return out