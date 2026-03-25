import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    def __init__(self, bottleneck_channels=512, hidden_dim=128, num_experts=3, num_classes=None):
        super(GatingNetwork, self).__init__()
        
        # Inputs:
        # 1. GAP(y_hat) -> 1
        # 2. GAP(|x_noisy - y_hat|) -> 1
        # 3. GAP(bottleneck_feature) -> bottleneck_channels
        
        input_dim = 1 + 1 + bottleneck_channels
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Gating Head
        self.gate_head = nn.Linear(hidden_dim, num_experts)
        
        # Optional Noise Classification Head
        self.num_classes = num_classes
        if num_classes is not None:
            self.cls_head = nn.Linear(hidden_dim, num_classes)
        else:
            self.cls_head = None

    def forward(self, y_hat, residual_abs, bottleneck):
        """
        Args:
            y_hat:       [B, 1, H, W]
            residual_abs:[B, 1, H, W]
            bottleneck:  [B, C_b, H_b, W_b]
        return:
            alphas: Tensor [B, 3] corresponding to (alpha_img, alpha_grad, alpha_noise)
        """
        B = y_hat.size(0)
        
        # 1. Global Pooling
        gap_y = torch.mean(y_hat.view(B, -1), dim=1, keepdim=True) # [B, 1]
        gap_res = torch.mean(residual_abs.view(B, -1), dim=1, keepdim=True) # [B, 1]
        gap_bot = torch.mean(bottleneck.view(B, bottleneck.size(1), -1), dim=2) # [B, C_b]
        
        # 2. Concat
        feat = torch.cat([gap_y, gap_res, gap_bot], dim=1) # [B, 2+C]
        
        # 3. MLP
        h = self.net(feat)
        
        # 4. Heads
        logits = self.gate_head(h)
        alphas = F.softmax(logits, dim=1) # [B, 3]
        
        # Optional classification
        # if self.cls_head is not None:
        #     cls_logits = self.cls_head(h)
            
        return alphas

    def balance_loss(self, alphas):
        """
        alphas: [B, 3]
        Returns a scalar, calculating the distance between batch mean and uniform distribution (1/3, 1/3, 1/3)
        """
        mean = alphas.mean(dim=0)          # [3]
        target = torch.full_like(mean, 1.0 / 3.0)
        return torch.sum((mean - target) ** 2)
