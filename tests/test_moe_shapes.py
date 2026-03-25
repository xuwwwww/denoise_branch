import torch
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.DUGAN.noise_discriminator import NoiseDiscriminator
from models.DUGAN.gating_network import GatingNetwork
from models.DUGAN.DUGAN_MoE import DUGAN_MoE

def test_moe_shapes():
    print("Testing NoiseDiscriminator shapes...")
    B, C, H, W = 2, 1, 128, 128
    x_noisy = torch.randn(B, C, H, W).cuda()
    y_hat = torch.randn(B, C, H, W).cuda()
    noise_level_map = torch.abs(x_noisy - y_hat)
    
    model = NoiseDiscriminator().cuda()
    adv_map, noise_map, bottleneck = model(x_noisy, y_hat, noise_level_map)
    
    print("Adv map shape:", adv_map.shape)
    print("Noise map shape:", noise_map.shape)
    print("Bottleneck shape:", bottleneck.shape)
    
    assert adv_map.shape == (B, 1, H, W)
    assert noise_map.shape == (B, 1, H, W)
    
    print("Testing GatingNetwork shapes...")
    residual_abs = torch.abs(x_noisy - y_hat)
    gate = GatingNetwork(bottleneck_channels=bottleneck.size(1)).cuda()
    
    alphas = gate(y_hat, residual_abs, bottleneck)
    print("Alphas shape:", alphas.shape)
    assert alphas.shape == (B, 3)
    
    loss_balance = gate.balance_loss(alphas)
    print("Balance loss:", loss_balance.item())
    
    print("All shape tests passed!")

def test_dugan_moe_run():
    print("\nTesting DUGAN_MoE forward pass...")
    # Mock options
    class Opt:
        num_layers = 10
        num_channels = 32
        g_lr = 1e-4
        d_lr = 1e-4
        d_iter = 1
        cutmix_prob = 0.5
        img_gen_loss_weight = 0.1
        grad_gen_loss_weight = 0.1
        pix_loss_weight = 1.0
        grad_loss_weight = 20.0
        cr_loss_weight = 1.0
        cutmix_warmup_iter = 1000
        use_grad_discriminator = True
        moving_average = 0.999
        repeat_num = 6
        weight_decay = 0.
        moe_phase = 2
        lambda_noise_fixed = 0.3
        lambda_gate_balance = 0.01
        lambda_noise_reg = 1.0
        num_experts = 3
        
    opt = Opt()
    
    # Mock Logger
    class MockLogger:
        def __init__(self):
            self.modules = []
        def msg(self, d, i):
            print(f"Iter {i}: {d}")
        def save_image(self, img, i, tag):
            pass
            
    # Instantiate
    model = DUGAN_MoE(opt, MockLogger())
    model.set_model()
    
    # Dummy inputs
    B, C, H, W = 2, 1, 64, 64
    low_dose = torch.randn(B, C, H, W)
    full_dose = torch.randn(B, C, H, W)
    
    # Train step
    model.train([low_dose, full_dose], n_iter=1)
    print("DUGAN_MoE train step successful!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_moe_shapes()
        test_dugan_moe_run()
    else:
        print("CUDA not available, skipping test.")
