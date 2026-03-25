import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from models import model_dict, TrainTask
from PIL import Image
import gc

def save_single_image(tensor, path):
    if tensor.dim() == 4: tensor = tensor.squeeze(0)
    if tensor.dim() == 3 and tensor.size(0) == 1: tensor = tensor.squeeze(0)
    arr = tensor.cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def load_moe_model():
    print("Loading MoE Model...")
    default_parser = TrainTask.build_default_options()
    default_parser.add_argument('--phase', type=str, default='test')
    default_opt, unknown_opt = default_parser.parse_known_args()
    
    default_opt.model_name = 'DUGAN_MoE'
    default_opt.run_name = '1201_MoE'
    default_opt.resume_iter = 200000
    default_opt.test_dataset_name = 'cmayo_test_512'
    default_opt.batch_size = 1
    default_opt.moe_phase = 2 # Important for Gating
    
    MODEL = model_dict['DUGAN_MoE']
    private_parser = MODEL.build_options()
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)
    opt.isTrain = False
    
    model = MODEL(opt)
    model.logger.load_checkpoints(200000)
    
    model.generator.eval()
    model.noise_discriminator.eval()
    model.gating_network.eval()
    
    return model

if __name__ == '__main__':
    out_dir = os.path.join('output', 'gating_analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    model = load_moe_model()
    
    # Store weights: [N, 3]
    all_alphas = []
    
    # Keep track of max indices
    max_noise_val = -1
    max_noise_idx = -1
    max_grad_val = -1
    max_grad_idx = -1
    
    # We need to save the images for the max cases, so we might need to cache them or re-run.
    # To save memory, let's just store the index and re-run specific indices later? 
    # Or just save them if they are the current max.
    
    print("Analyzing Gating Weights over Test Set...")
    
    # Limit samples to avoid taking forever if dataset is huge
    max_samples = 100 
    
    with torch.no_grad():
        for i, batch in enumerate(model.test_loader):
            if i >= max_samples: break
            
            if isinstance(batch, (list, tuple)):
                low_dose, full_dose = batch
            else:
                low_dose = batch
                full_dose = batch
                
            low_dose = low_dose.cuda()
            full_dose = full_dose.cuda()
            
            # Forward
            gen = model.generator(low_dose).clamp(0., 1.)
            noise_map = torch.abs(low_dose - full_dose)
            
            # Get Bottleneck
            _, _, bottleneck = model.noise_discriminator(low_dose, gen, noise_map)
            
            # Get Alphas
            residual = torch.abs(low_dose - gen)
            alphas = model.gating_network(gen, residual, bottleneck) # [B, 3]
            
            # Assuming B=1
            alpha_vals = alphas.cpu().numpy()[0] # [img, grad, noise]
            all_alphas.append(alpha_vals)
            
            # Check for Max Noise
            if alpha_vals[2] > max_noise_val:
                max_noise_val = alpha_vals[2]
                max_noise_idx = i
                save_single_image(low_dose, os.path.join(out_dir, "case_high_noise_input.png"))
                save_single_image(full_dose, os.path.join(out_dir, "case_high_noise_gt.png"))
                print(f"New Max Noise Alpha: {max_noise_val:.4f} at idx {i}")
                
            # Check for Max Grad
            if alpha_vals[1] > max_grad_val:
                max_grad_val = alpha_vals[1]
                max_grad_idx = i
                save_single_image(low_dose, os.path.join(out_dir, "case_high_grad_input.png"))
                save_single_image(full_dose, os.path.join(out_dir, "case_high_grad_gt.png"))
                print(f"New Max Grad Alpha: {max_grad_val:.4f} at idx {i}")

    # Plot Histograms
    all_alphas = np.array(all_alphas) # [N, 3]
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_alphas[:, 0], bins=20, alpha=0.5, label='Alpha Img', color='blue')
    plt.hist(all_alphas[:, 1], bins=20, alpha=0.5, label='Alpha Grad', color='green')
    plt.hist(all_alphas[:, 2], bins=20, alpha=0.5, label='Alpha Noise', color='red')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gating Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    hist_path = os.path.join(out_dir, 'gating_weights_histogram.png')
    plt.savefig(hist_path)
    print(f"Saved histogram to {hist_path}")
    
    # Save stats text
    with open(os.path.join(out_dir, 'gating_stats.txt'), 'w') as f:
        f.write(f"Analyzed {len(all_alphas)} samples.\n")
        f.write(f"Mean Alphas: Img={all_alphas[:,0].mean():.4f}, Grad={all_alphas[:,1].mean():.4f}, Noise={all_alphas[:,2].mean():.4f}\n")
        f.write(f"Max Noise Alpha: {max_noise_val:.4f} (Index {max_noise_idx})\n")
        f.write(f"Max Grad Alpha: {max_grad_val:.4f} (Index {max_grad_idx})\n")
        
    print(f"Done. Check {out_dir}")
