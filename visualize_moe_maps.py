import os
import torch
import torchvision
from models import model_dict, TrainTask
import numpy as np
from PIL import Image

def normalize_to_01(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-6)

def save_single_image(tensor, path):
    # tensor: [1, H, W] or [H, W]
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    
    # Normalize to 0-255
    arr = tensor.cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(path)
    print(f"Saved {path}")

if __name__ == '__main__':
    # Parse arguments
    default_parser = TrainTask.build_default_options()
    default_parser.add_argument('--phase', type=str, default='test', help='train, test')
    default_opt, unknown_opt = default_parser.parse_known_args()
    
    # Hardcode options for MoE visualization
    default_opt.model_name = 'DUGAN_MoE'
    default_opt.run_name = '1201_MoE' # Matches DUGAN_MoE_1201_MoE folder
    default_opt.resume_iter = 200000
    default_opt.test_dataset_name = 'cmayo_test_512' # Use full 512x512 images for better visualization
    default_opt.batch_size = 1
    
    # Load Model
    MODEL = model_dict[default_opt.model_name]
    private_parser = MODEL.build_options()
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)
    
    opt.isTrain = False
    opt.moe_phase = 2 # Ensure all parts are active
    
    # Initialize Model
    model = MODEL(opt)
    
    # Load Checkpoint
    if opt.resume_iter > 0:
        model.logger.load_checkpoints(opt.resume_iter)
    
    model.generator.eval()
    if hasattr(model, 'noise_discriminator'):
        model.noise_discriminator.eval()
    
    # Create output dir
    vis_dir = os.path.join('output', f'DUGAN_MoE_{opt.run_name}', 'visualization_maps')
    os.makedirs(vis_dir, exist_ok=True)
    
    print("Generating visualization maps...")
    
    # Get a sample
    # We can use model.test_images (fixed batch) or iterate loader
    # Let's iterate loader to get a few examples
    
    count = 0
    max_samples = 5
    
    with torch.no_grad():
        for i, batch in enumerate(model.test_loader):
            if count >= max_samples:
                break
                
            if isinstance(batch, (list, tuple)):
                low_dose, full_dose = batch
            else:
                low_dose = batch
                full_dose = batch
            
            low_dose = low_dose.cuda()
            full_dose = full_dose.cuda()
            
            # Forward Generator
            gen_full_dose = model.generator(low_dose).clamp(0., 1.)
            
            # Forward Noise Discriminator
            if hasattr(model, 'noise_discriminator'):
                noise_level_map = torch.abs(low_dose - full_dose)
                # D_noise(x_noisy, y_hat, noise_map)
                adv_map, noise_pred_map, _ = model.noise_discriminator(low_dose, gen_full_dose, noise_level_map)
                
                # Also get Real Noise Map (Ground Truth for Regression)
                # adv_real, noise_real, _ = model.noise_discriminator(low_dose, full_dose, noise_level_map)
                # Actually noise_level_map IS the ground truth for regression
                
                # Normalize maps for visualization
                # adv_map is [0, 1] from Sigmoid.
                # If it's all gray (approx 0.5), it means D is unsure.
                # We normalize it to [0, 1] relative to itself to see the pattern (contrast stretch).
                
                print(f"Sample {i} Stats:")
                print(f"  Adv Map: min={adv_map.min():.4f}, max={adv_map.max():.4f}, mean={adv_map.mean():.4f}")
                print(f"  Noise Pred: min={noise_pred_map.min():.4f}, max={noise_pred_map.max():.4f}")
                
                adv_vis = normalize_to_01(adv_map) 
                
                noise_pred_vis = normalize_to_01(noise_pred_map)
                noise_gt_vis = normalize_to_01(noise_level_map)
                
                # Residual Map
                residual_vis = normalize_to_01(torch.abs(low_dose - gen_full_dose))
                
                # Save Images
                prefix = f"sample_{i}"
                save_single_image(low_dose[0,0], os.path.join(vis_dir, f"{prefix}_input.png"))
                save_single_image(gen_full_dose[0,0], os.path.join(vis_dir, f"{prefix}_denoised.png"))
                save_single_image(full_dose[0,0], os.path.join(vis_dir, f"{prefix}_gt.png"))
                
                save_single_image(adv_vis[0,0], os.path.join(vis_dir, f"{prefix}_adv_map_fake.png"))
                save_single_image(noise_pred_vis[0,0], os.path.join(vis_dir, f"{prefix}_noise_pred.png"))
                save_single_image(noise_gt_vis[0,0], os.path.join(vis_dir, f"{prefix}_noise_gt.png"))
                save_single_image(residual_vis[0,0], os.path.join(vis_dir, f"{prefix}_residual.png"))
                
                count += 1
                
    print(f"Visualization complete. Check {vis_dir}")
