import os
import torch
from models import model_dict, TrainTask

if __name__ == '__main__':
    # Parse arguments
    default_parser = TrainTask.build_default_options()
    default_parser.add_argument('--phase', type=str, default='test', help='train, test')
    default_opt, unknown_opt = default_parser.parse_known_args()
    
    # Load Model
    MODEL = model_dict[default_opt.model_name]
    private_parser = MODEL.build_options()
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)
    
    # Force test mode settings
    opt.isTrain = False
    
    # Initialize Model
    model = MODEL(opt)
    
    # Load Checkpoint
    if opt.resume_iter > 0:
        model.logger.load_checkpoints(opt.resume_iter)
    else:
        print("Warning: No resume_iter specified. Using random weights?")

    # Run Test
    print(f"Running test for model {opt.model_name} at iter {opt.resume_iter}...")
    
    if not hasattr(model, 'test_loader'):
        print("Error: Model does not have test_loader.")
        exit(1)
        
    # Evaluation Loop
    total_psnr = 0
    total_ssim = 0
    total_rmse = 0
    count = 0
    
    from dugan_utils.metrics import compute_psnr, compute_ssim, compute_rmse
    import tqdm
    
    model.generator.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(model.test_loader)):
            # Unpack batch
            # DUGAN expects (low, full)
            if isinstance(batch, (list, tuple)):
                low_dose, full_dose = batch
            else:
                low_dose = batch
                full_dose = batch # Placeholder if no GT
            
            low_dose = low_dose.cuda()
            full_dose = full_dose.cuda()
            
            # Forward
            gen_full_dose = model.generator(low_dose).clamp(0., 1.)
            
            # Calculate Metrics
            p = compute_psnr(gen_full_dose, full_dose)
            s = compute_ssim(gen_full_dose, full_dose)
            r = compute_rmse(gen_full_dose, full_dose)
            
            total_psnr += p.item()
            total_ssim += s.item()
            total_rmse += r.item()
            count += 1
            
    print(f"Test Results for {opt.model_name} (Iter {opt.resume_iter}):")
    print(f"PSNR: {total_psnr/count:.4f}")
    print(f"SSIM: {total_ssim/count:.4f}")
    print(f"RMSE: {total_rmse/count:.4f}")
