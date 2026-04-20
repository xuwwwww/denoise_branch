import argparse
import os

import numpy as np
import torch
from PIL import Image

from models import TrainTask, model_dict
from dugan_utils.metrics import compute_psnr, compute_rmse, compute_ssim


def normalize_to_01(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-6)


def save_single_image(tensor: torch.Tensor, path: str) -> None:
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def build_model(model_name: str, run_name: str, resume_iter: int, moe_phase: int = 0):
    default_parser = TrainTask.build_default_options()
    default_opt = default_parser.parse_args([])
    default_opt.model_name = model_name
    default_opt.run_name = run_name
    default_opt.resume_iter = resume_iter
    default_opt.test_dataset_name = 'cmayo_test_512'
    default_opt.batch_size = 1
    default_opt.test_batch_size = 1

    model_cls = model_dict[model_name]
    private_parser = model_cls.build_options()
    opt = private_parser.parse_args([], namespace=default_opt)
    opt.isTrain = False
    if hasattr(opt, 'moe_phase'):
        opt.moe_phase = moe_phase

    model = model_cls(opt)
    if resume_iter > 0:
        model.logger.load_checkpoints(resume_iter)

    model.generator.eval()
    if hasattr(model, 'noise_discriminator'):
        model.noise_discriminator.eval()
    return model


def tensor_metrics(pred: torch.Tensor, target: torch.Tensor):
    return {
        'ssim': float(compute_ssim(pred, target).item()),
        'psnr': float(compute_psnr(pred, target).item()),
        'rmse': float(compute_rmse(pred, target).item()),
    }


def save_text(path: str, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser(description='Compare LDCT denoising checkpoints.')
    parser.add_argument('--baseline-model', default='DUGAN')
    parser.add_argument('--baseline-run', required=True)
    parser.add_argument('--baseline-iter', type=int, required=True)
    parser.add_argument('--candidate-model', default='DUGAN_MoE')
    parser.add_argument('--candidate-run', required=True)
    parser.add_argument('--candidate-iter', type=int, required=True)
    parser.add_argument('--candidate-moe-phase', type=int, default=1)
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--output-dir', default='output/comparison_ldct')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline = build_model(args.baseline_model, args.baseline_run, args.baseline_iter, moe_phase=0)
    candidate = build_model(
        args.candidate_model,
        args.candidate_run,
        args.candidate_iter,
        moe_phase=args.candidate_moe_phase,
    )

    summary_lines = [
        f'baseline={args.baseline_model}:{args.baseline_run}:{args.baseline_iter}',
        f'candidate={args.candidate_model}:{args.candidate_run}:{args.candidate_iter}:phase{args.candidate_moe_phase}',
        '',
    ]

    baseline_iter = iter(baseline.test_loader)
    candidate_iter = iter(candidate.test_loader)

    with torch.no_grad():
        for idx in range(args.num_samples):
            low_b, full_b = next(baseline_iter)
            low_c, full_c = next(candidate_iter)

            low_b = low_b.cuda()
            full_b = full_b.cuda()
            low_c = low_c.cuda()
            full_c = full_c.cuda()

            base_pred = baseline.generator(low_b).clamp(0.0, 1.0)
            cand_pred = candidate.generator(low_c).clamp(0.0, 1.0)

            base_metrics = tensor_metrics(base_pred, full_b)
            cand_metrics = tensor_metrics(cand_pred, full_c)

            base_residual = torch.abs(low_b - base_pred)
            cand_residual = torch.abs(low_c - cand_pred)
            gt_residual = torch.abs(low_b - full_b)

            prefix = os.path.join(args.output_dir, f'sample_{idx}')
            save_single_image(low_b[0, 0], f'{prefix}_input.png')
            save_single_image(full_b[0, 0], f'{prefix}_gt.png')
            save_single_image(base_pred[0, 0], f'{prefix}_baseline.png')
            save_single_image(cand_pred[0, 0], f'{prefix}_candidate.png')
            save_single_image(normalize_to_01(gt_residual[0, 0]), f'{prefix}_gt_residual.png')
            save_single_image(normalize_to_01(base_residual[0, 0]), f'{prefix}_baseline_residual.png')
            save_single_image(normalize_to_01(cand_residual[0, 0]), f'{prefix}_candidate_residual.png')

            if hasattr(candidate, 'noise_discriminator'):
                noise_map = torch.abs(low_c - full_c)
                adv_map, noise_pred, _ = candidate.noise_discriminator(low_c, cand_pred, noise_map)
                save_single_image(normalize_to_01(adv_map[0, 0]), f'{prefix}_candidate_adv_map.png')
                save_single_image(normalize_to_01(noise_pred[0, 0]), f'{prefix}_candidate_noise_pred.png')
                save_single_image(normalize_to_01(noise_map[0, 0]), f'{prefix}_candidate_noise_gt.png')

            summary_lines.append(
                f'sample_{idx}: '
                f'baseline(ssim={base_metrics["ssim"]:.5f}, psnr={base_metrics["psnr"]:.5f}, rmse={base_metrics["rmse"]:.5f}) | '
                f'candidate(ssim={cand_metrics["ssim"]:.5f}, psnr={cand_metrics["psnr"]:.5f}, rmse={cand_metrics["rmse"]:.5f})'
            )

    save_text(os.path.join(args.output_dir, 'summary.txt'), '\n'.join(summary_lines) + '\n')
    print(f'Comparison saved to {args.output_dir}')


if __name__ == '__main__':
    main()
