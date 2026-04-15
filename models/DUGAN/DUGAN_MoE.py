from .DUGAN import DUGAN
from .noise_discriminator import NoiseDiscriminator
from .gating_network import GatingNetwork
import torch
import torch.nn.functional as F
from dugan_utils.ops import turn_on_spectral_norm

class DUGAN_MoE(DUGAN):
    @staticmethod
    def build_options():
        # Inherit options from DUGAN
        parser = DUGAN.build_options()
        
        # MoE specific options
        parser.add_argument('--moe_phase', type=int, default=0, help='0: Baseline, 1: Train Noise Branch, 2: Train Gate')
        parser.add_argument('--lambda_noise_reg', type=float, default=1.0, help='Weight for noise regression loss')
        parser.add_argument('--lambda_gate_balance', type=float, default=0.01, help='Weight for gate load balancing loss')
        parser.add_argument('--lambda_noise_fixed', type=float, default=0.3, help='Fixed weight for noise branch in Phase 1')
        
        return parser

    def set_model(self):
        super().set_model()
        opt = self.opt
        
        # Initialize Noise Discriminator
        self.noise_discriminator = NoiseDiscriminator(
            in_channels=3, # y_hat, residual, noise_map
            base_channels=64
        )
        self.noise_discriminator = turn_on_spectral_norm(self.noise_discriminator).cuda()
        
        self.noise_d_optimizer = torch.optim.Adam(
            self.noise_discriminator.parameters(),
            lr=opt.d_lr,
            betas=(opt.momentum, 0.999),
            weight_decay=opt.weight_decay
        )
        
        # Initialize Gating Network
        # Input dim: 2 (GAP of y_hat, residual) + bottleneck_dim (from noise discriminator)
        # We need to know bottleneck dim. Based on NoiseDiscriminator implementation, it's 256 (default)
        # Let's assume 256 for now, or get it from model.
        # In NoiseDiscriminator, bottleneck is produced by conv 1x1 to `ndf * 4` = 256.
        self.gating_network = GatingNetwork(
            bottleneck_channels=512, # 64 * 8 = 512
            num_experts=3
        ).cuda()
        
        self.gating_optimizer = torch.optim.Adam(
            self.gating_network.parameters(),
            lr=opt.d_lr,
            betas=(opt.momentum, 0.999),
            weight_decay=opt.weight_decay
        )

        # Register new modules to logger for saving/loading
        # Assign to local variables so get_varname can find the names
        noise_discriminator = self.noise_discriminator
        noise_d_optimizer = self.noise_d_optimizer
        gating_network = self.gating_network
        gating_optimizer = self.gating_optimizer
        
        self.logger.modules = [noise_discriminator, noise_d_optimizer, gating_network, gating_optimizer]

    def train(self, inputs, n_iter):
        opt = self.opt
        self.update_moving_average()
        low_dose, full_dose = inputs
        low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
        noise_level_map = torch.abs(low_dose - full_dose)
        
        self.generator.train()
        self.img_discriminator.train()
        self.grad_discriminator.train()
        self.noise_discriminator.train()
        if hasattr(self, 'gating_network'):
            self.gating_network.train()
        
        # 1. Generator Forward
        gen_full_dose = self.generator(low_dose)
        grad_gen_full_dose = self.sobel(gen_full_dose)
        grad_low_dose = self.sobel(low_dose)
        grad_full_dose = self.sobel(full_dose)
        
        # 2. Train Discriminators (Img & Grad) - Always active
        self.train_discriminator(self.img_discriminator, self.img_d_optimizer, 
                               full_dose, low_dose, gen_full_dose, 'img', n_iter)
        
        if opt.use_grad_discriminator:
            self.train_discriminator(self.grad_discriminator, self.grad_d_optimizer,
                                   grad_full_dose, grad_low_dose, grad_gen_full_dose, 'grad', n_iter)
                                   
        # 3. Train Noise Discriminator (Active in Phase 1 & 2)
        if opt.moe_phase >= 1:
            # Detach generator output for discriminator training
            y_hat_detach = gen_full_dose.detach()

            # D_noise Forward Real
            adv_real, noise_real, _ = self.noise_discriminator(low_dose, full_dose, noise_level_map)

            # D_noise Forward Fake
            adv_fake, _, _ = self.noise_discriminator(low_dose, y_hat_detach, noise_level_map)

            # Loss D_noise
            l_d_noise_adv = self.gan_metric(adv_real, 1.) + self.gan_metric(adv_fake, 0.)
            l_d_noise_reg = F.l1_loss(noise_real, noise_level_map)
            l_d_noise_total = l_d_noise_adv + l_d_noise_reg

            self.noise_d_optimizer.zero_grad()
            l_d_noise_total.backward()
            self.noise_d_optimizer.step()

            self.logger.msg({
                'loss/noise_disc_adv': l_d_noise_adv,
                'loss/noise_disc_reg': l_d_noise_reg,
            }, n_iter)

        # 4. Train Generator
        if n_iter % opt.d_iter == 0:
            self.g_optimizer.zero_grad()
            if opt.moe_phase == 2:
                self.gating_optimizer.zero_grad()

            # Re-forward for generator graph after discriminator updates.
            gen_full_dose = self.generator(low_dose)
            grad_gen_full_dose = self.sobel(gen_full_dose)

            # Baseline DU-GAN objective: keep this intact for fair comparison.
            fake_enc, fake_dec = self.img_discriminator(gen_full_dose)
            l_g_img = self.gan_metric(fake_enc, 1.) + self.gan_metric(fake_dec, 1.)

            l_g_grad = 0
            if opt.use_grad_discriminator:
                grad_gen_enc, grad_gen_dec = self.grad_discriminator(grad_gen_full_dose)
                l_g_grad = self.gan_metric(grad_gen_enc, 1.) + self.gan_metric(grad_gen_dec, 1.)

            pix_loss = F.mse_loss(gen_full_dose, full_dose)
            l1_loss = F.l1_loss(gen_full_dose, full_dose)
            grad_loss = F.l1_loss(grad_gen_full_dose, grad_full_dose)

            # D_noise feedback
            l_g_noise = 0
            l_noise_reg = 0
            l_gate_balance = 0
            bottleneck_g = None
            if opt.moe_phase >= 1:
                adv_fake_g, noise_fake_g, bottleneck_g = self.noise_discriminator(low_dose, gen_full_dose, noise_level_map)
                l_g_noise = self.gan_metric(adv_fake_g, 1.)
                l_noise_reg = F.l1_loss(noise_fake_g, noise_level_map)

            noise_adv_weight = 0.0
            if opt.moe_phase == 1:
                noise_adv_weight = opt.lambda_noise_fixed
            elif opt.moe_phase == 2:
                residual_abs = torch.abs(low_dose - gen_full_dose)
                alphas = self.gating_network(gen_full_dose, residual_abs, bottleneck_g)
                avg_alphas = alphas.mean(dim=0)
                noise_adv_weight = avg_alphas[2]
                l_gate_balance = self.gating_network.balance_loss(alphas)
                if n_iter % opt.save_freq == 0:
                    print(f"Iter {n_iter} Gate weights: img={avg_alphas[0]:.3f}, grad={avg_alphas[1]:.3f}, noise={avg_alphas[2]:.3f}")

            baseline_total = (
                l_g_img * opt.img_gen_loss_weight +
                pix_loss * opt.pix_loss_weight +
                grad_loss * opt.grad_loss_weight
            )
            if opt.use_grad_discriminator:
                baseline_total += l_g_grad * opt.grad_gen_loss_weight

            noise_total = (
                opt.img_gen_loss_weight * noise_adv_weight * l_g_noise +
                opt.lambda_noise_reg * l_noise_reg +
                opt.lambda_gate_balance * l_gate_balance
            )
            l_total = baseline_total + noise_total

            l_total.backward()
            self.g_optimizer.step()
            if opt.moe_phase == 2:
                self.gating_optimizer.step()

            self.logger.msg({
                'loss/img_gen_loss': l_g_img,
                'loss/grad_gen_loss': l_g_grad,
                'loss/noise_gen_adv': l_g_noise,
                'loss/noise_reg': l_noise_reg,
                'loss/gate_balance': l_gate_balance,
                'loss/pix': pix_loss,
                'loss/l1': l1_loss,
                'loss/grad': grad_loss,
                'loss/total': l_total,
            }, n_iter)

    @torch.no_grad()
    def generate_images(self, n_iter):
        self.generator.eval()
        low_dose, full_dose = self.test_images
        gen_full_dose = self.generator(low_dose).clamp(0., 1.)
        
        # Get maps
        # D_img returns (score, map). We want the map for visualization.
        _, map_img = self.img_discriminator(gen_full_dose)
        
        # D_noise
        map_noise_adv = torch.zeros_like(map_img)
        map_noise_reg = torch.zeros_like(map_img)
        if hasattr(self, 'noise_discriminator'):
            noise_level_map = torch.abs(low_dose - full_dose) # Proxy
            adv, reg, _ = self.noise_discriminator(low_dose, gen_full_dose, noise_level_map)
            map_noise_adv = adv
            map_noise_reg = reg
             
        # Stack for visualization
        # [Low, Full, Gen, Map_Img, Map_Noise_Adv, Map_Noise_Reg]
        # Normalize maps to [0,1] for vis
        def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-6)
        
        imgs = [low_dose, full_dose, gen_full_dose, norm(map_img), norm(map_noise_adv), norm(map_noise_reg)]
        
        # Reshape and save
        _, ch, w, h = low_dose.size()
        out = torch.stack(imgs).transpose(1, 0).reshape((-1, ch, w, h)) # [N*6, C, H, W]
        
        from torchvision.utils import make_grid
        grid = make_grid(out, nrow=6)
        self.logger.save_image(grid, n_iter, 'test_moe')
