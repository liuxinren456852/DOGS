# pylint: disable=[E1101,W0201]

from random import randint

from omegaconf.omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import numpy as np

from conerf.base.model_base import ModelBase
# from conerf.loss.ssim_torch import ssim
from conerf.model.gaussian_fields.scaffold_gs import ScaffoldGS
from conerf.render.scaffold_gs_render import render, prefilter_voxel
from conerf.trainers.gaussian_trainer import GaussianSplatTrainer, ExponentialLR

from fused_ssim import fused_ssim


class ScaffoldGSTrainer(GaussianSplatTrainer):
    """
    Trainer for Scaffold-GS.
    """

    def __init__(
        self,
        config: OmegaConf,
        prefetch_dataset: bool = True,
        trainset=None,
        valset=None,
        model: ModelBase = None,
        block_id: int = None,
        device_id: int = 0
    ) -> None:
        super().__init__(config, prefetch_dataset,
                         trainset, valset, model, block_id, device_id)

    def build_networks(self):
        self.model = ScaffoldGS(
            feat_dim=self.config.geometry.feat_dim,
            num_offsets=self.config.geometry.num_offsets,
            voxel_size=self.config.geometry.voxel_size,
            update_depth=self.config.geometry.update_depth,
            update_init_factor=self.config.geometry.update_init_factor,
            update_hierarchy_factor=self.config.geometry.update_hierarchy_factor,
            use_feat_bank=self.config.geometry.use_feat_bank,
            num_cameras=len(self.train_dataset),
            appearance_dim=self.config.texture.appearance_dim,
            max_sh_degree=self.config.texture.max_sh_degree,
            percent_dense=self.config.geometry.percent_dense,
            device=self.device,
        )

        self.init_gaussians()

    def setup_optimizer(self):
        # Trivial hack when model is passed to the constructor.
        if self.gaussians is None:
            self.gaussians = self.model

        lr_config = self.config.optimizer.lr
        lr_params = [
            {
                'params': [self.gaussians.get_raw_opacity],
                'lr': lr_config.opacity, "name": "opacity",
            },
            {
                'params': [self.gaussians.get_raw_scaling],
                'lr': lr_config.scaling, "name": "scaling",
            },
            {
                'params': [self.gaussians.get_raw_quaternion],
                'lr': lr_config.quaternion, "name": "quaternion",
            },
            # Specified for SAGS.
            {
                'params': [self.gaussians.get_anchor],
                'lr': lr_config.position_init * self.spatial_lr_scale, "name": "anchor",
            },
            {
                'params': [self.gaussians.get_offset],
                'lr': lr_config.offset_init, "name": "offset",
            },
            {
                'params': [self.gaussians.get_anchor_feat],
                'lr': lr_config.anchor_feat, "name": "anchor_feat",
            },
            {
                'params': self.gaussians.mlp_opacity.parameters(),
                'lr': lr_config.mlp_opacity_init, "name": "mlp_opacity",
            },
            {
                'params': self.gaussians.mlp_color.parameters(),
                'lr': lr_config.mlp_color_init, "name": "mlp_color",
            },
            {
                'params': self.gaussians.mlp_cov.parameters(),
                'lr': lr_config.mlp_cov_init, "name": "mlp_cov",
            },
        ]

        if self.gaussians.app_embedding is not None:
            lr_params.append({
                'params': self.gaussians.app_embedding.parameters(),
                'lr': lr_config.app_embedding_init, "name": "app_embedding",
            })

        if self.gaussians.use_feat_bank:
            lr_params.append({
                'params': self.gaussians.mlp_feature_bank.parameters(),
                'lr': lr_config.mlp_feat_bank_init, "name": "mlp_feature_bank"
            })

        self.optimizer = torch.optim.Adam(lr_params, lr=0.0, eps=1e-15)

        self.anchor_scheduler = ExponentialLR(
            lr_init=lr_config.position_init * self.spatial_lr_scale,
            lr_final=lr_config.position_final * self.spatial_lr_scale,
            lr_delay_mult=lr_config.position_delay_mult,
            max_steps=lr_config.position_max_iterations,
        )
        self.offset_scheduler = ExponentialLR(
            lr_init=lr_config.offset_init * self.spatial_lr_scale,
            lr_final=lr_config.offset_final * self.spatial_lr_scale,
            lr_delay_mult=lr_config.offset_delay_mult,
            max_steps=lr_config.offset_max_iterations,
        )
        self.mlp_opacity_scheduler = ExponentialLR(
            lr_init=lr_config.mlp_opacity_init,
            lr_final=lr_config.mlp_opacity_final,
            lr_delay_mult=lr_config.mlp_opacity_delay_mult,
            max_steps=lr_config.mlp_opacity_max_iterations,
        )
        self.mlp_color_scheduler = ExponentialLR(
            lr_init=lr_config.mlp_color_init,
            lr_final=lr_config.mlp_color_final,
            lr_delay_mult=lr_config.mlp_color_delay_mult,
            max_steps=lr_config.mlp_color_max_iterations,
        )
        self.mlp_cov_scheduler = ExponentialLR(
            lr_init=lr_config.mlp_cov_init,
            lr_final=lr_config.mlp_cov_final,
            lr_delay_mult=lr_config.mlp_cov_delay_mult,
            max_steps=lr_config.mlp_cov_max_iterations,
        )

        if self.gaussians.app_embedding is not None:
            self.app_embedding_scheduler = ExponentialLR(
                lr_init=lr_config.app_embedding_init,
                lr_final=lr_config.app_embedding_final,
                lr_delay_mult=lr_config.app_embedding_delay_mult,
                max_steps=lr_config.app_embedding_max_iterations,
            )

        if self.gaussians.use_feat_bank:
            self.mlp_feature_bank_scheduler = ExponentialLR(
                lr_init=lr_config.mlp_feat_bank_init,
                lr_final=lr_config.mlp_feat_bank_final,
                lr_delay_mult=lr_config.mlp_feat_bank_delay_mult,
                max_steps=lr_config.mlp_feat_bank_max_iterations,
            )

        self.exposure_optimizer = None
        self.exposure_scheduler = None
        if self.config.appearance.use_trained_exposure:
            self.exposure_optimizer = torch.optim.Adam([self.gaussians.get_exposure])
            self.exposure_scheduler = ExponentialLR(
                lr_init=lr_config.exposure_lr_init,
                lr_final=lr_config.exposure_lr_final,
                lr_delay_steps=lr_config.exposure_lr_delay_steps,
                lr_delay_mult=lr_config.exposure_lr_delay_mult,
                max_steps=lr_config.exposure_max_iterations,
            )

        self.setup_pose_optimizer()

    def update_learning_rate(self):
        lr_app_embedding = 0
        lr_feat_bank = 0

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr_offset = self.offset_scheduler(self.iteration)
                param_group["lr"] = lr_offset
            if param_group["name"] == "anchor":
                lr_anchor = self.anchor_scheduler(self.iteration)
                param_group["lr"] = lr_anchor
            if param_group["name"] == "mlp_opacity":
                lr_opacity = self.mlp_opacity_scheduler(self.iteration)
                param_group['lr'] = lr_opacity
            if param_group["name"] == "mlp_color":
                lr_color = self.mlp_color_scheduler(self.iteration)
                param_group['lr'] = lr_color
            if param_group["name"] == "mlp_cov":
                lr_cov = self.mlp_cov_scheduler(self.iteration)
                param_group['lr'] = lr_cov
            if self.gaussians.app_embedding is not None and param_group["name"] == "app_embedding":
                lr_app_embedding = self.app_embedding_scheduler(self.iteration)
                param_group['lr'] = lr_app_embedding
            if self.gaussians.use_feat_bank and param_group["name"] == "mlp_feature_bank":
                lr_feat_bank = self.mlp_feature_bank_scheduler(self.iteration)
                param_group['lr'] = lr_feat_bank

        self._update_exposure_params_lr()

        return lr_anchor, lr_offset, lr_opacity, lr_color, lr_cov, lr_app_embedding, lr_feat_bank

    def train_iteration(self, data_batch) -> None:
        self.gaussians.train()
        self.update_learning_rate()

        # Pick a random camera.
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.train_dataset.cameras.copy()
        camera = self.viewpoint_stack.pop(
            randint(0, len(self.viewpoint_stack) - 1))
        resolution = self.training_resolution()
        camera = camera.downsample(resolution).copy_to_device(self.device)
        self.scalars_to_log['train/resolution'] = resolution

        # pre-filter scene.
        visible_mask = prefilter_voxel(
            self.gaussians, camera, self.config.pipeline, self.color_bkgd)
        if visible_mask.sum() == 0:
            return

        render_results = render(
            gaussian_splat_model=self.gaussians,
            viewpoint_camera=camera,
            pipeline_config=self.config.pipeline,
            bkgd_color=self.color_bkgd,
            anti_aliasing=self.config.texture.anti_aliasing,
            visible_mask=visible_mask,
            use_trained_exposure=self.config.appearance.use_trained_exposure,
            depth_threshold=self.config.geometry.depth_threshold,
            device=self.device,
        )
        colors, screen_space_points, visibility_filter = (
            render_results["rendered_image"],
            render_results["screen_space_points"],
            render_results["visibility_filter"],
        )
        offset_selection_mask = render_results["combined_mask"]

        lambda_dssim = self.config.loss.lambda_dssim
        lambda_scale = self.config.loss.lambda_scale
        pixels = camera.image.permute(2, 0, 1)  # [RGB, height, width]

        loss_rgb_l1 = F.l1_loss(colors, pixels)
        # loss_ssim = ssim(pixels, colors)
        loss_ssim = fused_ssim(colors.unsqueeze(0), pixels.unsqueeze(0))
        loss_scaling = render_results["scaling"].prod(dim=1).mean()
        loss = (1.0 - lambda_dssim) * loss_rgb_l1 + \
            lambda_dssim * (1.0 - loss_ssim) + lambda_scale * loss_scaling

        loss.backward()

        self.ema_loss = 0.4 * loss.detach().item() + 0.6 * \
            self.ema_loss  # pylint: disable=W0201

        mse = F.mse_loss(colors, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)

        # training statistics.
        self.scalars_to_log["train/psnr"] = psnr.detach().item()
        self.scalars_to_log["train/loss"] = loss.detach().item()
        self.scalars_to_log["train/l1_loss"] = loss_rgb_l1.detach().item()
        self.scalars_to_log["train/scale_loss"] = loss_scaling.detach().item()
        self.scalars_to_log["train/ema_loss"] = self.ema_loss
        self.scalars_to_log["train/points"] = self.gaussians.get_anchor.shape[0]

        with torch.no_grad():
            if self.iteration < self.config.geometry.densify_end_iter and \
               self.iteration > self.config.geometry.stat_start_iter:
                self.gaussians.add_densification_stats(
                    screen_space_points, visibility_filter, render_results["neural_opacity"],
                    offset_selection_mask, visible_mask,
                )

                if self.iteration > self.config.geometry.densify_start_iter and \
                        self.iteration % self.config.geometry.densification_interval == 0:
                    self.gaussians.densify_and_prune(
                        max_grad=self.config.geometry.densify_grad_threshold,
                        min_opacity=0.005,
                        optimizer=self.optimizer,
                    )
            elif self.iteration == self.config.geometry.densify_end_iter:
                del self.gaussians.opacity_accum
                del self.gaussians.offset_gradient_accum
                del self.gaussians.offset_denom
                torch.cuda.empty_cache()

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.exposure_optimizer is not None:
            self.exposure_optimizer.step()
            self.exposure_optimizer.zero_grad(set_to_none=True)
