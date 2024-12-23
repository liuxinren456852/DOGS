# pylint: disable=[E1101,R1719,W0201]

import os
import random
import copy
from typing import List
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import numpy as np

from conerf.base.model_base import ModelBase
from conerf.base.task_queue import ImageReader
from conerf.datasets.dataset_base import MiniDataset
from conerf.datasets.utils import (
    fetch_ply, compute_nerf_plus_plus_norm, create_dataset, get_block_info_dir
)
# from conerf.loss.ssim_torch import ssim
from conerf.model.gaussian_fields.gaussian_splat_model import GaussianSplatModel
from conerf.model.gaussian_fields.masks import AppearanceEmbedding
from conerf.render.gaussian_render import render
from conerf.model.gaussian_fields.prune import calculate_v_imp_score, prune_list
from conerf.trainers.implicit_recon_trainer import ImplicitReconTrainer
from conerf.trainers.gaussian_trainer import GaussianSplatTrainer
from conerf.visualization.pose_visualizer import visualize_cameras

from diff_gaussian_rasterization import SparseGaussianAdam
from fused_ssim import fused_ssim


# NOTE: We define a callable class instead of a closure since a closure cannot
# be serialized by torch's rpc.
class ExponentialLR:  # pylint: disable=[R0903]
    def __init__(
        self,
        lr_init: float,
        lr_final: float,
        lr_delay_steps: int = 0,
        lr_delay_mult: float = 1.0,
        max_steps: int = 1000000
    ):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        self.max_steps = max_steps

    def __call__(self, step):
        if step < 0 or (self.lr_init == 0.0 and self.lr_final == 0.0):
            return 0.0

        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) +
                          np.log(self.lr_final) * t)

        return delay_rate * log_lerp


def extract_pose_tensor_from_cameras(
    cameras: List,
    noises: torch.Tensor = None,
    corrections: torch.Tensor = None,
) -> torch.Tensor:
    num_cameras = len(cameras)
    poses = torch.zeros((num_cameras, 4, 4), dtype=torch.float32).cuda()

    for i in range(num_cameras):
        poses[i, :3, :3] = cameras[i].R
        poses[i, :3, 3] = cameras[i].t
        poses[i, 3, 3] = 1.0

        # Add noise.
        if noises is not None:
            poses[i] = poses[i] @ noises[i]

        # Correct noise.
        if corrections is not None:
            poses[i] = poses[i] @ corrections[i]

    return poses


def load_val_dataset(config: OmegaConf, device: str = 'cuda'):
    val_config = copy.deepcopy(config)
    val_config.dataset.multi_blocks = False
    val_config.dataset.num_blocks = 1
    val_dataset = create_dataset(
        config=val_config,
        split=val_config.dataset.val_split,
        num_rays=None,
        apply_mask=val_config.dataset.apply_mask,
        device=device
    )
    return val_dataset


class SlaveGaussianSplatTrainer(GaussianSplatTrainer):
    """
    Trainer for 3D Gaussian Splatting model.
    """

    def __init__(
        self,
        config: OmegaConf,
        prefetch_dataset: bool = True,
        trainset=None,
        valset=None,
        model: ModelBase = None,
        block_id: int = None,
        device_id: int = 0,
    ) -> None:
        # self.gaussians = None

        if trainset is None and block_id is not None:
            mx = config.dataset.get("mx", None)  # pylint: disable=C0103
            my = config.dataset.get("my", None)  # pylint: disable=C0103
            data_dir = os.path.join(
                config.dataset.root_dir, config.dataset.scene)
            data_dir = get_block_info_dir(
                data_dir, config.dataset.num_blocks, mx, my)
            block_dir = os.path.join(data_dir, f'block_{block_id}')
            trainset = MiniDataset(
                cameras=[], camtoworlds=None, block_id=block_id)
            trainset.read(path=block_dir, block_id=block_id, device='cpu')

        # if valset is None:
        #     valset = load_val_dataset(config, 'cpu')

        # self.admm_enabled = False

        super().__init__(config, prefetch_dataset,
                         trainset, valset, model, block_id, device_id)

    def update_iteration(self, iteration: int):
        self.iteration = iteration

    @torch.no_grad()
    def send_local_model(self):
        xyz, features_dc, features_rest, scaling, quaternion, opacity = \
            self.gaussians.get_all_properties()

        return xyz, features_dc, features_rest, scaling, quaternion, opacity

    @torch.no_grad()
    def send_local_loss(self):
        return self.scalars_to_log

    def setup_dual_variables(self):
        # Set dual variables for all the properties of 3D Gaussians:
        # xyz, features_dc, features_rest, scaling, quaternion, opacity.
        # We follow DBACC (Distributed Bundle Adjustment based on Camera Consensus) to
        # initialize all dual variables to zeros.
        self.u_xyz = torch.zeros_like(
            self.gaussians.get_xyz, requires_grad=False)
        self.u_fdc = torch.zeros_like(
            self.gaussians.get_features_dc, requires_grad=False)
        self.u_fr = torch.zeros_like(
            self.gaussians.get_features_rest, requires_grad=False)
        self.u_s = torch.zeros_like(
            self.gaussians.get_raw_scaling, requires_grad=False)
        self.u_q = torch.zeros_like(
            self.gaussians.get_raw_quaternion, requires_grad=False)
        self.u_o = torch.zeros_like(
            self.gaussians.get_raw_opacity, requires_grad=False)

    @torch.no_grad()
    def update_dual_variables(self):
        # Apply over-relaxation by:
        #   u^{k+1} = u^k + (1 + alpha^k) * (x^{k+1} - z^{k+1}).
        over_relaxation_factor = 1 + self.config.trainer.admm.over_relaxation_coeff
        self.u_xyz += over_relaxation_factor * (
            self.gaussians.get_xyz - self.global_xyz
        )
        self.u_fdc += over_relaxation_factor * (
            self.gaussians.get_features_dc - self.global_features_dc
        )
        self.u_fr += over_relaxation_factor * (
            self.gaussians.get_features_rest - self.global_features_rest
        )
        self.u_s += over_relaxation_factor * (
            self.gaussians.get_raw_scaling - self.global_scaling
        )
        self.u_q += over_relaxation_factor * (
            self.gaussians.get_raw_quaternion - self.global_quaternion
        )
        self.u_o += over_relaxation_factor * (
            self.gaussians.get_raw_opacity - self.global_opacity
        )

    def set_penalty_parameters(
        self,
        rho_xyz: float,
        rho_fdc: float,
        rho_fr: float,
        rho_s: float,
        rho_q: float,
        rho_o: float,
    ):
        self.rho_xyz = rho_xyz
        self.rho_fdc = rho_fdc
        self.rho_fr = rho_fr
        self.rho_s = rho_s
        self.rho_q = rho_q
        self.rho_o = rho_o

    def set_global_indices(self, global_indices: torch.Tensor):
        self.global_indices = global_indices

    def set_global_gaussians(
        self,
        global_xyz,
        global_features_dc,
        global_features_rest,
        global_scaling,
        global_quaternion,
        global_opacity
    ):
        self.global_xyz = global_xyz
        self.global_features_dc = global_features_dc
        self.global_features_rest = global_features_rest
        self.global_scaling = global_scaling
        self.global_quaternion = global_quaternion
        self.global_opacity = global_opacity

    def enable_admm_training(self):
        self.admm_enabled = True

    def add_admm_penalties(self, loss):
        xyz = self.gaussians.get_xyz
        features_dc = self.gaussians.get_features_dc
        features_rest = self.gaussians.get_features_rest
        scaling = self.gaussians.get_raw_scaling
        quaternion = self.gaussians.get_raw_quaternion
        opacity = self.gaussians.get_raw_opacity

        xyz_penalty = 0.5 * self.rho_xyz * F.mse_loss(
            xyz + self.u_xyz, self.global_xyz
        )
        feat_dc_penalty = 0.5 * self.rho_fdc * F.mse_loss(
            features_dc + self.u_fdc, self.global_features_dc
        )
        feat_rest_penalty = 0.5 * self.rho_fr * F.mse_loss(
            features_rest + self.u_fr, self.global_features_rest
        )
        scaling_penalty = 0.5 * self.rho_s * F.mse_loss(
            scaling + self.u_s, self.global_scaling
        )
        quaternion_penalty = 0.5 * self.rho_q * F.mse_loss(
            quaternion + self.u_q, self.global_quaternion
        )
        opacity_penalty = 0.5 * self.rho_o * F.mse_loss(
            opacity + self.u_o, self.global_opacity
        )

        loss += xyz_penalty
        loss += feat_dc_penalty
        loss += feat_rest_penalty
        loss += scaling_penalty
        loss += quaternion_penalty
        loss += opacity_penalty

        self.scalars_to_log["train/xyz_penalty"] = xyz_penalty.detach().item()
        self.scalars_to_log["train/feat_dc_penalty"] = feat_dc_penalty.detach().item()
        self.scalars_to_log["train/feat_rest_penalty"] = feat_rest_penalty.detach().item()
        self.scalars_to_log["train/scaling_penalty"] = scaling_penalty.detach().item()
        self.scalars_to_log["train/quaternion_penalty"] = quaternion_penalty.detach().item()
        self.scalars_to_log["train/opacity_penalty"] = opacity_penalty.detach().item()

        return loss

    def train_every_x_interval(self, data_batch, interval: int = 100):
        for i in range(interval):  # pylint: disable=W0612
            self.increment_iteration()
            self.train_iteration(data_batch=data_batch)
