# pylint: disable=[E1101,E1102,W0201]

from functools import reduce
from typing import List

import einops
import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_max

from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

from conerf.datasets.utils import BasicPointCloud
from conerf.geometry.camera import Camera
from conerf.model.gaussian_fields.gaussian_splat_model import (
    GaussianSplatModel, inverse_sigmoid, cat_tensors_to_optimizer,
    detach_tensor_to_numpy,
)


def prune_optimizer(mask, optimizer):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if 'mlp' in group['name'] or 'conv' in group['name'] or \
           'feat_base' in group['name'] or 'embedding' in group['name']:
            continue

        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

            del optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(
                (group["params"][0][mask].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state

            if group['name'] == 'scaling':
                scales = group['params'][0]
                temp = scales[:, 3:]
                temp[temp > 0.05] = 0.05
                group['params'][0][:, 3:] = temp

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(
                group["params"][0][mask].requires_grad_(True))

            if group['name'] == 'scaling':
                scales = group['params'][0]
                temp = scales[:, 3:]
                temp[temp > 0.05] = 0.05
                group['params'][0][:, 3:] = temp

            optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors


class ScaffoldGS(GaussianSplatModel):
    def __init__(
        self,
        feat_dim: int = 32,
        num_offsets: int = 5,
        voxel_size: float = 0.01,
        # For voxel-based densification.
        update_depth: int = 3,
        update_init_factor: int = 100,
        update_hierarchy_factor: int = 4,
        use_feat_bank: bool = False,
        # Appearance embedding
        num_cameras: int = 0,
        appearance_dim: int = 32,
        max_sh_degree: int = 3,
        percent_dense: float = 0.01,
        device: str = "cuda"
    ) -> None:
        super().__init__(max_sh_degree, percent_dense, device)

        self.feat_dim = feat_dim
        self.num_offsets = num_offsets
        self.voxel_size = voxel_size

        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierarchy_factor = update_hierarchy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.app_embedding = None
        if appearance_dim:
            self.app_embedding = torch.nn.Embedding(
                num_cameras, self.appearance_dim
            ).to(self.device)

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).to(self.device)

        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, num_offsets),
            nn.Tanh(),
        ).to(self.device)

        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim + 3, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7 * self.num_offsets),
        ).to(self.device)

        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * num_offsets),
            nn.Sigmoid(),
        ).to(self.device)

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def get_anchor_feat(self):
        return self._anchor_feat

    @property
    def get_offset(self):
        return self._offset

    def voxelize_sample(self, data=None, voxel_size: float = 0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size
        return data

    def init_from_colmap_pcd(self, pcd: BasicPointCloud, image_idxs: List = None):
        points = self.voxelize_sample(pcd.points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(
            np.asarray(points)).float().to(self.device)
        offsets = torch.zeros(
            fused_point_cloud.shape[0], self.num_offsets, 3).float().to(self.device)
        anchors_feat = torch.zeros(
            fused_point_cloud.shape[0], self.feat_dim).float().to(self.device)
        print(
            f'[SCAFFOLD-GS] number of points after initialization: {fused_point_cloud.shape[0]}')

        num_points3d = fused_point_cloud.shape[0]
        opacities = inverse_sigmoid(
            0.1 * torch.ones((num_points3d, 1),
                             dtype=torch.float, device=self.device)
        )
        dist2 = torch.clamp_min(
            distCUDA2(fused_point_cloud).float().to(self.device), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)
        quats = torch.zeros(
            (fused_point_cloud.shape[0], 4), device=self.device)
        quats[:, 0] = 1.0

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._quaternion = nn.Parameter(quats.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        if image_idxs is not None:
            self.image_id_to_index = {idx: ind for ind, idx in enumerate(image_idxs)}
            exposure = torch.eye(3, 4, device=self.device)[None].repeat(
                len(image_idxs), 1, 1)
            self._exposure = nn.Parameter(exposure.requires_grad_(True))

        num_anchors = self.get_anchor.shape[0]
        self.opacity_accum = torch.zeros((num_anchors, 1), device=self.device)
        self.offset_gradient_accum = torch.zeros(
            (num_anchors * self.num_offsets, 1), device=self.device)
        self.offset_denom = torch.zeros(
            (num_anchors * self.num_offsets, 1), device=self.device)
        self.anchor_denom = torch.zeros((num_anchors, 1), device=self.device)

    @torch.no_grad()
    def extend_from_colmap_pcd(self, pcd: BasicPointCloud, optimizer):
        # TODO(chenyu): consider existed 3D Gaussians.
        points = self.voxelize_sample(pcd.points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(
            np.asarray(points)).float().to(self.device)
        offsets = torch.zeros(
            fused_point_cloud.shape[0], self.num_offsets, 3).float().to(self.device)
        anchors_feat = torch.zeros(
            fused_point_cloud.shape[0], self.feat_dim).float().to(self.device)
        print(
            f'[SCAFFOLD-GS] Extended scene with {fused_point_cloud.shape[0]} points')

        num_points3d = fused_point_cloud.shape[0]
        opacities = inverse_sigmoid(
            0.1 * torch.ones((num_points3d, 1),
                             dtype=torch.float, device=self.device)
        )
        dist2 = torch.clamp_min(
            distCUDA2(fused_point_cloud).float().to(self.device), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)
        quats = torch.zeros(
            (fused_point_cloud.shape[0], 4), device=self.device)
        quats[:, 0] = 1.0

        new_anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_offset = nn.Parameter(offsets.requires_grad_(True))
        new_anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_quaternion = nn.Parameter(quats.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        tensors_dict = {
            "anchor": new_anchor,
            "scaling": new_scaling,
            "quaternion": new_quaternion,
            "anchor_feat": new_anchor_feat,
            "offset": new_offset,
            "opacity": new_opacity,
        }

        temp_anchor_denom = torch.cat([
            self.anchor_denom,
            torch.zeros([new_opacity.shape[0], 1], device=self.device).float()
        ], dim=0)
        del self.anchor_denom
        self.anchor_denom = temp_anchor_denom

        temp_opacity_accum = torch.cat([
            self.opacity_accum, torch.zeros(
                [new_opacity.shape[0], 1], device=self.device)
        ], dim=0)
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        torch.cuda.empty_cache()

        optimizable_tensors = cat_tensors_to_optimizer(tensors_dict, optimizer)
        self._anchor = optimizable_tensors["anchor"]
        self._scaling = optimizable_tensors["scaling"]
        self._quaternion = optimizable_tensors["quaternion"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._offset = optimizable_tensors["offset"]
        self._opacity = optimizable_tensors["opacity"]

        padding_offset_denom = torch.zeros(
            [self.get_anchor.shape[0] * self.num_offsets -
                self.offset_denom.shape[0], 1],
            dtype=torch.int32, device=self.device
        )
        self.offset_denom = torch.cat(
            [self.offset_denom, padding_offset_denom], dim=0)

        padding_offset_gradient_denom = torch.zeros(
            [self.get_anchor.shape[0] * self.num_offsets -
                self.offset_gradient_accum.shape[0], 1],
            dtype=torch.int32, device=self.device)
        self.offset_gradient_accum = torch.cat(
            [self.offset_gradient_accum, padding_offset_gradient_denom], dim=0)

    def generate_neural_gaussians(
        self,
        viewpoint_camera: Camera,
        visible_mask: torch.Tensor = None,
    ):
        num_anchors = self.get_anchor.shape[0]
        if visible_mask is None:
            visible_mask = torch.ones(
                num_anchors,
                dtype=torch.bool,
                device=self.get_xyz.device,
            )

        feats = self.get_anchor_feat[visible_mask]
        anchors = self.get_anchor[visible_mask]
        grid_offsets = self.get_offset[visible_mask]
        grid_scaling = self.get_scaling[visible_mask]

        ob_view = anchors - viewpoint_camera.camera_center
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / ob_dist

        # View-adaptive feature.
        if self.use_feat_bank:
            cat_view = torch.cat([ob_view, ob_dist], dim=1)
            bank_weight = self.mlp_feature_bank(
                cat_view).unsqueeze(dim=1)  # [n,1,3]

            # Multi-resolution feat.
            feats = feats.unsqueeze(dim=-1)
            feats = feats[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1] + \
                feats[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2] + \
                feats[:, ::1, :1] * bank_weight[:, :, 2:]
            feats = feats.squeeze(dim=-1)  # [n, c]

        cat_feats_view = torch.cat([feats, ob_view], dim=1)  # [N,c+3]
        # cat_feat_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N,c+3+1]
        if self.appearance_dim:
            camera_indices = torch.ones_like(
                cat_feats_view[:, 0], dtype=torch.long, device=ob_dist.device
            ) * viewpoint_camera.image_index
            appearance = self.app_embedding(camera_indices)

        neural_opacity = self.mlp_opacity(cat_feats_view)  # [N,k]
        neural_opacity = neural_opacity.reshape(-1, 1)
        mask = (neural_opacity > 0).view(-1)
        opacity = neural_opacity[mask]

        if self.appearance_dim > 0:
            color = self.mlp_color(
                torch.cat([cat_feats_view, appearance], dim=1))
        else:
            color = self.mlp_color(cat_feats_view)
        color = color.reshape(-1, 3)

        scale_rot = self.mlp_cov(cat_feats_view).reshape(-1, 7)
        offsets = grid_offsets.view(-1, 3)

        # combine for parallel masking.
        concatenated = torch.cat([grid_scaling, anchors], dim=-1)
        concatenated_repeated = einops.repeat(
            concatenated, 'n (c) -> (n k) (c)', k=self.num_offsets
        )
        concatenated_all = torch.cat(
            [concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, anchor_repeat, color, scale_rot, offsets = masked.split(
            [6, 3, 3, 7, 3], dim=-1)

        # Post-process cov.
        scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
        rotation = self.quaternion_activation(scale_rot[:, 3:7])

        # Post-process offsets to get centers for gaussians.
        offsets = offsets * scaling_repeat[:, :3]
        xyz = anchor_repeat + offsets

        return xyz, opacity, color, scaling, rotation, neural_opacity, mask

    def densification_postfix(
        self, new_anchor, new_scaling, new_quaternion,
        new_feat, new_offset, new_opacity, offset_mask, optimizer
    ):
        tensors_dict = {
            "anchor": new_anchor,
            "scaling": new_scaling,
            "quaternion": new_quaternion,
            "anchor_feat": new_feat,
            "offset": new_offset,
            "opacity": new_opacity,
        }

        temp_anchor_denom = torch.cat([
            self.anchor_denom,
            torch.zeros([new_opacity.shape[0], 1],
                        device=self.device).float()
        ], dim=0)
        del self.anchor_denom
        self.anchor_denom = temp_anchor_denom

        temp_opacity_accum = torch.cat([
            self.opacity_accum, torch.zeros(
                [new_opacity.shape[0], 1], device=self.device)
        ], dim=0)
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        torch.cuda.empty_cache()

        optimizable_tensors = cat_tensors_to_optimizer(tensors_dict, optimizer)
        self._anchor = optimizable_tensors["anchor"]
        self._scaling = optimizable_tensors["scaling"]
        self._quaternion = optimizable_tensors["quaternion"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._offset = optimizable_tensors["offset"]
        self._opacity = optimizable_tensors["opacity"]

        self.offset_denom[offset_mask] = 0
        padding_offset_denom = torch.zeros(
            [self.get_anchor.shape[0] * self.num_offsets -
             self.offset_denom.shape[0], 1],
            dtype=torch.int32, device=self.device
        )
        self.offset_denom = torch.cat(
            [self.offset_denom, padding_offset_denom], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros(
            [self.get_anchor.shape[0] * self.num_offsets -
             self.offset_gradient_accum.shape[0], 1],
            dtype=torch.int32, device=self.device
        )
        self.offset_gradient_accum = torch.cat(
            [self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

    def add_densification_stats(
        self,
        screen_space_points,
        update_filter,
        opacity,
        offset_selection_mask,
        anchor_visible_mask,
    ):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0
        temp_opacity = temp_opacity.view([-1, self.num_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(
            dim=1, keepdim=True)

        self.anchor_denom[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat(
            [1, self.num_offsets]).view(-1)
        combined_mask = torch.zeros_like(
            self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(
            screen_space_points.grad[update_filter, :2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def anchor_growing(self, grads, threshold, offset_mask, optimizer):
        init_length = self.get_anchor.shape[0] * self.num_offsets
        added_anchor, added_scaling, added_quaternion = [], [], []
        added_feat, added_offset, added_opacity = [], [], []
        for i in range(self.update_depth):
            cur_threshold = threshold * \
                ((self.update_hierarchy_factor // 2) ** i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = (torch.rand_like(candidate_mask.float())
                         > (0.5 ** (i + 1))).to(self.device)
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0] * \
                self.num_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat(
                    [candidate_mask, torch.zeros(
                        length_inc, dtype=torch.bool, device=self.device)],
                    dim=0
                )

            all_xyz = self.get_anchor.unsqueeze(
                dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            size_factor = self.update_init_factor // (
                self.update_hierarchy_factor ** i)
            cur_size = self.voxel_size * size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(
                selected_grid_coords, return_inverse=True, dim=0)

            # Split data for reducing peak memory calling.
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (
                1 if (grid_coords.shape[0] % chunk_size) else 0
            )
            remove_duplicates_list = []
            for _ in range(max_iters):
                cur_remove_duplicates = (
                    selected_grid_coords_unique.unsqueeze(
                        1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]
                ).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = ~(  # pylint: disable=E1130
                reduce(torch.logical_or, remove_duplicates_list))
            candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size

            if candidate_anchor.shape[0] == 0:
                return

            new_scaling = torch.ones_like(candidate_anchor
                                          ).repeat([1, 2]).float().to(self.device) * cur_size
            new_scaling = torch.log(new_scaling)
            new_rotation = torch.zeros(
                [candidate_anchor.shape[0], 4], device=self.device).float()
            new_rotation[:, 0] = 1.0

            new_opacities = inverse_sigmoid(
                0.1 * torch.ones((candidate_anchor.shape[0], 1),
                                 dtype=torch.float, device=self.device)
            )
            new_feat = self._anchor_feat.unsqueeze(dim=1).repeat(
                [1, self.num_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
            new_feat = scatter_max(
                new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0
            )[0][remove_duplicates]
            new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat(
                [1, self.num_offsets, 1]).float().to(self.device)

            added_anchor.append(candidate_anchor)
            added_scaling.append(new_scaling)
            added_quaternion.append(new_rotation)
            added_feat.append(new_feat)
            added_offset.append(new_offsets)
            added_opacity.append(new_opacities)

        added_anchor = torch.concat(added_anchor, dim=0)
        added_scaling = torch.concat(added_scaling, dim=0)
        added_quaternion = torch.concat(added_quaternion, dim=0)
        added_feat = torch.concat(added_feat, dim=0)
        added_offset = torch.concat(added_offset, dim=0)
        added_opacity = torch.concat(added_opacity, dim=0)
        self.densification_postfix(
            added_anchor, added_scaling, added_quaternion,
            added_feat, added_offset, added_opacity, offset_mask, optimizer
        )

    def prune_anchors(self,
                      min_opacity,
                      optimizer,
                      check_interval: int = 100,
                      success_threshold: float = 0.8
                      ):
        prune_mask = (self.opacity_accum < min_opacity *
                      self.anchor_denom).squeeze(dim=1)
        anchors_mask = (self.anchor_denom > check_interval *
                        success_threshold).squeeze(dim=1)  # [N,1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset denom.
        offset_denom = self.offset_denom.view(
            [-1, self.num_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view(
            [-1, self.num_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # Update opacity accum.
        if anchors_mask.sum():
            self.opacity_accum[anchors_mask] = torch.zeros(
                [anchors_mask.sum(), 1], device=self.device).float()
            self.anchor_denom[anchors_mask] = torch.zeros(
                [anchors_mask.sum(), 1], device=self.device).float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_denom[~prune_mask]
        del self.anchor_denom
        self.anchor_denom = temp_anchor_demon

        if prune_mask.shape[0]:
            valid_points_mask = ~prune_mask

            optimizable_tensors = prune_optimizer(valid_points_mask, optimizer)

            self._anchor = optimizable_tensors['anchor']
            self._offset = optimizable_tensors['offset']
            self._anchor_feat = optimizable_tensors['anchor_feat']
            self._opacity = optimizable_tensors['opacity']
            self._scaling = optimizable_tensors['scaling']
            self._quaternion = optimizable_tensors['quaternion']

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        optimizer,
        check_interval: int = 100,
        success_threshold: float = 0.8,
        prune: bool = False,
    ):
        # Compute the average gradient for all included neural gaussians (anchors+offsets).
        grads = self.offset_gradient_accum / self.offset_denom  # [N*k,1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval *
                       success_threshold * 0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, max_grad, offset_mask, optimizer)

        if prune:
            self.prune_anchors(min_opacity, optimizer,
                               check_interval, success_threshold)

    def eval(self):
        super().eval()

        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()

        if self.app_embedding is not None:
            self.app_embedding.eval()

        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        super().train()

        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()

        if self.app_embedding is not None:
            self.app_embedding.eval()

        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    @torch.no_grad()
    def save_ply(self, path: str):
        xyz = detach_tensor_to_numpy(self.get_anchor)
        rgb = np.zeros_like(xyz)
        rgb[..., -1] = 0.8 * 255.
        normal = np.zeros_like(xyz)

        dtype_full = [(attribute, 'f4')
                      for attribute in ['x', 'y', 'z', 'nx', 'ny', 'nz']]
        dtype_full += [(attribute, 'u1')
                       for attribute in ['red', 'green', 'blue']]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normal, rgb,), axis=1)

        elements[:] = list(map(tuple, attributes))
        ply = PlyElement.describe(elements, 'vertex')
        PlyData([ply]).write(path)

    @torch.no_grad()
    def save_colmap_ply(self, path: str):
        pass

    def save_splat(self, output_path: str = ""):
        # return super().save_splat(output_path)
        pass
