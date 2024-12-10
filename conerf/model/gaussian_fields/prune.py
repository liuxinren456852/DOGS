import gc
from typing import List

import torch
import tqdm
from omegaconf import OmegaConf

from conerf.geometry.camera import Camera
from conerf.model.gaussian_fields.gaussian_splat_model import GaussianSplatModel
from conerf.render.gaussian_render import count_render


@torch.no_grad()
def calculate_v_imp_score(gaussians: GaussianSplatModel, imp_list: List, v_pow: float):
    """
    :param gaussians: A data structure containing Gaussian components with a get_scaling method
    :param imp_list: The importance scores for each Gaussian componnet.
    :param v_pow: The power to which the volume ratios are raised.
    :return: A list of adjusted values (v_list) used for pruning.
    """
    # Calculate the volume of each Gaussian component.
    volume = torch.prod(gaussians.get_scaling, dim=1)
    # Determine the kth_percent_largest value
    index = int(len(volume) * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    v_list = torch.pow(volume / kth_percent_largest, v_pow)
    v_list = v_list * imp_list

    return v_list


@torch.no_grad()
def prune_list(
    gaussians: GaussianSplatModel,
    cameras: List[Camera],
    pipeline_config: OmegaConf,
    bkgd_color: torch.Tensor
):
    gaussian_list, imp_list = None, None
    camera = cameras.pop()
    camera = camera.copy_to_device("cuda")
    render_pkg = count_render(gaussians, camera, pipeline_config, bkgd_color)
    gaussian_list, imp_list = (
        render_pkg["gaussians_count"],
        render_pkg["important_score"],
    )

    pbar = tqdm.trange(len(cameras), desc="Computing pruning statistics", leave=False)
    for iteration in range(len(cameras)): # pylint: disable=W0612
        camera = cameras.pop()
        camera = camera.copy_to_device("cuda")
        render_pkg = count_render(gaussians, camera, pipeline_config, bkgd_color)

        gaussians_count, important_score = (
            render_pkg["gaussians_count"].detach(),
            render_pkg["important_score"].detach(),
        )
        gaussian_list += gaussians_count
        imp_list += important_score
        gc.collect()
        pbar.update(1)

    return gaussian_list, imp_list
