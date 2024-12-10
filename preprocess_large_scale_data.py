import os
import copy
import omegaconf

from omegaconf import OmegaConf

from conerf.datasets.dataset_base import MiniDataset
from conerf.datasets.utils import create_dataset, get_block_info_dir
from conerf.utils.config import config_parser, load_config
from conerf.utils.utils import setup_seed


def preprocess_large_scale_data(config: OmegaConf):
    print("Preprocessing Large Scale Data...")

    block_train_dataset = create_dataset(
        config=config,
        split=config.dataset.train_split,
        num_rays=config.dataset.get("num_rays", 256),
        apply_mask=config.dataset.apply_mask,
        device="cuda"
    )

    mx, my = block_train_dataset.mx, block_train_dataset.my # pylint: disable=C0103
    if mx is not None or my is not None:
        num_blocks = mx * my
    else:
        num_blocks = config.dataset.num_blocks

    # Export block data to disk.
    data_dir = os.path.join(config.dataset.root_dir, config.dataset.scene)
    save_dir = get_block_info_dir(data_dir, num_blocks, mx, my)
    os.makedirs(save_dir, exist_ok=True)

    for k in range(num_blocks):
        block_train_dataset.move_to_block(k)
        dataset_k = MiniDataset(
            cameras=block_train_dataset.cameras,
            camtoworlds=block_train_dataset.camtoworlds,
            block_id=block_train_dataset.current_block,
        )
        block_dir = os.path.join(save_dir, f'block_{k}')
        os.makedirs(block_dir, exist_ok=True)
        dataset_k.write(block_dir)


if __name__ == "__main__":
    args = config_parser()

    # parse YAML config to OmegaConf
    config = load_config(args.config)
    config["config_file_path"] = args.config

    assert config.dataset.scene != ""
    assert config.dataset.multi_blocks is True

    setup_seed(config.seed)

    scenes = []
    if (
        type(config.dataset.scene) == omegaconf.listconfig.ListConfig # pylint: disable=C0123
    ):
        scene_list = list(config.dataset.scene)
        for sc in config.dataset.scene:
            scenes.append(sc)
    else:
        scenes.append(config.dataset.scene)

    for scene in scenes:
        local_config = copy.deepcopy(config)
        local_config.dataset.scene = scene

        data_dir = os.path.join(config.dataset.root_dir, scene)
        assert os.path.exists(data_dir), f"Dataset does not exist: {data_dir}!"

        preprocess_large_scale_data(local_config)
