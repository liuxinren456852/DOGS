# pylint: disable=[E1101,E0402]

import os
import json

from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation as R

import torch
import numpy as np
import open3d as o3d
import tqdm

from conerf.datasets.utils import store_ply
from .database import COLMAPDatabase, array_to_blob, fetch_images_from_database
from .read_write_model import (
    Camera, Image, write_cameras_text, write_images_text, write_points3D_text
)
from .utils import list_images, get_camera_id, list_jsons


# ------------- settings -------------
SCENES = [
    # ("aerial_street_fusion", ""),           # both aerial and street
    # ("aerial_street_fusion", "street"),     # only street
    # ("aerial_street_fusion", "aerial"),     # only aerial
    ("small_city", ""),                     # all
]
COLMAP_PATH = '/usr/local/bin/colmap'
data_dir_list = [
    "/home/chenyu/HD_Datasets/datasets/MatrixCity",
]
DATA_DIR = None
for path in data_dir_list:
    if os.path.isdir(path):
        DATA_DIR = path
        break
if DATA_DIR is None:
    raise ValueError("Data directory not found!")
# -----------------------------------

# Refer to the link for COLMAP's coordinate system:
# https://colmap.github.io/format.html#images-txt
# And the issue: https://github.com/city-super/MatrixCity/issues/21 for
# MatrixCity's coordinate system.
MATRIX_CITY_TO_COLMAP = torch.FloatTensor([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])
TO_MANHATTAN_WORLD = torch.FloatTensor([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
])


def downsample_point_clouds(input_path: str, output_path: str, voxel_size: float = 0.5):
    pcd = o3d.io.read_point_cloud(input_path)
    print(f'input points number: {np.asarray(pcd.points).shape[0]}')
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    points = torch.from_numpy(np.asarray(down_pcd.points))
    points = TO_MANHATTAN_WORLD @ points.transpose(0, 1).float()
    down_pcd.points = o3d.utility.Vector3dVector(points.transpose(0, 1).numpy())
    print(f'output points number: {np.asarray(down_pcd.points).shape[0]}')
    # o3d.io.write_point_cloud(
    #     output_path, down_pcd, write_ascii=True, compressed=True
    # )
    points_np = np.asarray(down_pcd.points)
    colors_np = np.asarray(down_pcd.colors)

    store_ply(output_path, points_np, (colors_np * 255.).astype(np.uint8))


def read_transformations(json_path: str, image_width: int) -> Tuple[List, float]:
    with open(json_path, "r", encoding="utf-8") as json_file:
        meta = json.load(json_file)

    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * image_width / np.tan(0.5  * camera_angle_x)

    camtoworlds = [None] * len(meta["frames"])
    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        camtoworld = frame["rot_mat"]
        frame_index = frame["frame_index"]
        camtoworlds[frame_index] = camtoworld

    return camtoworlds, focal


def read_poses_json_aerial_street_fusion(
    json_path: str, images: Dict, cameras: Dict, name_to_image_id: Dict, data_type = "aerial"
) -> Tuple[Dict, Dict]:
    with open(json_path, "r", encoding="utf-8") as json_file:
        metadata = json.load(json_file)

    num_frames = len(metadata["frames"])
    camera_model = metadata["camera_model"]

    for i in range(num_frames):
        frame = metadata["frames"][i]
        fx, fy = frame["fl_x"], frame["fl_y"]
        cx, cy = frame["cx"], frame["cy"]
        w, h = frame["w"], frame["h"]
        assert fx == fy, "Invalid PINHOLE params!"

        params = [fx, cx, cy]
        camera = Camera(id=-1, model=camera_model, width=w, height=h, params=params)
        camera_id = get_camera_id(cameras, camera)
        camera = Camera(id=camera_id, model=camera_model, width=w, height=h, params=params)
        cameras[camera_id] = camera

        image_path = frame["file_path"]

        if data_type != '':
            image_name_start_index = image_path.find(data_type)
            if image_name_start_index < 0:
                continue

            image_name_start_index += (len(data_type) + 1)
            image_name = image_path[image_name_start_index:]
        else:
            image_name_start_index = image_path.find('street')
            if image_name_start_index >= 0:
                image_name = image_path[image_name_start_index:]
            else:
                image_name = image_path[image_path.find('aerial'):]
        # print(f'image name: {image_name}')
        image_id = name_to_image_id[image_name]
        c2w = torch.from_numpy(np.array(frame["transform_matrix"])).float()
        c2w[:3, :3] = TO_MANHATTAN_WORLD @ c2w[:3, :3] @ MATRIX_CITY_TO_COLMAP
        c2w[:3, 3] = TO_MANHATTAN_WORLD @ c2w[:3, 3]

        w2c = torch.inverse(c2w)
        qvec = R.from_matrix(w2c[:3, :3].numpy()).as_quat()
        tvec = w2c[:3, 3]
        qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])
        tvec = np.array([tvec[0], tvec[1], tvec[2]])

        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=[],
            point3D_ids=[],
        )

    return images, cameras


def convert_aerial_street_fusion(scene_dir: str, data_type: str = "aerial"):
    pose_dir = os.path.join(scene_dir, 'pose')
    poses_file = list_jsons(pose_dir)

    database_path = os.path.join(scene_dir, data_type, 'database.db')
    print(f'database path: {database_path}')
    assert os.path.exists(database_path)
    if data_type == "aerial":
        image_paths = list_images(os.path.join(scene_dir, data_type))
    elif data_type == "street":
        image_paths = list_images(os.path.join(scene_dir, "street/test")) + \
                      list_images(os.path.join(scene_dir, "street/train"))
    else:
        image_paths = list_images(os.path.join(scene_dir, "aerial")) + \
                      list_images(os.path.join(scene_dir, "street/test")) + \
                      list_images(os.path.join(scene_dir, "street/train"))

    assert len(image_paths) != 0
    name_to_image_id = fetch_images_from_database(database_path)

    images, cameras = {}, {}
    for json_filename in poses_file:
        read_poses_json_aerial_street_fusion(
            json_filename, images, cameras, name_to_image_id, data_type
        )

    db = COLMAPDatabase.connect(database_path) # pylint: disable=[C0103]
    db.execute("DELETE FROM cameras")
    for camera_id, camera in cameras.items():
        db.execute('INSERT OR IGNORE INTO cameras (camera_id, model, width, height, params) ' +
                   'VALUES (?, ?, ?, ?, ?)',
                   (camera_id, camera.model, camera.width, camera.height,
                    array_to_blob(np.array(camera.params)))
        )

    colmap_dir = os.path.join(scene_dir, data_type, "sparse/0")
    os.makedirs(colmap_dir, exist_ok=True)

    print(f'num images: {len(images)}')

    print(f'colmap data written to:{colmap_dir}')
    write_cameras_text(cameras, os.path.join(colmap_dir, "cameras.txt"))
    write_images_text(images, os.path.join(colmap_dir, "images.txt"))
    write_points3D_text({}, os.path.join(colmap_dir, "points3D.txt"))


def read_poses_json_small_city(
    json_path: str, images: Dict, cameras: Dict, name_to_image_id: Dict, data_type = "aerial"
) -> Tuple[Dict, Dict]:
    with open(json_path, "r", encoding="utf-8") as json_file:
        metadata = json.load(json_file)

    num_frames = len(metadata["frames"])
    print(f'num_frames: {num_frames}')
    camera_model = "SIMPLE_PINHOLE"

    fx, fy = metadata["fl_x"], metadata["fl_y"]
    cx, cy = metadata["cx"], metadata["cy"]
    w, h = int(metadata["w"]), int(metadata["h"])
    assert fx == fy, "Invalid PINHOLE params!"

    for i in range(num_frames):
        frame = metadata["frames"][i]

        params = [fx, cx, cy]
        camera = Camera(id=-1, model=camera_model, width=w, height=h, params=params)
        camera_id = get_camera_id(cameras, camera)
        camera = Camera(id=camera_id, model=camera_model, width=w, height=h, params=params)
        cameras[camera_id] = camera

        image_path = frame["file_path"]

        if data_type != '':
            image_name_start_index = image_path.rfind('..')

            image_name_start_index += 3
            image_name = image_path[image_name_start_index:]
        else:
            image_name_start_index = image_path.find('street')
            if image_name_start_index >= 0:
                image_name = image_path[image_name_start_index:]
            else:
                image_name = image_path[image_path.find('aerial'):]

        # print(f'image name: {image_name}')
        image_id = name_to_image_id[image_name]
        c2w = torch.from_numpy(np.array(frame["transform_matrix"])).float()
        c2w[:3, :3] = TO_MANHATTAN_WORLD @ (c2w[:3, :3] @ MATRIX_CITY_TO_COLMAP)
        c2w[:3, 3] = TO_MANHATTAN_WORLD @ c2w[:3, 3]

        w2c = torch.inverse(c2w)
        qvec = R.from_matrix(w2c[:3, :3].numpy()).as_quat()
        tvec = w2c[:3, 3]
        qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])
        tvec = np.array([tvec[0], tvec[1], tvec[2]])

        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=[],
            point3D_ids=[],
        )

    return images, cameras


def convert_small_city_aerial(scene_dir: str, data_type: str = "aerial"):
    root_dir = os.path.join(scene_dir, data_type)

    if data_type == 'aerial':
        pose_dir = os.path.join(root_dir, 'pose', 'block_all')
    else:
        pose_dir = root_dir
    poses_file = list_jsons(pose_dir)

    database_path = os.path.join(root_dir, 'database.db')
    print(f'database path: {database_path}')
    assert os.path.exists(database_path)

    name_to_image_id = fetch_images_from_database(database_path)

    images, cameras = {}, {}
    print(f'num_poses file: {len(poses_file)}')
    for json_filename in poses_file:
        read_poses_json_small_city(json_filename, images, cameras, name_to_image_id)

    db = COLMAPDatabase.connect(database_path) # pylint: disable=[C0103]
    db.execute("DELETE FROM cameras")
    for camera_id, camera in cameras.items():
        db.execute('INSERT OR IGNORE INTO cameras (camera_id, model, width, height, params) ' +
                   'VALUES (?, ?, ?, ?, ?)',
                   (camera_id, camera.model, camera.width, camera.height,
                    array_to_blob(np.array(camera.params)))
        )

    colmap_dir = os.path.join(root_dir, "sparse/0")
    os.makedirs(colmap_dir, exist_ok=True)

    write_cameras_text(cameras, os.path.join(colmap_dir, "cameras.txt"))
    write_images_text(images, os.path.join(colmap_dir, "images.txt"))
    write_points3D_text({}, os.path.join(colmap_dir, "points3D.txt"))

    print("Downsampling point clouds...")
    if data_type == 'aerial':
        ply_dir = os.path.join(root_dir, 'pcd', data_type)
        ply_path = os.path.join(ply_dir, 'Block_all.ply')
    else:
        ply_dir = os.path.join(root_dir, 'pcd')
        ply_path = os.path.join(ply_dir, 'Block_A.ply')
    output_ply_path = os.path.join(colmap_dir, "points3D.ply")
    downsample_point_clouds(ply_path, output_ply_path, 0.01)


def move_images_to_new_dir(src_dir: str, dst_dir: str, json_path: str):
    with open(json_path, "r", encoding="utf-8") as json_file:
        metadata = json.load(json_file) 
    num_frames = len(metadata["frames"])
    pbar = tqdm.trange(num_frames, desc="Moving images to new folder...", leave=False)
    for i in range(num_frames):
        frame = metadata["frames"][i]
        image_path = frame["file_path"]
        image_name_start_index = image_path.rfind('..')

        image_name_start_index += 3
        image_name = image_path[image_name_start_index:]

        image_path = os.path.join(src_dir, image_name)
        new_image_path = os.path.join(dst_dir, image_name)
        new_image_dir = os.path.split(new_image_path)[0]
        if not os.path.exists(new_image_dir):
            os.makedirs(new_image_dir, exist_ok=True)
        # print(f'image_path: {image_path}')
        # print(f'new_image_path: {new_image_path}')
        os.system(f"cp {image_path} {new_image_dir}")
        pbar.update(1)


def preprocess_matrix_city_street_blockA(scene_dir: str, data_type: str = "street"):
    root_dir = os.path.join(scene_dir, data_type)

    train_json_file = os.path.join(root_dir, "pose/block_A/transforms_train.json")
    test_json_file = os.path.join(root_dir, "pose/block_A/transforms_test.json")

    dst_dir = os.path.join(root_dir, 'block_A')
    os.makedirs(dst_dir, exist_ok=True)

    os.system(f'cp {train_json_file} {dst_dir}')
    os.system(f'cp {test_json_file} {dst_dir}')

    move_images_to_new_dir(root_dir, dst_dir, train_json_file)
    move_images_to_new_dir(root_dir, dst_dir, test_json_file)


def convert_matrix_city_to_colmap(data_dir: str, scene: str, data_type: str):
    scene_dir = os.path.join(data_dir, scene)
    if scene == "aerial_street_fusion":
        convert_aerial_street_fusion(scene_dir, data_type)
    elif scene == "small_city":
        assert data_type == ""
        # convert_small_city_aerial(scene_dir, "aerial")
        convert_small_city_aerial(scene_dir, "street/block_A")
        print('end convert small city')


if __name__ == "__main__":
    # Uncomment the code below if you want to convert the block_A to COLMAP format.
    # preprocess_matrix_city_street_blockA(
    #     os.path.join(DATA_DIR, "small_city")
    # )
    for scene, data_type in SCENES:
        convert_matrix_city_to_colmap(
            data_dir=DATA_DIR,
            scene=scene,
            data_type=data_type
        )
