# pylint: disable=[E0402, C0103]

import os
from scipy.spatial.transform import Rotation as R

import torch
import numpy as np

from .database import COLMAPDatabase, array_to_blob, fetch_images_from_database
from .read_write_model import (
    Camera, Image, write_cameras_text, write_images_text, write_points3D_text
)
from .utils import (
    list_images,
    list_metadata,
    get_filename_from_path,
    get_camera_id,
    read_meganerf_mappings,
)


MILL19_SCENES = ['building', 'rubble']
URBAN_SCENES = ['ArtsQuad', 'Sci-Art', 'Residence']
COLMAP_PATH = '/usr/local/bin/colmap'


DRB_TO_RDF = torch.tensor([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1]
]).float()


def convert_artsquad_to_colmap(data_dir: str, scene: str):
    scene_dir = os.path.join(data_dir, scene)
    colmap_dir = os.path.join(scene_dir, "sparse/0")
    os.makedirs(colmap_dir, exist_ok=True)

    database_path = os.path.join(scene_dir, 'database.db')

    mappings_path = os.path.join(scene_dir, 'mappings.txt')
    image_name_to_metadata, metadata_to_image_name = read_meganerf_mappings(mappings_path)
    train_dir = os.path.join(scene_dir, 'train')
    val_dir = os.path.join(scene_dir, 'val')

    train_metadata_dir = os.path.join(train_dir, 'metadata')
    val_metadata_dir = os.path.join(val_dir, 'metadata')
    new_metadata_dir = os.path.join(scene_dir, 'metadata')
    os.makedirs(new_metadata_dir, exist_ok=True)
    os.system(f'cp -r {train_metadata_dir}/* {new_metadata_dir}')
    os.system(f'cp -r {val_metadata_dir}/* {new_metadata_dir}')
    all_metadata_paths = list_metadata(new_metadata_dir)
    all_metadata_paths.sort()

    assert len(image_name_to_metadata) == len(all_metadata_paths)

    name_to_image_id = fetch_images_from_database(database_path)

    W, H = None, None
    images, cameras = {}, {}

    for index, metadata_path in enumerate(all_metadata_paths): # pylint: disable=[W0612]
        metadata = torch.load(metadata_path, map_location='cpu')
        metadata_name = get_filename_from_path(str(metadata_path))
        image_name = metadata_to_image_name[metadata_name]
        image_id = name_to_image_id[image_name]

        c2w = torch.eye(4)
        c2w[:3, 0:1] = -metadata["c2w"][:, 1:2]
        c2w[:3, 1:2] = metadata["c2w"][:, 0:1]
        c2w[:3, 2:4] = metadata["c2w"][:, 2:4]
        c2w[:3, :3] = DRB_TO_RDF @ c2w[:3, :3] @ DRB_TO_RDF
        c2w[:3, 3:] = DRB_TO_RDF @ c2w[:3, 3:]
        w2c = torch.inverse(c2w)
        qvec = R.from_matrix(w2c[:3, :3].numpy()).as_quat()
        tvec = w2c[:3, 3]
        qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])
        tvec = np.array([tvec[0], tvec[1], tvec[2]])

        fx = metadata["intrinsics"][0].item()
        fy = metadata["intrinsics"][1].item()
        cx = metadata["intrinsics"][2].item()
        cy = metadata["intrinsics"][3].item()
        w, h = metadata["W"], metadata["H"]
        W, H = w, h

        params = [fx, fy, cx, cy]
        camera = Camera(id=-1, model="PINHOLE", width=W, height=H, params=params)
        camera_id = get_camera_id(cameras, camera)
        camera = Camera(camera_id, model="PINHOLE", width=W, height=H, params=params)
        cameras[camera_id] = camera

        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=[],
            point3D_ids=[],
        )

        # print(f'W: {W}, H: {H}')
        # print(f'fx fy cx cy: {fx} {fy} {cx} {cy}')
        # print(f'params: {params}')

    db = COLMAPDatabase.connect(database_path)
    db.execute("DELETE FROM cameras")
    for camera_id, camera in cameras.items():
        db.execute('INSERT OR IGNORE INTO cameras (camera_id, model, width, height, params) ' +
                   'VALUES (?, ?, ?, ?, ?)',
                   (camera_id, camera.model, camera.width, camera.height,
                    array_to_blob(np.array(camera.params)))
        )

    write_cameras_text(cameras, os.path.join(colmap_dir, "cameras.txt"))
    write_images_text(images, os.path.join(colmap_dir, "images.txt"))
    write_points3D_text({}, os.path.join(colmap_dir, "points3D.txt"))


def convert_mill19_to_colmap(data_dir: str, scene: str):
    scene_dir = os.path.join(data_dir, scene)
    colmap_dir = os.path.join(scene_dir, "sparse/0")
    os.makedirs(colmap_dir, exist_ok=True)

    database_path = os.path.join(scene_dir, 'database.db')

    train_dir = os.path.join(scene_dir, 'train')
    val_dir = os.path.join(scene_dir, 'val')

    train_image_dir = os.path.join(train_dir, 'rgbs')
    val_image_dir = os.path.join(val_dir, 'rgbs')

    new_image_dir = os.path.join(scene_dir, 'images')
    os.makedirs(new_image_dir, exist_ok=True)
    os.system(f'cp -r {train_image_dir}/* {new_image_dir}')
    os.system(f'cp -r {val_image_dir}/* {new_image_dir}')
    new_image_paths = list_images(new_image_dir)
    new_image_paths.sort()

    train_metadata_dir = os.path.join(train_dir, 'metadata')
    val_metadata_dir = os.path.join(val_dir, 'metadata')
    new_metadata_dir = os.path.join(scene_dir, 'metadata')
    os.makedirs(new_metadata_dir, exist_ok=True)
    os.system(f'cp -r {train_metadata_dir}/* {new_metadata_dir}')
    os.system(f'cp -r {val_metadata_dir}/* {new_metadata_dir}')
    all_metadata_paths = list_metadata(new_metadata_dir)
    all_metadata_paths.sort()

    assert len(new_image_paths) == len(all_metadata_paths)

    name_to_image_id = fetch_images_from_database(database_path)

    W, H = None, None
    images, cameras = {}, {}
    camera_id = 1
    for index, metadata_path in enumerate(all_metadata_paths):
        metadata = torch.load(metadata_path, map_location='cpu')
        image_path = new_image_paths[index]

        image_name = get_filename_from_path(str(image_path))
        image_id = name_to_image_id[image_name]
        c2w = torch.eye(4)
        c2w[:3, 0:1] = -metadata["c2w"][:, 1:2]
        c2w[:3, 1:2] = metadata["c2w"][:, 0:1]
        c2w[:3, 2:4] = metadata["c2w"][:, 2:4]
        c2w[:3, :3] = DRB_TO_RDF @ c2w[:3, :3] @ DRB_TO_RDF
        c2w[:3, 3:] = DRB_TO_RDF @ c2w[:3, 3:]
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

        fx = metadata["intrinsics"][0].item()
        fy = metadata["intrinsics"][1].item()
        cx = metadata["intrinsics"][2].item()
        cy = metadata["intrinsics"][3].item()
        w, h = metadata["W"], metadata["H"]
        W, H = w, h

    params = [fx, fy, cx, cy]

    cameras[0] = Camera(id=camera_id, model="PINHOLE", width=W, height=H, params=params)

    write_cameras_text(cameras, os.path.join(colmap_dir, "cameras.txt"))
    write_images_text(images, os.path.join(colmap_dir, "images.txt"))
    write_points3D_text({}, os.path.join(colmap_dir, "points3D.txt"))


if __name__ == "__main__":
    for scene in MILL19_SCENES:
        convert_mill19_to_colmap(data_dir="/home/chenyu/HD_Datasets/datasets/aerial", scene=scene)

    for scene in URBAN_SCENES:
        convert_artsquad_to_colmap(data_dir="/home/chenyu/HD_Datasets/datasets/aerial", scene=scene)
