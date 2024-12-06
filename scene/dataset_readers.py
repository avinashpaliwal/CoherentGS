#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from PIL import Image
from typing import NamedTuple
import numpy as np
from plyfile import PlyData, PlyElement
from kornia.geometry.depth import depth_to_3d
import torch
import yaml

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from scene.gaussian_model import BasicPointCloud
from bilateral_filtering import sparse_bilateral_filtering
from scene.depth_layering import get_depth_bins

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    render_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    radii2 = vertices['radius2'][..., None]
    return BasicPointCloud(points=positions, colors=colors, normals=normals, radii2=radii2)

def storePly(path, xyz, rgb, normals, radii2):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('radius2', 'f4')]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, radii2), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


config = yaml.safe_load(open('argument.yaml', 'r'))

def readLLFFInfo(path, num_cameras, num_mask_channels):

    def generate_spiral_path(poses, bounds, fix_rot, n_frames=120, n_rots=1, zrate=.5):
        def poses_avg(poses):
            """New pose using average position, z-axis, and up vector of input poses."""
            position = poses[:, :3, 3].mean(0)
            z_axis = poses[:, :3, 2].mean(0)
            up = poses[:, :3, 1].mean(0)
            cam2world = viewmatrix(z_axis, up, position)
            return cam2world

        def viewmatrix(lookdir, up, position, subtract_position=False):
            """Construct lookat view matrix."""
            vec2 = normalize((lookdir - position) if subtract_position else lookdir)
            vec0 = normalize(np.cross(up, vec2))
            vec1 = normalize(np.cross(vec2, vec0))
            m = np.stack([vec0, vec1, vec2, position], axis=1)
            return m

        def normalize(x):
            """Normalization helper function."""
            return x / np.linalg.norm(x)
    
        """Calculates a forward facing spiral path for rendering."""
        # Find a reasonable 'focus depth' for this dataset as a weighted average
        # of near and far bounds in disparity space.
        close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
        dt = .75
        focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

        # Get radii for spiral path using 90th percentile of camera positions.
        positions = poses[:, :3, 3]
        radii = np.percentile(np.abs(positions), 90, 0)
        radii = np.concatenate([radii, [1.]])

        print(focal, radii)

        # Generate poses for spiral path.
        render_poses = []
        cam2world = poses_avg(poses)
        up = poses[:, :3, 1].mean(0)
        for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
            t = radii * [np.cos(theta), np.sin(theta), np.sin(theta * zrate), 1.]
            position = cam2world @ t
            lookat = cam2world @ [0, 0, focal, 1.]
            z_axis = -position + lookat
            render_poses.append(viewmatrix(z_axis, up, position))
        render_poses = np.stack(render_poses, axis=0)
        return render_poses

    factor = 8
    llffhold = 8
    n_input_views = num_cameras

    cam_infos = []

    # Load images.
    imgdir_suffix = ''
    if factor > 0:
      imgdir_suffix = f'_{factor}'
    else:
      factor = 1
    imgdir = os.path.join(path, 'images' + imgdir_suffix)
    if not os.path.isdir(imgdir):
      raise ValueError(f'Image folder {imgdir} does not exist.')
    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    ]

    for i, imgfile in enumerate(imgfiles):
      with open(imgfile, 'rb') as imgin:
        image = Image.open(imgin).convert('RGB')
        # norm_data = im_data / 255.0
        # image = Image.fromarray(np.array(norm_data*255.0, dtype=np.byte), "RGB")
        image_name = os.path.basename(imgfile)#.split(".")[0]

        width, height = image.size
        # print(image.size)
        cam_info = CameraInfo(uid=i, R=None, T=None, FovY=None, FovX=None, image=image,
                                image_path=imgfile, image_name=image_name, width=width, height=height)

        cam_infos.append(cam_info)


    # Load poses and bounds.
    with open(os.path.join(path, 'poses_bounds.npy'),
                         'rb') as fp:
      poses_arr = np.load(fp)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    bounds = poses_arr[:, -2:]

    # Pull out focal length before processing poses.
    focal = poses[0, -1, -1] / factor

    # Correct rotation matrix ordering (and drop 5th column of poses).
    fix_rotation = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
                            dtype=np.float32)

    poses = poses[:, :3, :4] @ fix_rotation

    # Rescale according to a default bd factor.
    scale = 1. / (bounds.min() * .75)
    poses[:, :3, 3] *= scale
    bounds *= scale

    # Center and scale poses.
    camtoworlds = poses

    # FOV processing
    FovY = focal2fov(focal, height)
    FovX = focal2fov(focal, width)

    for i in range(1, camtoworlds.shape[0] + 1):

        cam_infos[i-1] = cam_infos[i-1]._replace(FovX=FovX)
        cam_infos[i-1] = cam_infos[i-1]._replace(FovY=FovY)

        # get the world-to-camera transform and set R, T
        pose = np.eye(4, dtype=np.float32)
        pose[:3] = camtoworlds[i-1]
        w2c = np.linalg.inv(pose)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        cam_infos[i-1] = cam_infos[i-1]._replace(R=R)
        cam_infos[i-1] = cam_infos[i-1]._replace(T=T)

    # Select the split.
    all_indices = np.arange(len(cam_infos))
    split_indices = {
        'test': all_indices[all_indices % llffhold == 0],
        'train': all_indices[all_indices % llffhold != 0],
    }

    train_indices = np.linspace(0, split_indices['train'].shape[0] - 1, n_input_views)
    train_indices = [round(i) for i in train_indices]

    train_indices = [split_indices['train'][i] for i in train_indices]
    train_cam_infos = [cam_infos[i] for i in train_indices]
    test_cam_infos  = [cam_infos[i] for i in split_indices['test']]

    render_cam_infos = []
    render_poses = generate_spiral_path(camtoworlds, bounds, fix_rotation, n_frames=120)

    for i, render_pose in enumerate(render_poses):
        # get the world-to-camera transform and set R, T
        pose = np.eye(4, dtype=np.float32)
        pose[:3] = render_pose
        w2c = np.linalg.inv(pose)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        cam_info = CameraInfo(uid=i, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=None, image_name=None, width=width, height=height)
        render_cam_infos.append(cam_info)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # SETUP train depth etc
    xyz_arr = []
    rgb_arr = []
    radii2_arr = []
    dep_mask_arr = []

    for idx, cam_info in enumerate(train_cam_infos):

        im_data = np.array(cam_info.image, dtype=np.float32)

        # Disparity actually
        disparity = torch.load(f'{os.path.dirname(cam_info.image_path)}/depth_rel/{os.path.splitext(cam_info.image_name)[0]}.pt').detach().cpu()
        disparity_max = disparity.max()
        disparity_norm = (disparity / disparity_max) + 0.2

        _, vis_depths = sparse_bilateral_filtering((disparity_norm).numpy().copy(), im_data.copy()[..., :3], config, num_iter=config['sparse_iter'], spdb=False)
        
        disparity = (torch.tensor(vis_depths[-1])[None, None] - 0.2) * disparity_max
        bins = get_depth_bins(disparity=disparity, num_bins=num_mask_channels)
        bins = [1 / x for x in bins]
        bins.reverse()

        depth_filtered = 1/vis_depths[-1]
        depth = torch.Tensor(depth_filtered)[None, None]

        # Cluster depth to create multi-channel mask
        dep_masks = []
        for i in range(len(bins[:-1])):
            dep = disparity[0, 0]
            dep = np.where((dep > bins[i]) & (dep <= bins[i+1]), 1, 0)
            dep_masks.append(torch.tensor(dep[None]))
        dep_masks = torch.cat(dep_masks, dim=0)[None]

        focal = .5 * width / np.tan(.5 * float(cam_info.FovX))

        # Init gaussian radius equal to shorter length of the rectangle. Default: Height
        fovy = focal2fov(fov2focal(cam_info.FovX, width), height)
        # Radii per frame
        radii = np.tan(0.5 * float(fovy))  * depth / height
        radii2 = radii**2
        radii2 = radii2[0].permute(1, 2, 0).reshape(-1).numpy()

        K = torch.eye(3)[None]
        K[:, 0, 0] = focal
        K[:, 0, 2] = width / 2.0
        K[:, 1, 1] = focal
        K[:, 1, 2] = height / 2.0


        camera3d = depth_to_3d(depth, K)
        xyz_cam = camera3d[0].permute(1, 2, 0).reshape(-1, 3).numpy()

        rgb = torch.Tensor(im_data).reshape(-1, 3).numpy()

        xyz_arr.append(xyz_cam)
        rgb_arr.append(rgb)
        radii2_arr.append(radii2)
        dep_mask_arr.append(dep_masks)

    dep_mask_arr = torch.cat(dep_mask_arr, dim=0).float().cuda()

    # Get optical flows and consistency masks
    image_names = [cam_info.image_name.split('.')[0] for cam_info in train_cam_infos]
    flows, masks = {}, {}
    for idx, im1 in enumerate(image_names):
        for idx2, im2 in enumerate(image_names):
            if idx == idx2:
                continue

            flows[f"{idx}_{idx2}"] = torch.tensor(np.load(f"{os.path.dirname(cam_info.image_path)}/flow/{im1}_{im2}.npy")).cuda()
            masks[f"{idx}_{idx2}"] = torch.tensor(np.load(f"{os.path.dirname(cam_info.image_path)}/flow/{im1}_{im2}w.npy")).cuda()


    xyz = np.concatenate(xyz_arr, axis=0)
    rgb = np.concatenate(rgb_arr, axis=0)
    radii2 = np.concatenate(radii2_arr, axis=0)[..., None]

    ply_path = os.path.join(path, "points3d.ply")

    if os.path.exists(ply_path):
        os.remove(ply_path)

    storePly(ply_path, xyz, rgb, np.zeros_like(xyz), radii2)
    pcd = fetchPly(ply_path)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           render_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info, flows, masks, dep_mask_arr

sceneLoadTypeCallbacks = {
    "Colmap": readLLFFInfo
}
