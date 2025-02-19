import argparse
import cv2
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from glob import glob
import json

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


def get_samples_llff(path):
    llffhold = 8

    imgdir = os.path.join(path)
    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    ]

    # Select the split.
    all_indices = np.arange(len(imgfiles))
    split_indices = {
        'test': all_indices[all_indices % llffhold == 0],
        'train': all_indices[all_indices % llffhold != 0],
    }

    n_input_views = 3
    train_indices = np.linspace(0, split_indices['train'].shape[0] - 1, n_input_views)
    train_indices = [round(i) for i in train_indices]

    train_indices = [split_indices['train'][i] for i in train_indices]
    print(train_indices)
    train_cam_infos_3 = [imgfiles[i].split('.')[0].split('/')[-1] for i in train_indices]


    n_input_views = 4
    train_indices = np.linspace(0, split_indices['train'].shape[0] - 1, n_input_views)
    train_indices = [round(i) for i in train_indices]

    train_indices = [split_indices['train'][i] for i in train_indices]
    print(train_indices)
    train_cam_infos_4 = [imgfiles[i].split('.')[0].split('/')[-1] for i in train_indices]

    return (set(sorted(train_cam_infos_3 + train_cam_infos_4)))




    if config.dtu_split_type == 'pixelnerf':
      train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
      exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
      test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
      split_indices = {'test': test_idx, 'train': train_idx}


def get_samples_dtu(path):

    imgdir = os.path.join(path)
    # imgfiles = [
    #     os.path.join(imgdir, f)
    #     for f in sorted(os.listdir(imgdir))
    #     if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    # ]
    imgfiles = glob(f"{imgdir}/rect*.png") + glob(f"{imgdir}/rect*.jpg")
    # print(imgfiles)
    # exit()

    train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]

    return [imgfiles[x].split('.')[0].split('/')[-1] for x in train_idx[:4]]

    # return (set(sorted(train_cam_infos_3 + train_cam_infos_4)))


def get_samples_kinect(path):
    llffhold = 8

    imgdir = os.path.join(path)
    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    ]

    train_indices = []
    for i, f in enumerate(imgfiles):
        # assert(f.endswith('JPG') or f.endswith('jpg') or f.endswith('png'))
        if f.endswith('_train.JPG') or f.endswith('_train.jpg') or f.endswith('_train.png'):
            train_indices.append(i)

    print(train_indices)
    train_cam_infos = [imgfiles[i].split('.')[0].split('/')[-1] for i in train_indices]

    return (set(train_cam_infos))


def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])

    args = parser.parse_args()

    # margin_width = 50
    # caption_height = 60

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 1
    # font_thickness = 2

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


    dataset = "/home/avinashpaliwal/github/data/nerf_llff_data"
    post    = "images_8"

    scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
    # scenes = ["flower"]
    sample_func = get_samples_llff


    # dataset = "/home/avinashpaliwal/github/data/DTU4/images"
    # post    = ""

    # scenes = os.listdir(dataset)
    # sample_func = get_samples_dtu


    # dataset = "/home/avinashpaliwal/github/data/NVS-RGBD/kinect"
    # post    = "images_2"
    # dataset = "/home/avinashpaliwal/github/data/NVS-RGBD/zed2"
    # post    = "images_4"
    # scenes = [f'scene0{x}' for x in range(1, 9)]

    # dataset = "/home/avinashpaliwal/github/data/NVS-RGBD/iphone"
    # post    = "images_8"

    # scenes = [f'scene0{x}' for x in range(1, 5)]
    # # scenes = ["flower"]
    # sample_func = get_samples_kinect



    for scene in tqdm(scenes):

        # path    = f"{dataset}/{scene}/9"

        # save_dir = f"{dataset}/{scene}/{post}/depth_rel"
        # os.makedirs(save_dir, exist_ok=True)

        # samples = sample_func(path, dataset, scene)


        path    = f"{scene}/6"

        save_dir = f"{scene}/{post}/depth_rel"
        os.makedirs(save_dir, exist_ok=True)

        samples = sample_func(path, scene)

        print(scene, samples)
        # exit()


        for sample in samples:

            raw_image = cv2.imread(f"{path}/{sample}.jpg")
            # raw_image = cv2.imread(f"{path}/{sample}.JPG")
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

            h, w = image.shape[:2]

            image = transform({'image': image})['image']
            image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                depth = depth_anything(image)

            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

            torch.save(depth.detach().cpu(), f"{save_dir}/{sample}.pt")
            # np.save(f"{save_dir}/{sample}.npy", depth.detach().cpu().numpy())

            # Visualization
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

            depth = depth.cpu().numpy().astype(np.uint8)

            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

            cv2.imwrite(f"{save_dir}/cc{sample}.png", depth)
