import sys
sys.path.append('core')

from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.submissions import get_cfg
from core.utils.misc import process_cfg
import datasets
from utils import flow_viz
from utils import frame_utils
import cv2
import math
import os.path as osp

from core.FlowFormer import build_flowformer
from backwarp import backWarp
from kornia.morphology import erosion

from utils.utils import InputPadder, forward_interpolate
import itertools

epsilon = 0.00001

def consistency(x, flow, warper):
    warped = warper(x, flow)

    # return warped

    dist   = torch.sum(torch.abs(warped-flow),1,keepdim=True)
    weight = torch.exp(-0.5*dist)
    # print(weight.max(), weight.min(), x.shape[-1])
    # weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
    return weight#_normalized

def compute_grid_indices(image_shape, patch_size=None, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]

def compute_flow(model, image1, image2, weights=None, train_size=None):
    print(f"computing flow...")

    image_size = image1.shape[1:]

    image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size, patch_size=train_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre

    return flow

def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    return model

def visualize_flow(viz_root_dir, model, img_pairs, images, train_size):
    weights = None

    flows = {}
    for img_pair in img_pairs:
        fn1, fn2 = img_pair.split('+')
        # fn1, fn2 = int(fn1), int(fn2)
        image1, image2 = images[fn1][:3], images[fn2][:3]
        # alpha1, alpha2 = images[fn1][None][:, 3:].cuda(), images[fn2][None][:, 3:].cuda()
        # print(image1.shape, image2.shape, alpha1.max(), alpha1.min(), alpha2.max(), alpha2.min())
        print(f"processing {fn1}, {fn2}...")

        flow = compute_flow(model, image1, image2, weights, train_size)# * alpha1 / 255
        flows[img_pair] = flow
        flow_img = flow_viz.flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy())
        cv2.imwrite(f"{viz_root_dir}/{fn1}_{fn2}.png", flow_img[:, :, [2,1,0]])
        np.save(f"{viz_root_dir}/{fn1}_{fn2}.npy", flow[0].permute(1, 2, 0).cpu().numpy())

    # max_side_length = max(train_size)

    _, C, H, W = flow.shape
    warper = backWarp(W, H, "cuda")

    for img_pair in img_pairs:
        fn1, fn2 = img_pair.split('+')
        flow01 = flows[img_pair]# / max_side_length
        flow10 = flows[f"{fn2}+{fn1}"]# / max_side_length
        # print(flow01.max(), flow01.min(), max_side_length)
        # fn1, fn2 = int(fn1), int(fn2)
        image1, image2 = images[fn1][None][:, :3].cuda(), images[fn2][None][:, :3].cuda()
        # alpha1, alpha2 = images[fn1][None][:, 3:].cuda(), images[fn2][None][:, 3:].cuda()
        # flow01 *= 1 - alpha1
        # flow10 *= 1 - alpha2
        print(img_pair)

        # w01 = (consistency(-flow10, flow01)[0, 0].cpu().numpy() * 255).astype(np.uint8)
        # w01 = (consistency(image2, flow01, warper)[0].permute(1, 2, 0).cpu().numpy()[:, :, [2,1,0]]).astype(np.uint8)
        # cv2.imwrite(f"{viz_root_dir}/{fn1}_{fn2}w.png", w01)

        w01 = consistency(-flow10, flow01, warper)[0, 0].cpu()
        kernel = torch.ones(5, 5)
        w01 = erosion(w01[None, None], kernel).numpy()[0, 0]
        # # w01 = (consistency(image2, flow01)[0].permute(1, 2, 0).cpu().numpy()[:, :, [2,1,0]]).astype(np.uint8)
        np.save(f"{viz_root_dir}/{fn1}_{fn2}w.npy", w01)
        cv2.imwrite(f"{viz_root_dir}/{fn1}_{fn2}w.png", (w01 * 255).astype(np.uint8))
        # print((w01 * 255).astype(np.uint8).shape, (w01 * 255).astype(np.uint8).max(), (w01 * 255).astype(np.uint8).min())

def generate_pairs(dirname, image_paths):
    # image_paths = os.listdir(dirname)

    images = {}
    for idx, path in enumerate(image_paths):
        image = frame_utils.read_gen(f"{dirname}/{path}.jpg")
        image = np.array(image).astype(np.uint8)
        H, W, C = image.shape
        # image = cv2.resize(image, dsize=(W//4, H//4), interpolation=cv2.INTER_CUBIC)
        # print(image.shape, H, W)
        # exit()

        # cv2.imwrite(f"{viz_root_dir}/{idx}_in.png", image[:, :, [2,1,0]])
        # print(image.shape)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        images[path] = image

    img_pairs = []
    for path1 in (image_paths):
        for path2 in (image_paths):
            if path1 == path2:
                continue

            img_pairs.append(f"{path1}+{path2}")

    return images, img_pairs


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

    # return sorted(train_cam_infos_3)

    n_input_views = 4
    train_indices = np.linspace(0, split_indices['train'].shape[0] - 1, n_input_views)
    train_indices = [round(i) for i in train_indices]

    train_indices = [split_indices['train'][i] for i in train_indices]
    print(train_indices)
    train_cam_infos_4 = [imgfiles[i].split('.')[0].split('/')[-1] for i in train_indices]

    return sorted(train_cam_infos_3), sorted(train_cam_infos_4)#(set(sorted(train_cam_infos_3 + train_cam_infos_4)))


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

    return ((train_cam_infos))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', default='sintel')
    parser.add_argument('--sintel_dir', default='datasets/Sintel/test/clean')
    parser.add_argument('--viz_root_dir', default='viz_results')

    args = parser.parse_args()


    model = build_model()

    dataset = "/home/avinashpaliwal/github/data/nerf_llff_data"
    post    = "images_8"

    scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
    # scenes = ["flower"]
    # scenes = ["fortress", "horns", "leaves", "orchids", "room", "trex"]

    get_samples = get_samples_llff


    # dataset = "/home/avinashpaliwal/github/data/DTU4/images"
    # post    = ""

    # scenes = os.listdir(dataset)
    # get_samples = get_samples_dtu

    # dataset = "/home/avinashpaliwal/github/data/NVS-RGBD/kinect"
    # post    = "images_2"
    # dataset = "/home/avinashpaliwal/github/data/NVS-RGBD/zed2"
    # post    = "images_4"

    # scenes = [f'scene0{x}' for x in range(1, 9)]


    # dataset = "/home/avinashpaliwal/github/data/NVS-RGBD/iphone"
    # post    = "images_8"

    # scenes = [f'scene0{x}' for x in range(1, 5)]
    # # scenes = ["flower"]
    # # scenes = ["flower"]
    # get_samples = get_samples_kinect

    dataset_id = "llff"

    for scene in scenes:
        path = f"{dataset}/{scene}/{post}"


        viz_root_dir = f"{path}/flow"
        if not osp.exists(viz_root_dir):
            os.makedirs(viz_root_dir)

        if dataset_id == "llff":
            samples3, samples4 = get_samples(path)

            # image_paths = [f"{path}/{sample}" for sample in samples]

            # print(samples)
            # images, img_pairs = generate_pairs(path, samples)
            # print(images.keys(), img_pairs)
            # exit()

            images3, img_pairs3 = generate_pairs(path, samples3)
            images4, img_pairs4 = generate_pairs(path, samples4)
            print(images3.keys(), img_pairs3)
            print(images4.keys(), img_pairs4)

            images4.update(images3)
            images = images4
            samples = samples4
            img_pairs4.extend(img_pairs3)
            img_pairs = sorted(list(set(img_pairs4)))

            print(images4.keys(), img_pairs4)

        elif dataset_id == "dtu":
            samples = get_samples(path)

            images, img_pairs = generate_pairs(path, samples)
            print(images.keys(), img_pairs)
            # exit()

        print(len(samples))
        TRAIN_SIZE = list(images[samples[0]].shape[1:])
        print(f"TRAIN  SIZE: {TRAIN_SIZE}")
        with torch.no_grad():
            visualize_flow(viz_root_dir, model, img_pairs, images, TRAIN_SIZE)
