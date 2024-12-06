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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import imageio
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from matplotlib import cm
import numpy as np
from kornia.geometry.depth import depth_to_normals

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)

    writer = imageio.get_writer(os.path.join(render_path, 'video.mp4'), fps=30)
    writerD = imageio.get_writer(os.path.join(render_path, 'videoD.mp4'), fps=30)
    writerN = imageio.get_writer(os.path.join(render_path, 'videoN.mp4'), fps=30)

    K = torch.eye(3)[None].cuda()
    viewpoint_camera = views[0]
    K[0, 0, 0] = .5 * viewpoint_camera.image_width / np.tan(.5 * float(viewpoint_camera.FoVx))
    K[0, 1, 1] = .5 * viewpoint_camera.image_height / np.tan(.5 * float(viewpoint_camera.FoVy))
    K[0, 0, 2] = .5 * viewpoint_camera.image_width
    K[0, 1, 2] = .5 * viewpoint_camera.image_height

    pool_op = torch.nn.AvgPool2d(3).cuda()

    min_cl, dep_pw = 1.4, 0.4
    depths = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refactor(3)
        outputs = render(view, gaussians, pipeline, background)
        alpha = outputs["alpha"] + 1e-6
        outputs["render"] = pool_op(outputs["render"])
        outputs["depth"] = pool_op(outputs["depth"])
        alpha = pool_op(alpha)
        outputs["depth"] = (outputs["depth"]/alpha)[0]
        rendering = outputs["render"]/alpha
        rendering = torch.clamp(rendering, 0, 1)

        normal = (depth_to_normals(outputs["depth"][None, None], K)[0].permute(1, 2, 0) + 1) / 2

        dep = outputs['depth']
        if idx == 0:
            dep_min = dep.min()
            dep_max = dep.max()
        else:
            dep_min = min(dep_min, dep.min())
            dep_max = max(dep_max, dep.max())

        depths.append(dep)
        writerN.append_data((normal.cpu().numpy() * 255).astype('uint8'))
        writer.append_data((rendering.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))

    for dep in depths:

        dep = ((dep - dep_min) / (dep_max - dep_min)) ** dep_pw
        dep = cm.get_cmap('turbo')(1-dep.detach().cpu().numpy())[..., :3]

        writerD.append_data((dep * 255).astype('uint8'))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.num_cameras, dataset.radius_mult)
        gaussians.test = True
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "render", scene.loaded_iter, scene.getRenderCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
