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
import torch
from random import randint, choice, shuffle
import sys
import matplotlib.cm as cm
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from datetime import datetime
import json
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from backwarp import backWarp
from utils.loss_utils import l1_loss, ssim, VGGLoss, ML1Loss
from gaussian_renderer import render


def backupConfigAndCode(runpath, args):

    model_path = os.path.join(runpath, "configncode")
    os.makedirs(model_path, exist_ok = True)
    now = datetime.now()
    t = now.strftime("_%Y_%m_%d_%H:%M:%S")
    with open(model_path + "/args.json", 'w') as out:
        json.dump(vars(args), out, indent=2, sort_keys=True)

    os.system("cp --parents `find -name \*.py -not -path '*/output/*'` {} 2>/dev/null".format(model_path))
    os.system('cp -r "{}" "{}"'.format(model_path, model_path+t))

def prepare_output_and_logger(args, full_args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])


    import shutil
    if os.path.exists(args.model_path):
        shutil.rmtree(args.model_path)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    backupConfigAndCode(args.model_path, full_args)

def tv(inp):

    dx  = inp[:, :, :-1] - inp[:, :, 1:]
    dy  = inp[:, :-1, :] - inp[:, 1:, :]
    loss_smooth_ren = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

    return loss_smooth_ren

def second_order_tv(ren_disparity, depth_mask_seg, gamma, num_mask_channels):
    temp = depth_mask_seg * ren_disparity.repeat(num_mask_channels, 1, 1).contiguous()

    loss_smooth_ren = (gamma) * tv(temp) + (1 - gamma) * tv(ren_disparity)

    return loss_smooth_ren


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, full_args):

    def warping_loss(xyz):

        shuffle(fr_indices)

        idx1 = fr_indices[0]
        xyz = xyz.reshape(-1, H, W, 3).permute(0, 3, 1, 2)
        xyz0 = xyz[idx1:idx1+1]

        loss = 0
        threshold = 0.5

        for idx2 in fr_indices[1:]:

            flow01 = gaussians.flows[f"{idx1}_{idx2}"].permute(2, 0, 1)[None]
            mask01 = gaussians.masks[f"{idx1}_{idx2}"][None, None]

            xyz1 = xyz[idx2:idx2+1]
            warped01 = warp(xyz1, flow01)

            mask01[mask01 <= threshold] = 0
            mask01[mask01 > threshold] = 1

            loss += flow_loss(warped01 * mask01, xyz0 * mask01)

        return loss

    def update_and_save(ema_loss_for_log, loss):

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")



    first_iter = 6000
    prepare_output_and_logger(dataset, full_args)
    gaussians = GaussianModel(dataset.sh_degree, dataset.num_cameras, dataset.radius_mult)
    scene = Scene(dataset, gaussians)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda").contiguous()

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    # Pick a random Camera
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()

    viewpoint_cam = viewpoint_stack[0]
    C, H, W = viewpoint_cam.original_image.shape
    warp = backWarp(W, H, "cuda")
    gaussians.set_ref_camera(viewpoint_cam, outChannels=dataset.num_mask_channels)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    flow_loss = torch.nn.L1Loss().to(device='cuda')

    switch_iter = 7001
    gaussians.set_scheduler(switch_iter - first_iter)

    fr_indices = [0, 1, 2, 3][:gaussians.num_cameras]


    # Coarse monocular depth alignment using optical flow with scale-offset [1000 iterations]
    for iteration in range(first_iter, switch_iter):

        loss = warping_loss(xyz=gaussians.get_xyz['xyz'])
        loss.backward()

        # Update and save
        update_and_save(ema_loss_for_log, loss)

        # Optimizer step
        gaussians.monoso_optimizer.step()
        gaussians.monoso_optimizer.zero_grad()
        gaussians.monoso_scheduler.step()

    # Reinitalize depth based on scale and offset. Set gaussian radius (scaling)
    gaussians.update_initz_so()
    gaussians.set_scaling()

    # Enable decoders and optimizers
    gaussians.enable_decoder = True
    gaussians.enable_learned_opacity = True
    gaussians.training_setup(opt)

    # Training [13000 iterations]

    smooth_coeff = 5
    warp_coeff = 0.1

    for iteration in range(switch_iter, opt.iterations + 1):

        # Every 1000 its we increase the levels of SH up to a maximum degree after trn_it_lst[1]=15000 iterations
        if iteration in [trn_it_lst[1], trn_it_lst[1] + 1000, trn_it_lst[1] + 2000]:
        # if iteration in trn_it_lst[1:]:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        cam_id = viewpoint_cam.uid

        # Every 100 iterations we reset the gaussian radius until trn_it_lst[1]=15000 iterations
        if iteration < trn_it_lst[1] and iteration % 100 == 0:
            gaussians.set_scaling(use_decoder=True)

        # Enable multisampling near the end of training
        if iteration > (trn_it_lst[0] + 2000):
            num_samp = randint(1, 4)
            pool_op = torch.nn.AvgPool2d(num_samp).cuda()
            viewpoint_cam.refactor(num_samp)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, ren_depth, alpha = render_pkg["render"], render_pkg["depth"], render_pkg['alpha']

        # Enable multisampling near the end of training
        if iteration > (trn_it_lst[0] + 2000):
            image = pool_op(image)
            ren_depth = pool_op(ren_depth)
            alpha = pool_op(alpha)

        # Loss
        gt_image = viewpoint_cam.original_image
        ren_depth = ren_depth / (alpha + 1e-6)

        # RGB reconstruction loss
        Ll1 = ml1_loss(image, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))) + 0.05 * vgg_loss(image, gt_image)

        # TV regularization
        ren_disp = 1 / (ren_depth + 1)
        loss_smooth_ren = second_order_tv(ren_disp, gaussians.dep_mask[cam_id], ((iteration) / (opt.iterations)), dataset.num_mask_channels)
        loss += smooth_coeff * loss_smooth_ren

        # Flow regularization
        wrploss = warping_loss(render_pkg["xyz"]) * warp_coeff
        loss += warp_coeff * wrploss

        loss.backward()

        # Update and save
        update_and_save(ema_loss_for_log, loss)

        # Optimizer step
        if iteration < opt.iterations:

            gaussians.decoder_opac_optimizer.step()
            gaussians.decoder_opac_optimizer.zero_grad()
            gaussians.decoder_opac_scheduler.step()

            gaussians.decoder_optimizer.step()
            gaussians.decoder_optimizer.zero_grad()
            gaussians.decoder_scheduler.step()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)


if __name__ == "__main__":
    trn_it_lst = [10000, 15000, 20000]#, 44000]
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 1000, 7_000, 10_000, 15_00, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[6_001, 7_000, 10_000, 15_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    args.test_iterations = args.test_iterations + trn_it_lst
    args.save_iterations = args.save_iterations + trn_it_lst

    print(args)

    print("Optimizing " + args.model_path)

    ml1_loss = ML1Loss()
    vgg_loss = VGGLoss()

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
