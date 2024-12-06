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
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.decoder import ImplicitDecoder
from scene.posenc import get_embedder

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, num_cameras : int, radius_mult : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.num_cameras = num_cameras
        self.radius_mult = radius_mult

        # Scale offset paramteres for monocular depth
        self.monodepth_scaling = torch.nn.Parameter(torch.ones(self.num_cameras).cuda().requires_grad_(True))
        self.monodepth_offset = torch.nn.Parameter(torch.zeros(self.num_cameras).cuda().requires_grad_(True))
        self.monoso_optimizer = torch.optim.Adam([{'params': [self.monodepth_scaling], 'lr': 1}, {'params': [self.monodepth_offset], 'lr': 5}])

        # Fixed decoder inputs for all cameras
        embed_fn = get_embedder(10)
        inp = np.linspace(0, 1, num=self.num_cameras, dtype=np.float32)
        inp = embed_fn(inp)
        self.inp = torch.tensor(inp)[..., None, None].cuda()


        self.enable_decoder = False
        self.mono_depth_scaleoffset_enable = True
        self.test = False
        self.enable_learned_opacity = False


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    # Setup implicit decoders for xyz and opacity
    def set_ref_camera(self, ref_camera, outChannels=5):

        xyz_dec_capacity  = {2:10, 3:15, 4:18}
        opac_dec_capacity = {2:6, 3:10, 4:12}

        _, self.height, self.width = ref_camera.original_image.shape
        self.ref_camera = ref_camera

        self.decoder = ImplicitDecoder(self.height, self.width, ngf=xyz_dec_capacity[self.num_cameras], inChannels=20, outChannels=outChannels).cuda()
        self.decoder_scaling = torch.nn.Parameter(torch.ones(self.num_cameras, outChannels, 1, 1).cuda().requires_grad_(True))
        self.decoder_offset = torch.nn.Parameter(torch.zeros(self.num_cameras, outChannels, 1, 1).cuda().requires_grad_(True))
        self.decoder_optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=1e-6)
        self.decoder_so_optimizer = torch.optim.Adam([{'params': [self.decoder_scaling], 'lr': 0.1}, {'params': [self.decoder_offset], 'lr': 0.5}])
        self.decoder_scheduler = torch.optim.lr_scheduler.CyclicLR(self.decoder_optimizer, base_lr=1e-6, max_lr=1e-4, mode='triangular2', step_size_up=1000, cycle_momentum=False)

        self.decoder_opac = ImplicitDecoder(self.height, self.width, ngf=opac_dec_capacity[self.num_cameras], inChannels=20, outChannels=outChannels).cuda()
        self.decoder_opac_optimizer = torch.optim.AdamW(self.decoder_opac.parameters(), lr=1e-6)
        self.decoder_opac_scheduler = torch.optim.lr_scheduler.CyclicLR(self.decoder_opac_optimizer, base_lr=1e-6, max_lr=1e-5, mode='triangular2', step_size_up=2000, cycle_momentum=False)

    def set_scheduler(self, end_iter):
        self.monoso_scheduler = CosineAnnealingLR(self.monoso_optimizer, T_max=end_iter)

    # Reset gaussian shape using depth
    def set_scaling(self, use_decoder=False, radius_mult=None):

        if radius_mult is None:
            radius_mult = self.radius_mult

        with torch.no_grad():
            depth = self._z

            if use_decoder:
                depth = depth + self.get_residual()

            radii = np.tan(0.5 * float(self.ref_camera.FoVy))  * depth / self.height
            radii2 = radii**2

            scales = torch.log(torch.sqrt(radii2) * radius_mult).repeat(1, 3) #/ 1.1
            self._scaling.data = scales.contiguous()

    # Set z after scale offset optimization
    def update_initz_so(self):
        self.mono_depth_scaleoffset_enable = False

        with torch.no_grad():
            scaling = self.monodepth_scaling.repeat_interleave(self.height*self.width)[..., None]
            offset = self.monodepth_offset.repeat_interleave(self.height*self.width)[..., None]

            tempp = self._z * scaling + offset
            # del self.monodepth_scaling, self.monodepth_offset, self.monoso_optimizer
            print("Init z updated!!!!")

            self._z = tempp.contiguous()

    def get_learned_opac(self):

        opac = F.tanh(self.decoder_opac(self.inp))
        opac = opac * self.dep_mask / (torch.sum(self.dep_mask, dim=1, keepdim=True) + 1e-10)
        opac = torch.sum(opac, dim=1, keepdim=True)

        opac = 100 * opac.reshape(-1, 1)

        return opac

    def get_residual(self):

        delta = F.tanh(self.decoder(self.inp))
        delta = delta * self.dep_mask / (torch.sum(self.dep_mask, dim=1, keepdim=True) + 1e-10)
        delta = torch.sum(delta, dim=1, keepdim=True)

        delta = 50 * delta.reshape(-1, 1)

        return delta

    @property
    def get_xyz(self):
        if self.test:
            return {'xyz': self._xyz, 'depth': None}

        if self.mono_depth_scaleoffset_enable:
            scaling = self.monodepth_scaling.repeat_interleave(self.height*self.width)[..., None]
            offset = self.monodepth_offset.repeat_interleave(self.height*self.width)[..., None]

            z = self._z * scaling + offset

        elif self.enable_decoder:
            z = self._z + self.get_residual()

        else:
            z = self._z

        # Convert depth to xyz world
        xyz_homo = torch.cat((self._xy * z, z, torch.ones_like(z)), dim=1)[..., None]
        xyz_world_homo = torch.bmm(self.c2w, xyz_homo)
        xyz_world = (xyz_world_homo[:, :3] / xyz_world_homo[:, 3:])[..., 0]

        return {'xyz': xyz_world, 'depth': z}

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        if self.enable_learned_opacity:
            opacity = self.get_learned_opac() + self._opacity
            ret_opac = self.opacity_activation(opacity)

            return ret_opac
        else:
            return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def angle(self, u, v):
        cos_sim = torch.nn.CosineSimilarity()
        return torch.arccos(cos_sim(u,v))

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, train_cameras : list, flows, masks, dep_mask):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        # c2ws for z to xyz transformation
        w2cs = torch.stack([x.world_view_transform.transpose(0, 1) for x in train_cameras], dim=0)
        c2w = w2cs.inverse().clone().repeat_interleave(fused_point_cloud.shape[0]//self.num_cameras, dim=0)


        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.tensor(np.asarray(pcd.radii2.copy())).float().cuda()[:, 0]
        scales = torch.log(torch.sqrt(dist2) * self.radius_mult)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        init_opac = {2:0.6, 3:0.5, 4:0.35}
        opacities = inverse_sigmoid(init_opac[self.num_cameras] * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xy = (fused_point_cloud[:, :2] / fused_point_cloud[:, 2:]).contiguous()
        self._z = fused_point_cloud[:, 2:].contiguous()
        self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        self._scaling = scales.contiguous()
        self._rotation = rots.contiguous()
        self._opacity = opacities.contiguous()
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.c2w = c2w.contiguous()
        self.flows = flows
        self.masks = masks
        self.dep_mask = dep_mask.contiguous()


    def training_setup(self, training_args):
        self._features_rest = nn.Parameter(self._features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(self._scaling.requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation.requires_grad_(True))

        l = [
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr/10, "name": "f_rest"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz['xyz'].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(self.get_opacity)
        opacities = opacities.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree