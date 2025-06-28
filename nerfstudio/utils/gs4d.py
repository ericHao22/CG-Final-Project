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

def rotornorm(rotors1, rotors2, normalize_pesudo=False):

    rotors = torch.cat([rotors1, rotors2], dim=-1)

    if normalize_pesudo:

        a, bxy, bxz, byz, bxw, byw, bzw, pxyzw = torch.split(rotors, 1, dim=-1)

        eps = pxyzw * a - bxy * bzw + bxz * byw - bxw * byz

        mask = eps.abs()[:, 0] > 1e-7

        if mask.sum():

            rotors_pick = rotors[mask]
            eps = eps[mask]
            a, bxy, bxz, byz, bxw, byw, bzw, pxyzw = torch.split(rotors_pick, 1, dim=-1)

            # float l2 = a * a + bxy * bxy + bxz * bxz + byz * byz + bxw * bxw + byw * byw + bzw * bzw + pxyzw * pxyzw;
            l2 = (rotors_pick ** 2).sum(dim=-1, keepdim=True)
            delta = (torch.sqrt(l2 * l2 - 4 * eps * eps) - l2) / (2 * eps)

            da = +delta * pxyzw
            dpxyzw = +delta * a
            dbxy = -delta * bzw
            dbzw = -delta * bxy
            dbxz = +delta * byw
            dbyw = +delta * bxz
            dbyz = -delta * bxw
            dbxw = -delta * byz

            anew = a + da
            pxyzwnew = pxyzw + dpxyzw
            bxywnew = bxy + dbxy
            bxzwnew = bxz + dbxz
            byzwnew = byz + dbyz
            bxwwnew = bxw + dbxw
            bywwnew = byw + dbyw
            bzwwnew = bzw + dbzw

            rotors_new = torch.cat([anew, bxywnew, bxzwnew, byzwnew, bxwwnew, bywwnew,bzwwnew, pxyzwnew], dim=-1)
            # rotors[mask] = rotors_new
            # cannot be inplace operation
            rotors_new_full = torch.zeros_like(rotors)
            rotors_new_full[mask] = rotors_new
            rotors = torch.where(mask.reshape(-1, 1).expand(-1, 8), rotors_new_full, rotors)
    
    length = rotors.norm(dim=-1, keepdim=True)
    rotors = rotors / (1e-7 + length)

    return rotors[:, :4], rotors[:, 4:]

def build_rotation_from_rotor8(rotors1, rotors2):
    a = rotors1[:,0]
    s = a
    bxy = rotors1[:,1]
    bxz = rotors1[:,2]
    byz = rotors1[:,3]

    bxw = rotors2[:,0]
    byw = rotors2[:,1]
    bzw = rotors2[:,2]
    pxyzw = rotors2[:,3]

    N = a.shape[0]
    r = torch.zeros(N, 4, 4, device=a.device)

    s2 = a * a
    bxy2 = bxy * bxy
    bxz2 = bxz * bxz
    bxw2 = bxw * bxw
    byz2 = byz * byz
    byw2 = byw * byw
    bzw2 = bzw * bzw
    bxyzw2 = pxyzw * pxyzw

    r[:, 0, 0] = -bxy2 - bxz2 - bxw2 + byz2 + byw2 + bzw2 - bxyzw2 + s2
    r[:, 1, 0] = 2 * (bxy * s - bxz * byz - bxw * byw + bzw * pxyzw)
    r[:, 2, 0] = 2 * (bxy * byz + bxz * s - bxw * bzw - byw * pxyzw)
    r[:, 3, 0] = 2 * (bxy * byw + bxz * bzw + bxw * s + byz * pxyzw)

    r[:, 0, 1] = -2 * (bxy * s + bxz * byz + bxw * byw + bzw * pxyzw)
    r[:, 1, 1] = -bxy2 + bxz2 + bxw2 - byz2 - byw2 + bzw2 - bxyzw2 + s2
    r[:, 2, 1] = 2 * (-bxy * bxz + bxw * pxyzw + byz * s - byw * bzw)
    r[:, 3, 1] = 2 * (-bxy * bxw - bxz * pxyzw + byz * bzw + byw * s)

    r[:, 0, 2] = 2 * (bxy * byz - bxz * s - bxw * bzw + byw * pxyzw)
    r[:, 1, 2] = -2 * (bxy * bxz + bxw * pxyzw + byw * bzw + byz * s)
    r[:, 2, 2] = bxy2 - bxz2 + bxw2 - byz2 + byw2 - bzw2 - bxyzw2 + s2
    r[:, 3, 2] = 2 * (bxy * pxyzw - bxz * bxw - byw * byz + bzw * s)

    r[:, 0, 3] = 2 * (bxy * byw + bxz * bzw - bxw * s - byz * pxyzw)
    r[:, 1, 3] = 2 * (-bxy * bxw + bxz * pxyzw + byz * bzw - byw * s)
    r[:, 2, 3] = -2 * (bxy * pxyzw + bxz * bxw + byz * byw + bzw * s)
    r[:, 3, 3] = bxy2 + bxz2 - bxw2 + byz2 - byw2 - bzw2 - bxyzw2 + s2

    return r.permute(0, 2, 1)

def build_scaling_rotation_4d(s, r1, r2):
    L = torch.zeros((s.shape[0], 4, 4), dtype=torch.float, device=s.device)
    R = build_rotation_from_rotor8(r1, r2)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L[:, 3, 3] = s[:, 3]

    L = R @ L
    return L

def slice_4d(scale, r1, r2):
    L = build_scaling_rotation_4d(scale, r1, r2)
    sigma = L @ L.permute(0, 2, 1)
    
    w = 1 / sigma[:, 3,3]
    alpha = sigma[:, 0, 3] * w
    beta = sigma[:, 1,3] * w
    gamma = sigma[:, 2, 3] * w


    cov_3d_out0 = sigma[:, 0, 0] - sigma[:, 0, 3] * alpha
    cov_3d_out1 = sigma[:, 0, 1] - sigma[:, 0, 3] * beta
    cov_3d_out2 = sigma[:, 0, 2] - sigma[:, 0, 3] * gamma
    cov_3d_out3 = sigma[:, 1, 1] - sigma[:, 1, 3] * beta
    cov_3d_out4 = sigma[:, 1, 2] - sigma[:, 1, 3] * gamma
    cov_3d_out5 = sigma[:, 2, 2] - sigma[:, 2, 3] * gamma
    cov_3d_out = torch.stack([cov_3d_out0, cov_3d_out1, cov_3d_out2, cov_3d_out3, cov_3d_out4, cov_3d_out5], dim=-1)
    
    speed = torch.stack([alpha, beta, gamma], dim=-1)
    
    return cov_3d_out, speed, w