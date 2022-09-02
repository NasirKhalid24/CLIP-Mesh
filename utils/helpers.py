"""
    Various helper functions

    create_scene -> combines multiple nvdiffmodeling meshes in to a single mesh with mega texture
"""
import sys
import torch

from math import ceil

sys.path.append("../nvdiffmodeling") 

import nvdiffmodeling.src.mesh as mesh
import nvdiffmodeling.src.texture as texture

cosine_sim = torch.nn.CosineSimilarity()

def cosine_sum(features, targets):
    return -cosine_sim(features, targets).sum()

def cosine_avg(features, targets):
    return -cosine_sim(features, targets).mean()
    
def _merge_attr_idx(a, b, a_idx, b_idx, scale_a=1.0, scale_b=1.0, add_a=0.0, add_b=0.0):
    if a is None and b is None:
        return None, None
    elif a is not None and b is None:
        return (a*scale_a)+add_a, a_idx
    elif a is None and b is not None:
        return (b*scale_b)+add_b, b_idx
    else:
        return torch.cat(((a*scale_a)+add_a, (b*scale_b)+add_b), dim=0), torch.cat((a_idx, b_idx + a.shape[0]), dim=0)

def create_scene(meshes, sz=1024):
    
    # Need to comment and fix code
    
    scene = mesh.Mesh()

    tot = len(meshes) if len(meshes) % 2 == 0 else len(meshes)+1

    nx = 2
    ny = ceil(tot / 2) if ceil(tot / 2) % 2 == 0 else ceil(tot / 2) + 1

    w = int(sz*ny)
    h = int(sz*nx)

    dev = meshes[0].v_tex.device

    kd_atlas = torch.ones ( (1, w, h, 4) ).to(dev)
    ks_atlas = torch.zeros( (1, w, h, 3) ).to(dev)
    kn_atlas = torch.ones ( (1, w, h, 3) ).to(dev)

    for i, m in enumerate(meshes):
        v_pos, t_pos_idx = _merge_attr_idx(scene.v_pos, m.v_pos, scene.t_pos_idx, m.t_pos_idx)
        v_nrm, t_nrm_idx = _merge_attr_idx(scene.v_nrm, m.v_nrm, scene.t_nrm_idx, m.t_nrm_idx)
        v_tng, t_tng_idx = _merge_attr_idx(scene.v_tng, m.v_tng, scene.t_tng_idx, m.t_tng_idx)

        pos_x = i % nx
        pos_y = int(i / ny)

        sc_x = 1./nx
        sc_y = 1./ny

        v_tex, t_tex_idx = _merge_attr_idx(
            scene.v_tex,
            m.v_tex,
            scene.t_tex_idx,
            m.t_tex_idx,
            scale_a=1.,
            scale_b=torch.tensor([sc_x, sc_y]).to(dev),
            add_a=0.,
            add_b=torch.tensor([sc_x*pos_x, sc_y*pos_y]).to(dev)
        )

        kd_atlas[:, pos_y*sz:(pos_y*sz)+sz, pos_x*sz:(pos_x*sz)+sz, :m.material['kd'].data.shape[-1]] = m.material['kd'].data
        ks_atlas[:, pos_y*sz:(pos_y*sz)+sz, pos_x*sz:(pos_x*sz)+sz, :m.material['ks'].data.shape[-1]] = m.material['ks'].data
        kn_atlas[:, pos_y*sz:(pos_y*sz)+sz, pos_x*sz:(pos_x*sz)+sz, :m.material['normal'].data.shape[-1]] = m.material['normal'].data

        scene = mesh.Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            v_nrm=v_nrm,
            t_nrm_idx=t_nrm_idx,
            v_tng=v_tng,
            t_tng_idx=t_tng_idx,
            v_tex=v_tex,
            t_tex_idx=t_tex_idx,
            base=scene 
        )

    scene = mesh.Mesh(
        material={
            'bsdf': 'diffuse',
            'kd': texture.Texture2D(
                kd_atlas
            ),
            'ks': texture.Texture2D(
                ks_atlas
            ),
            'normal': texture.Texture2D(
                kn_atlas
            ),
        },
        base=scene # gets uvs etc from here
    )

    return scene