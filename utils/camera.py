import glm
import torch
import random

import numpy as np
import torchvision.transforms as transforms

from .resize_right import resize

blurs = [
    transforms.Compose([
        transforms.GaussianBlur(11, sigma=(5, 5))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(11, sigma=(2, 2))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(5, sigma=(5, 5))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(5, sigma=(2, 2))
    ]),
]

def get_random_bg(h, w):

        p = torch.rand(1)

        if p > 0.66666:
            background =  blurs[random.randint(0, 3)]( torch.rand((1, 3, h, w)) ).permute(0, 2, 3, 1)
        elif p > 0.333333:
            size = random.randint(5, 10)
            background = torch.vstack([
                torch.full( (1, size, size), torch.rand(1).item() / 2),
                torch.full( (1, size, size), torch.rand(1).item() / 2 ),
                torch.full( (1, size, size), torch.rand(1).item() / 2 ),
            ]).unsqueeze(0)

            second = torch.rand(3)

            background[:, 0, ::2, ::2] = second[0]
            background[:, 1, ::2, ::2] = second[1]
            background[:, 2, ::2, ::2] = second[2]

            background[:, 0, 1::2, 1::2] = second[0]
            background[:, 1, 1::2, 1::2] = second[1]
            background[:, 2, 1::2, 1::2] = second[2]

            background = blurs[random.randint(0, 3)]( resize(background, out_shape=(h, w)) )

            background = background.permute(0, 2, 3, 1)

        else:
            background = torch.vstack([
                torch.full( (1, h, w), torch.rand(1).item()),
                torch.full( (1, h, w), torch.rand(1).item()),
                torch.full( (1, h, w), torch.rand(1).item()),
            ]).unsqueeze(0).permute(0, 2, 3, 1)

        return background

def cosine_sample(N : np.ndarray) -> np.ndarray:
    """
    #----------------------------------------------------------------------------
    # Cosine sample around a vector N
    #----------------------------------------------------------------------------

    Copied from nvdiffmodelling

    """
    # construct local frame
    N = N/np.linalg.norm(N)

    dx0 = np.array([0, N[2], -N[1]])
    dx1 = np.array([-N[2], 0, N[0]])

    dx = dx0 if np.dot(dx0,dx0) > np.dot(dx1,dx1) else dx1
    dx = dx/np.linalg.norm(dx)
    dy = np.cross(N,dx)
    dy = dy/np.linalg.norm(dy)

    # cosine sampling in local frame
    phi = 2.0*np.pi*np.random.uniform()
    s = np.random.uniform()
    costheta = np.sqrt(s)
    sintheta = np.sqrt(1.0 - s)

    # cartesian vector in local space
    x = np.cos(phi)*sintheta
    y = np.sin(phi)*sintheta
    z = costheta

    # local to world
    return dx*x + dy*y + N*z

def persp_proj(fov_x=45, ar=1, near=1.0, far=50.0):
    """
    From https://github.com/rgl-epfl/large-steps-pytorch by @bathal1 (Baptiste Nicolet)

    Build a perspective projection matrix.
    Parameters
    ----------
    fov_x : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_rad = np.deg2rad(fov_x)

    tanhalffov = np.tan( (fov_rad / 2) )
    max_y = tanhalffov * near
    min_y = -max_y
    max_x = max_y * ar
    min_x = -max_x

    z_sign = -1.0
    proj_mat = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

    proj_mat[0, 0] = 2.0 * near / (max_x - min_x)
    proj_mat[1, 1] = 2.0 * near / (max_y - min_y)
    proj_mat[0, 2] = (max_x + min_x) / (max_x - min_x)
    proj_mat[1, 2] = (max_y + min_y) / (max_y - min_y)
    proj_mat[3, 2] = z_sign

    proj_mat[2, 2] = z_sign * far / (far - near)
    proj_mat[2, 3] = -(far * near) / (far - near)
    
    return proj_mat

def get_camera_params(elev_angle, azim_angle, distance, resolution, fov=60, look_at=[0, 0, 0], up=[0, -1, 0]):
    
    elev = np.radians( elev_angle )
    azim = np.radians( azim_angle ) 
    
    # Generate random view
    cam_z = distance * np.cos(elev) * np.sin(azim)
    cam_y = distance * np.sin(elev)
    cam_x = distance * np.cos(elev) * np.cos(azim)

    modl = glm.mat4()
    view  = glm.lookAt(
        glm.vec3(cam_x, cam_y, cam_z),
        glm.vec3(look_at[0], look_at[1], look_at[2]),
        glm.vec3(up[0], up[1], up[2]),
    )

    a_mv = view * modl
    a_mv = np.array(a_mv.to_list()).T
    proj_mtx = persp_proj(fov)
    
    a_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]
    
    a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
    a_campos = a_lightpos

    return {
        'mvp' : a_mvp,
        'lightpos' : a_lightpos,
        'campos' : a_campos,
        'resolution' : [resolution, resolution], 
        }

# Returns a batch of camera parameters
class CameraBatch(torch.utils.data.Dataset):
    def __init__(
        self,
        image_resolution,
        distances,
        azimuths,
        elevation_params,
        fovs,
        aug_loc, 
        aug_light,
        aug_bkg,
        bs,
        look_at=[0, 0, 0], up=[0, -1, 0]
    ):

        self.res = image_resolution

        self.dist_min = distances[0]
        self.dist_max = distances[1]

        self.azim_min = azimuths[0]
        self.azim_max = azimuths[1]

        self.fov_min = fovs[0]
        self.fov_max = fovs[1]
        
        self.elev_alpha = elevation_params[0]
        self.elev_beta  = elevation_params[1]
        self.elev_max   = elevation_params[2]

        self.aug_loc   = aug_loc
        self.aug_light = aug_light
        self.aug_bkg   = aug_bkg

        self.look_at = look_at
        self.up = up

        self.batch_size = bs

    def __len__(self):
        return self.batch_size
        
    def __getitem__(self, index):

        elev = np.radians( np.random.beta( self.elev_alpha, self.elev_beta ) * self.elev_max )
        azim = np.radians( np.random.uniform( self.azim_min, self.azim_max+1.0 ) )
        dist = np.random.uniform( self.dist_min, self.dist_max )
        fov = np.random.uniform( self.fov_min, self.fov_max )
        
        proj_mtx = persp_proj(fov)
        
        # Generate random view
        cam_z = dist * np.cos(elev) * np.sin(azim)
        cam_y = dist * np.sin(elev)
        cam_x = dist * np.cos(elev) * np.cos(azim)
        
        if self.aug_loc:

            # Random offset
            limit  = self.dist_min // 2
            rand_x = np.random.uniform( -limit, limit )
            rand_y = np.random.uniform( -limit, limit )

            modl = glm.translate(glm.mat4(), glm.vec3(rand_x, rand_y, 0))

        else:
        
            modl = glm.mat4()
            
        view  = glm.lookAt(
            glm.vec3(cam_x, cam_y, cam_z),
            glm.vec3(self.look_at[0], self.look_at[1], self.look_at[2]),
            glm.vec3(self.up[0], self.up[1], self.up[2]),
        )

        r_mv = view * modl
        r_mv = np.array(r_mv.to_list()).T

        mvp     = np.matmul(proj_mtx, r_mv).astype(np.float32)
        campos  = np.linalg.inv(r_mv)[:3, 3]

        if self.aug_light:
            lightpos = cosine_sample(campos)*dist
        else:
            lightpos = campos*dist

        if self.aug_bkg:
            bkgs = get_random_bg(self.res, self.res).squeeze(0)
        else:
            bkgs = torch.ones(self.res, self.res, 3)

        return {
            'mvp': torch.from_numpy( mvp ).float(),
            'lightpos': torch.from_numpy( lightpos ).float(),
            'campos': torch.from_numpy( campos ).float(),
            'bkgs': bkgs
        }