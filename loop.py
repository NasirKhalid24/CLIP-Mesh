# Main optimization loop, takes in dictionary config
# and performs optimization as highlighted in paper

import os
import clip
import yaml
import torch
import kornia
import torchvision

import numpy                as np
import nvdiffrast.torch     as dr
import matplotlib.pyplot    as plt

from tqdm                   import tqdm
from datetime               import datetime
from dalle2_pytorch         import DiffusionPrior, DiffusionPriorNetwork

from PIL                    import Image
from utils.video            import Video
from utils.limit_subdivide  import LimitSubdivide
from utils.helpers          import cosine_avg, create_scene
from utils.camera           import CameraBatch, get_camera_params
from utils.resize_right     import resize, cubic, linear, lanczos2, lanczos3

from nvdiffmodeling.src     import obj
from nvdiffmodeling.src     import util
from nvdiffmodeling.src     import mesh
from nvdiffmodeling.src     import render
from nvdiffmodeling.src     import texture
from nvdiffmodeling.src     import regularizer

def loop(cfg):

    # Set unique output path
    now = datetime.now()
    cfg["path"] = os.path.join(
        cfg["output_path"],
        now.strftime("%m-%d-%Y_%H-%M-%S") + cfg["text_prompt"]
    )
    
    cfg['path'] = cfg['path'].replace(" ", "_")
    os.makedirs(cfg['path'])
    
    with open(os.path.join(cfg["path"], "config.yml"), 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    print("Result directory '%s' created" % cfg["path"])
    
    # Get CUDA device
    device = torch.device("cuda:" + cfg["gpu"])
    torch.cuda.set_device(device)

    # Initialize CLIP model
    model, _ = clip.load(cfg["clip_model"], device=device)

    clip_mean = torch.tensor([0.48154660, 0.45782750, 0.40821073], device=device)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    # Initialize Video
    video = Video(cfg["path"])

    # Intialize GL Context
    glctx = dr.RasterizeGLContext()

    # Get text embedding
    print("Text is %s" % cfg["text_prompt"])

    texts_embeds = clip.tokenize([cfg["text_prompt"]]).to(device)
    with torch.no_grad():
        texts_embeds = model.encode_text(texts_embeds).detach()
        texts_embeds = texts_embeds / texts_embeds.norm(dim=1, keepdim=True)

    # Setup Prior model & get image prior (text embed -> image embed)
    if cfg["prior_path"] is not None:

        state_dict = torch.load(cfg["prior_path"], map_location=device)["model"]

        prior_network = DiffusionPriorNetwork( 
            dim=cfg["diffusion_prior_network_dim"],
            depth=cfg["diffusion_prior_network_depth"], 
            dim_head=cfg["diffusion_prior_network_dim_head"], 
            heads=cfg["diffusion_prior_network_heads"],
            normformer=cfg["diffusion_prior_network_normformer"]
        ).to(device)

        diffusion_prior = DiffusionPrior( 
            net=prior_network,
            clip=None,
            image_embed_dim=cfg["diffusion_prior_embed_dim"], 
            timesteps=cfg["diffusion_prior_timesteps"],
            cond_drop_prob=cfg["diffusion_prior_cond_drop_prob"], 
            loss_type=cfg["diffusion_prior_loss_type"], 
            condition_on_text_encodings=cfg["diffusion_prior_condition_on_text_encodings"]
        ).to(device)

        diffusion_prior.load_state_dict(state_dict, strict=True)

        text_cond = dict(text_embed = texts_embeds)
        prior_embeds = diffusion_prior.p_sample_loop((1, 512), text_cond = text_cond)

        prior_embeds = prior_embeds.detach().clone().to(device)

        del prior_network, diffusion_prior, state_dict
        torch.cuda.empty_cache()

    # Load all meshes and setup training parameters
    meshes = [] # store Mesh objects
    subdiv = [] # store per mesh limit subdivison
    train_params = [] # store all trainable paramters
    vert_train = False

    for idx, m in enumerate(cfg["meshes"]): # Loop over each mesh path

        load_mesh = obj.load_obj(m)

        if cfg["unit"][idx]: # If mesh is to be unit sized
            load_mesh = mesh.unit_size(load_mesh)

        # Scale vertices by factors provided and then offset by offsets provided
        v_pos = torch.tensor(cfg["scales"][idx]).to(load_mesh.v_pos.device) * load_mesh.v_pos.clone().detach()
        v_pos = torch.tensor(cfg["offsets"][idx]).to(v_pos.device) + v_pos.clone().detach()

        # Final mesh after all adjustments
        load_mesh = mesh.Mesh(v_pos, base=load_mesh)

        # If true is in train_mesh_idx[mesh_idx] then we initialize
        # all textures else we start with textures already on mesh
        if True in cfg["train_mesh_idx"][idx]:

            # vertices 
            vertices = load_mesh.v_pos.clone().detach().requires_grad_(True)

            # faces
            faces = load_mesh.t_pos_idx.clone().detach()

            # texture map
            texture_map = texture.create_trainable(np.random.uniform(size=[cfg["texture_resolution"]]*2 + [cfg["channels"]], low=0.0, high=1.0), [cfg["texture_resolution"]]*2, True)

            # normal map
            normal_map = texture.create_trainable(np.array([0, 0, 1]), [cfg["texture_resolution"]]*2, True)

            # specular map
            specular_map = texture.create_trainable(np.array([0, 0, 0]), [cfg["texture_resolution"]]*2, True)

        else:

            # vertices 
            vertices = load_mesh.v_pos.clone().detach().requires_grad_(True)

            # faces
            faces = load_mesh.t_pos_idx.clone().detach()

            # get existing texture and specular maps
            kd_ = load_mesh.material['kd'].data.permute(0, 3, 1, 2)
            ks_ = load_mesh.material['ks'].data.permute(0, 3, 1, 2)

            # if there is a normal map load it or initial a plain one
            try:
                nrml_ = load_mesh.material['normal'].data.permute(0, 3, 1, 2)
            except:
                nrml_ = torch.zeros( (1, 3, cfg["texture_resolution"], cfg["texture_resolution"]) ).to(device)
                nrml_[:, 2, :, :] = 1.0

            # convert all texture maps to trainable tensors
            texture_map  = texture.create_trainable( resize(kd_, out_shape=(cfg["texture_resolution"], cfg["texture_resolution"])).permute(0, 2, 3, 1), [cfg["texture_resolution"]]*2, True)
            specular_map = texture.create_trainable( resize(ks_, out_shape=(cfg["texture_resolution"], cfg["texture_resolution"])).permute(0, 2, 3, 1), [cfg["texture_resolution"]]*2, True)
            normal_map   = texture.create_trainable( resize(nrml_, out_shape=(cfg["texture_resolution"], cfg["texture_resolution"])).permute(0, 2, 3, 1), [cfg["texture_resolution"]]*2, True)

        # Training parameters
        if "verts" in cfg["train_mesh_idx"][idx]:
            train_params += [vertices]
            vert_train = True
        if "texture" in cfg["train_mesh_idx"][idx]:
            train_params += texture_map.getMips()
        if "normal" in cfg["train_mesh_idx"][idx]:
            train_params += normal_map.getMips()
        if "specular" in cfg["train_mesh_idx"][idx]:
            train_params += specular_map.getMips()
        
        # Create final mesh with all textures
        load_mesh = mesh.Mesh(
            vertices,
            faces,
            material={
                'bsdf': cfg['bsdf'],
                'kd': texture_map,
                'ks': specular_map,
                'normal': normal_map,
            },
            base=load_mesh # Get UVs from original loaded mesh
        )
        meshes.append( load_mesh )

        # Create limit subdivision class for mesh
        if "verts" in cfg["train_mesh_idx"][idx]:
            subdiv.append( LimitSubdivide(
                load_mesh.v_pos.clone().detach(),
                load_mesh.t_pos_idx.clone().detach(),
            ) )
        else:
            subdiv.append( None )

    # Optimizer and Scheduler
    optimizer  = torch.optim.Adam(train_params, lr=cfg["lr"])
    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002))) 
        
    # Dataset to get random camera parameters
    cams_data = CameraBatch(
        cfg["train_res"],
        [cfg["dist_min"], cfg["dist_max"]],
        [cfg["azim_min"], cfg["azim_max"]],
        [cfg["elev_alpha"], cfg["elev_beta"], cfg["elev_max"]],
        [cfg["fov_min"], cfg["fov_max"]],
        cfg["aug_loc"],
        cfg["aug_light"],
        cfg["aug_bkg"],
        cfg["batch_size"]
    )

    cams = torch.utils.data.DataLoader(
        cams_data,
        cfg["batch_size"],
        num_workers=0,
        pin_memory=True
    )

    # Optimization Loop
    rot_ang = 0.0
    t_loop = tqdm(range(cfg["epochs"]), leave=False)

    for it in t_loop:
        
        render_meshes = []          # store meshes with texture that will be rendered
        render_meshes_notex = []    # store meshes without texture that will be rendered

        lapl_funcs    = []          # store laplacian for each mesh

        # For each mesh initialized
        for i, m in enumerate(meshes):
            
            # Limit subdivide vertices if needed
            if subdiv[i] != None:

                n_vert = subdiv[i].get_limit(
                    m.v_pos.to('cpu').double()
                ).to(device)

            else:

                n_vert = m.v_pos

            # Low pass filter for textures
            ready_texture = texture.Texture2D(
                kornia.filters.gaussian_blur2d(
                    m.material['kd'].data.permute(0, 3, 1, 2),
                    kernel_size=(cfg["kernel_size"], cfg["kernel_size"]),
                    sigma=(cfg["blur_sigma"], cfg["blur_sigma"]),
                ).permute(0, 2, 3, 1).contiguous()
            )

            ready_specular = texture.Texture2D(
                kornia.filters.gaussian_blur2d(
                    m.material['ks'].data.permute(0, 3, 1, 2),
                    kernel_size=(cfg["kernel_size"], cfg["kernel_size"]),
                    sigma=(cfg["blur_sigma"], cfg["blur_sigma"]),
                ).permute(0, 2, 3, 1).contiguous()
            )

            ready_normal = texture.Texture2D(
                kornia.filters.gaussian_blur2d(
                    m.material['normal'].data.permute(0, 3, 1, 2),
                    kernel_size=(cfg["kernel_size"], cfg["kernel_size"]),
                    sigma=(cfg["blur_sigma"], cfg["blur_sigma"]),
                ).permute(0, 2, 3, 1).contiguous()
            )
                
            # Final mesh with vertices and textures
            load_mesh = mesh.Mesh(
                n_vert,
                m.t_pos_idx,
                material={
                    'bsdf': cfg['bsdf'],
                    'kd': ready_texture,
                    'ks': ready_specular,
                    'normal': ready_normal,
                },
                base=m # gets uvs etc from here
            )
            
            if it < cfg["epochs"] * cfg["shape_imgs_frac"] and vert_train:

                # Initialize the no texture mesh
                kd_notex = torch.full_like( ready_texture.data, 0.5)

                if kd_notex.shape[-1] == 4:
                    kd_notex[:, :, :, 3] = 1.0

                load_mesh_notex = mesh.Mesh(
                    n_vert,
                    m.t_pos_idx,
                    material={
                        'bsdf': cfg['bsdf'],
                        'kd': kd_notex,
                        'ks': ready_specular,
                        'normal': ready_normal,
                    },
                    base=m # gets uvs etc from here
                )

                render_meshes_notex.append(load_mesh_notex.eval())


            render_meshes.append(load_mesh.eval())

            if subdiv[i] != None:
                lapl_funcs.append(regularizer.laplace_regularizer_const(m))
            else:
                lapl_funcs.append(None)

        # Create a scene with the textures and another without textures
        complete_scene = create_scene(render_meshes, sz=cfg["texture_resolution"])
        complete_scene = mesh.auto_normals(complete_scene)
        complete_scene = mesh.compute_tangents(complete_scene)

        if it < cfg["epochs"] * cfg["shape_imgs_frac"] and vert_train:
            complete_scene_notex = create_scene(render_meshes_notex, sz=cfg["texture_resolution"])
            complete_scene_notex = mesh.auto_normals(complete_scene_notex)
            complete_scene_notex = mesh.compute_tangents(complete_scene_notex)

        # Logging
        if it % cfg["log_interval"] == 0:

            with torch.no_grad():
                
                params = get_camera_params(
                    cfg["log_elev"],
                    rot_ang,
                    cfg["log_dist"],
                    cfg["log_res"],
                    cfg["log_fov"]
                )

                rot_ang += 1

                log_image = render.render_mesh(
                    glctx,
                    complete_scene.eval(params),
                    params['mvp'],
                    params['campos'],
                    params['lightpos'],
                    cfg["log_light_power"],
                    cfg["log_res"],
                    num_layers=cfg["layers"],
                    background=torch.ones(1, cfg["log_res"], cfg["log_res"], 3).to(device)
                )

                log_image = video.ready_image(log_image)


        # Render scene for training
        params_camera = next(iter(cams))

        for key in params_camera:
            params_camera[key] = params_camera[key].to(device)

        # Render with and without texture to enable shape growth
        if it < cfg["epochs"] * cfg["shape_imgs_frac"] and vert_train:
            
            with_tex = cfg["batch_size"] // 2

            with_tex_params = {
                'mvp': params_camera['mvp'][:with_tex],
                'lightpos': params_camera['lightpos'][:with_tex],
                'campos': params_camera['campos'][:with_tex],
                'resolution': [cfg["train_res"], cfg["train_res"]]
            }

            no_tex_params = {
                'mvp': params_camera['mvp'][with_tex:],
                'lightpos': params_camera['lightpos'][with_tex:],
                'campos': params_camera['campos'][with_tex:],
                'resolution': [cfg["train_res"], cfg["train_res"]]
            }

            with_tex_train_render = render.render_mesh(
                glctx,
                complete_scene.eval(with_tex_params),
                with_tex_params["mvp"],
                with_tex_params["campos"],
                with_tex_params["lightpos"],
                cfg["light_power"],
                cfg["train_res"],
                spp=1, # no upscale here / render at any resolution then use resize_right to downscale
                num_layers=cfg["layers"],
                msaa=False,
                background=params_camera["bkgs"][:with_tex],
            ).permute(0, 3, 1, 2) # switch to B, C, H, W

            no_tex_train_render = render.render_mesh(
                glctx,
                complete_scene_notex.eval(no_tex_params),
                no_tex_params["mvp"],
                no_tex_params["campos"],
                no_tex_params["lightpos"],
                cfg["light_power"],
                cfg["train_res"],
                spp=1, # no upscale here / render at any resolution then use resize_right to downscale
                num_layers=1,
                msaa=False,
                background=params_camera["bkgs"][with_tex:],
            ).permute(0, 3, 1, 2) # switch to B, C, H, W

            train_render = torch.cat([
                with_tex_train_render,
                no_tex_train_render
            ])
            
        # Render with only textured meshes
        else:

            params = {
                'mvp': params_camera['mvp'],
                'lightpos': params_camera['lightpos'],
                'campos': params_camera['campos'],
                'resolution': [cfg["train_res"], cfg["train_res"]]
            }

            train_render = render.render_mesh(
                glctx,
                complete_scene.eval(params),
                params["mvp"],
                params["campos"],
                params["lightpos"],
                cfg["light_power"],
                cfg["train_res"],
                spp=1, # no upscale here / render at any resolution then use resize_right to downscale
                num_layers=cfg["layers"],
                msaa=False,
                background=params_camera["bkgs"],
            ).permute(0, 3, 1, 2) # switch to B, C, H, W
            
        # resize to CLIP input size: cubic, linear, lanczos2, lanczos3
        if cfg["resize_method"] == "cubic":

            train_render = resize(
                train_render,
                out_shape=(224, 224), # resize to clip
                interp_method=cubic
            )

        elif cfg["resize_method"] == "linear":

            train_render = resize(
                train_render,
                out_shape=(224, 224), # resize to clip
                interp_method=linear
            )

        elif cfg["resize_method"] == "lanczos2":

            train_render = resize(
                train_render,
                out_shape=(224, 224), # resize to clip
                interp_method=lanczos2
            )
        elif cfg["resize_method"] == "lanczos3":

            train_render = resize(
                train_render,
                out_shape=(224, 224), # resize to clip
                interp_method=lanczos3
            )

        # Log renders
        if it % cfg["log_interval_im"] == 0:
            
            s_log = train_render[torch.randint(low=0, high=cfg["batch_size"], size=(5 if cfg["batch_size"] > 5 else cfg["batch_size"], )) , :, :, :]

            # Source code of save_image
            s_log = torchvision.utils.make_grid(s_log)

            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            ndarr = s_log.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            im = Image.fromarray(ndarr)

            if cfg["colab"]:
                plt.figure()
                plt.imshow(ndarr)
                plt.show()

            im.save(os.path.join(cfg["path"], 'epoch_%d.png' % it))

        # Convert image to image embeddings
        image_embeds = model.encode_image(
            (train_render - clip_mean[None, :, None, None]) / clip_std[None, :, None, None]
        )
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

        # Get loss between text embeds and image embeds
        clip_loss  = cosine_avg(image_embeds, texts_embeds)

        # Get loss between image prior embedding and image embeds
        if cfg["prior_path"] is not None:
            prior_loss = cosine_avg(image_embeds, prior_embeds)

        # Evaluate laplacian for each mesh in scene to be deformed
        lapls = []
        lapls_l = 0
        for fn_l in lapl_funcs:
            if fn_l is not None:
                lapls.append(fn_l.eval(params))

        # Laplace loss weighting
        if it == 0:
            laplacian_weight = cfg["laplacian_weight"]
            laplacian_min = cfg["laplacian_min"]
        else:
            laplacian_weight = (laplacian_weight - laplacian_min) * 10**(-it*0.000001) + laplacian_min

        for lap_l in lapls:
            lapls_l += (laplacian_weight * lap_l)

        # Get total loss and backprop
        if cfg["prior_path"] is not None:
            total_loss = (cfg["clip_weight"] * clip_loss) + (cfg["diff_loss_weight"] * prior_loss) + lapls_l
        else:
            total_loss = (cfg["clip_weight"] * clip_loss) + lapls_l

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        normal_map.clamp_(min=-1, max=1)
        specular_map.clamp_(min=0, max=1)
        texture_map.clamp_(min=0, max=1)

        t_loop.set_description("CLIP Loss = %.6f" % clip_loss.item() )
    
    video.close()

    for idx, m in enumerate(render_meshes):
        out_path = os.path.join( cfg["path"], "meshes", "mesh_%d" % idx )
        os.makedirs(out_path)

        obj.write_obj(
            out_path,
            m
        )

    return cfg["path"]
