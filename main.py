# This file reads a .yml config and if any command line arguments
# are passed it overrides the config with them. It then validates
# the config file, sets seed and sends config to the main opt 
# loop where all the magic happens. The loop returns the final 
# mesh file for saving to disk

import os
import yaml
import torch
import random
import argparse
import numpy as np

from loop import loop

def main():

    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      help='Path to config file', type=str, required=True)

    # Basic
    parser.add_argument('--output_path', help='Where to store output files', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--gpu',         help='GPU index', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--seed',        help='Seed for reproducibility', type=int, default=argparse.SUPPRESS)

    # CLIP Related
    parser.add_argument('--text_prompt', help='Text prompt for mesh generation', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--clip_model',  help='CLIP Model size', type=str, default=argparse.SUPPRESS)

    # Text-Image Prior Related
    parser.add_argument('--prior_path',      help='Path to weights for the prior network, not used if left blank', type=str, default=argparse.SUPPRESS)

    ## Parameters for diffusion prior network (code by lucidrains)
    parser.add_argument('--diffusion_prior_network_dim',        help='Diffusion Prior Network - Dimension', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--diffusion_prior_network_depth',      help='Diffusion Prior Network - Depth', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--diffusion_prior_network_dim_head',   help='Diffusion Prior Network - Head Dimension', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--diffusion_prior_network_heads',      help='Diffusion Prior Network - # of Heads', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--diffusion_prior_network_normformer', help='Diffusion Prior Network - Normformer?', type=bool, default=argparse.SUPPRESS)

    ## Parameters for diffusion prior (code by lucidrains)
    parser.add_argument('--diffusion_prior_embed_dim',                   help='Diffusion Prior Network - Embedding Dimension', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--diffusion_prior_timesteps',                   help='Diffusion Prior Network - Timesteps', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--diffusion_prior_cond_drop_prob',              help='Diffusion Prior Network - Conditional Drop Probability', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--diffusion_prior_loss_type',                   help='Diffusion Prior Network - Loss Type', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--diffusion_prior_condition_on_text_encodings', help='Diffusion Prior Network - Condition Prior on Text Encodings?', type=bool, default=argparse.SUPPRESS)

    # Parameters
    parser.add_argument('--epochs',             help='Number of optimization steps', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--lr',                 help='Maximum learning rate', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--batch_size',         help='Number of images rendered at the same time', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--train_res',          help='Resolution of render before downscaling to CLIP size', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--resize_method',      help='Image downsampling/upsampling method', type=str, default=argparse.SUPPRESS, choices=["cubic", "linear", "lanczos2", "lanczos3"])
    parser.add_argument('--bsdf',               help='Render technique', type=str, default=argparse.SUPPRESS, choices=["diffuse", "pbr"])
    parser.add_argument('--texture_resolution', help='Resolution of texture maps (ex: 512 -> 512x512)', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--channels',           help='Texture map image channels (4 for alpha, 3 for RGB only)', type=int, default=argparse.SUPPRESS, choices=[3, 4])
    parser.add_argument('--init_c',             help='Initial alpha channel value if channels == 4', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--kernel_size',        help='Kernel size for gaussian blurring of textures to reduce artifacts', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--blur_sigma',         help='Variance of gaussian kernel for blurring of textures', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--shape_imgs_frac',    help='What % of epochs should the renders include plain shape renders as well as textures - after which only textured renders are done', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--aug_light',          help='Augment the direction of light around the camera', type=bool, default=argparse.SUPPRESS)
    parser.add_argument('--aug_bkg',            help='Augment the background', type=bool, default=argparse.SUPPRESS)
    parser.add_argument('--diff_loss_weight',   help='Weight of Diffusion prior loss', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--clip_weight',        help='Weight of CLIP Text loss', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--laplacian_weight',   help='Initial uniform laplacian weight', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--laplacian_min',      help='Minimum uniform laplacian weight (set to 2% of max usually)', type=float, default=argparse.SUPPRESS)

    # Camera Parameters
    parser.add_argument('--fov_min',            help='Minimum camera field of view angle during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--fov_max',            help='Maximum camera field of view angle during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--dist_min',           help='Minimum distance of camera from mesh during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--dist_max',           help='Maximum distance of camera from mesh during renders', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--light_power',        help='Light intensity', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--elev_alpha',         help='Alpha parameter for Beta distribution for elevation sampling', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--elev_beta',          help='Beta parameter for Beta distribution for elevation sampling', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--elev_max',           help='Maximum elevation angle in degree', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--azim_min',           help='Minimum azimuth angle in degree',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--azim_max',           help='Maximum azimuth angle in degree', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--aug_loc',            help='Offset mesh from center of image?', type=bool, default=argparse.SUPPRESS)

    # Logging Parameters
    parser.add_argument('--log_interval',       help='Interval for logging, every X epochs',  type=int, default=argparse.SUPPRESS)
    parser.add_argument('--log_interval_im',    help='Interval for logging renders image, every X epochs',  type=int, default=argparse.SUPPRESS)
    parser.add_argument('--log_elev',           help='Logging elevation angle',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--log_fov',            help='Logging field of view',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--log_dist',           help='Logging distance from object',  type=float, default=argparse.SUPPRESS)
    parser.add_argument('--log_res',            help='Logging render resolution',  type=int, default=argparse.SUPPRESS)
    parser.add_argument('--log_light_power',    help='Light intensity for logging',  type=float, default=argparse.SUPPRESS)

    # Mesh Parameters
    parser.add_argument('--meshes',             help="Path to all meshes in scene", nargs='+', default=argparse.SUPPRESS, type=str)
    parser.add_argument('--unit',               help="Should mesh be unit scaled? True/False for each mesh in meshes", nargs='+', default=argparse.SUPPRESS, type=bool)
    parser.add_argument('--train_mesh_idx',     help="What parameters to optimize for each mesh or none at all (vertices, texture map, normal map, true/false for limit subdivide) ?", nargs='+', action='append', default=argparse.SUPPRESS)
    parser.add_argument('--scales',             help="Scale mesh size by some value", nargs='+', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--offsets',            help="After scaling (x, y, z) offset vertices", nargs='+', action='append', type=float, default=argparse.SUPPRESS)

    args = vars(parser.parse_args())

    # Check if config passed - if so then parse it
    if args['config'] is not None:
        with open(args['config'], "r") as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        raise("No config passed!")

    # Override YAML with CL args
    for key in args:
        cfg[key] = args[key]

    # Config validation
    lists = ["meshes", "unit", "train_mesh_idx", "scales", "offsets", "prior_path"]
    for item in parser._actions[1:]:
        if item.type != type(cfg[ item.dest ]) and item.dest not in lists:
            raise ValueError("%s is not of type %s" % (item.dest, item.type) )
                    
    if not( len(cfg["meshes"]) == len(cfg["unit"]) == len(cfg["train_mesh_idx"]) == len(cfg["scales"]) == len(cfg["offsets"])):
        raise("Unit, train_mesh_idx, scales and offsets is not specified for each mesh OR there is an extra item in some list. Ensure all are the same length")

    print(yaml.dump(cfg, default_flow_style=False))

    # Set seed
    random.seed(cfg["seed"])
    os.environ['PYTHONHASHSEED'] = str(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = True

    loop(cfg)

if __name__ == '__main__':
    main()