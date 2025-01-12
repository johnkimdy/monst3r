# --------------------------------------------------------
# Gradio Demo with Hyperparameter-Included Output Filenames
# --------------------------------------------------------
import gc
import argparse
import math
import gradio
import os
import torch
import numpy as np
import tempfile
import functools
import copy
import pickle
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import random
import re

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.viz_demo import convert_scene_output_to_glb, get_dynamic_mask_from_pairviewer
import matplotlib.pyplot as pl
from tqdm import tqdm

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for GPU >= Ampere and PyTorch >= 1.12
batch_size = 1

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seeds()

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="Make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="Server URL, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="Image size")
    parser.add_argument("--server_port", type=int, help=("Start Gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--weights", type=str, help="Path to the model weights", default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument("--model_name", type=str, default='Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt', help="Model name")
    parser.add_argument("--device", type=str, default='cuda', help="PyTorch device")
    parser.add_argument("--output_dir", type=str, default='./demo_tmp', help="Directory for output files")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="Silence logs")
    parser.add_argument("--input_dir", type=str, help="Path to input images directory", default=None)
    parser.add_argument("--seq_name", type=str, help="Sequence name for evaluation", default='NULL')
    parser.add_argument('--use_gt_davis_masks', action='store_true', default=False, help='Use ground truth masks for DAVIS')
    parser.add_argument('--not_batchify', action='store_true', default=False, help='Use non-batchify mode for global optimization')
    parser.add_argument('--fps', type=int, default=12, help='FPS for video processing')
    parser.add_argument('--num_frames', type=int, default=100, help='Maximum number of frames for video processing')
    
    # Add "share" argument if you want to make the demo accessible on the public internet
    parser.add_argument("--share", action='store_true', default=False, help="Share the demo")


    return parser

def sanitize_value(value):
    """
    Sanitize hyperparameter values to be filesystem-friendly.
    Replace periods with 'p' for floats, remove or replace special characters, and limit length.
    """
    if isinstance(value, float):
        value_str = str(value).replace('.', 'p')
    elif isinstance(value, bool):
        value_str = 'True' if value else 'False'
    else:
        value_str = str(value)
    
    # Remove any characters that are not alphanumeric, underscores, or hyphens
    value_str = re.sub(r'[^\w\-]', '', value_str)
    
    # Truncate if necessary to prevent excessively long filenames
    max_length = 10
    if len(value_str) > max_length:
        value_str = value_str[:max_length]
    
    return value_str

def generate_hyperparam_str(params, include_keys=None):
    """
    Generate a sanitized hyperparameter string for filenames.
    
    Args:
        params (dict): The hyperparameter combination.
        include_keys (list, optional): List of hyperparameters to include. If None, include all.
        
    Returns:
        str: A sanitized hyperparameter string.
    """
    if include_keys is None:
        include_keys = params.keys()
    
    parts = []
    for key in include_keys:
        value = sanitize_value(params[key])
        parts.append(f"{key}_{value}")
    
    hyperparam_str = "_".join(parts)
    return hyperparam_str

def expand_hyperparameter_grid(grid):
    """
    Expands hyperparameters defined with min, max, step_size into lists of values.
    
    Args:
        grid (dict): The hyperparameter grid with possible range specifications.
        
    Returns:
        dict: The expanded hyperparameter grid with all lists.
    """
    expanded_grid = {}
    for param, values in grid.items():
        if isinstance(values, dict):
            if 'step_size' in values:
                min_val = values['min']
                max_val = values['max']
                step_size = values['step_size']
                
                # Generate the list using np.arange
                expanded_values = list(np.arange(min_val, max_val + step_size, step_size))
                
                # Optional: Round to desired decimal places to avoid floating point issues
                expanded_values = [round(v, 10) for v in expanded_values]
                
                expanded_grid[param] = expanded_values
            else:
                raise ValueError(f"Unsupported range specification for hyperparameter '{param}': {values}")
        elif isinstance(values, list):
            expanded_grid[param] = values
        elif isinstance(values, (int, float, bool, str)):
            # Wrap single values into lists for consistency
            expanded_grid[param] = [values]
        else:
            raise ValueError(f"Invalid format for hyperparameter '{param}': {values}")
    return expanded_grid

def get_3D_model_from_scene(outdir, silent, scene, device, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, show_cam=True, save_name=None, thr_for_init_conf=True):
    """
    Extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # Post-processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # Get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses, and intrinsics
    pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
    scene.min_conf_thr = min_conf_thr
    scene.thr_for_init_conf = thr_for_init_conf
    msk = to_numpy(scene.get_masks())
    cmap = pl.get_cmap('viridis')
    cam_color = [cmap(i/len(rgbimg))[:3] for i in range(len(rgbimg))]
    cam_color = [(255*c[0], 255*c[1], 255*c[2]) for c in cam_color]
    return convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, show_cam=show_cam, silent=silent, save_name=save_name,
                                        cam_color=cam_color)

def get_reconstructed_scene(args, outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
                            seq_name, temporal_smoothing_weight, translation_weight, shared_focal, 
                            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask, fps, num_frames):
    """
    From a list of images, run dust3r inference, global aligner.
    Then run get_3D_model_from_scene
    """
    translation_weight = float(translation_weight)
    
    # Use args.weights as the model weights
    new_model_weights = args.weights
    
    if new_model_weights != args.weights:
       model = AsymmetricCroCo3DStereo.from_pretrained(new_model_weights).to(device)
    model.eval()
    
    if seq_name != "NULL":
        dynamic_mask_path = f'data/davis/DAVIS/masked_images/480p/{seq_name}'
    else:
        dynamic_mask_path = None
    
    imgs = load_images(filelist, size=image_size, verbose=not silent, dynamic_mask_root=dynamic_mask_path, fps=fps, num_frames=num_frames)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type in ["swin", "swinstride", "swin2stride"]:
        scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
    
    if len(imgs) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer  
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent, shared_focal=shared_focal, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                               flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
                               num_total_iter=niter, empty_cache=len(filelist) > 72, batchify=not args.not_batchify)
    else:
        mode = GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    save_folder = outdir  # Updated to use the provided combination directory
    os.makedirs(save_folder, exist_ok=True)
    outfile = get_3D_model_from_scene(save_folder, silent, scene, device, min_conf_thr, as_pointcloud, mask_sky,
                            clean_depth, transparent_cams, cam_size, show_cam)
    
    poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
    K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
    depth_maps = scene.save_depth_maps(save_folder)
    dynamic_masks = scene.save_dynamic_masks(save_folder)
    conf = scene.save_conf_maps(save_folder)
    init_conf = scene.save_init_conf_maps(save_folder)
    rgbs = scene.save_rgb_imgs(save_folder)
    enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3) 

    # Also return rgb, depth and confidence imgs
    # Depth is normalized with the max value for all images
    # We apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    init_confs = to_numpy([c for c in scene.init_conf_maps])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [cmap(d/depths_max) for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d/confs_max) for d in confs]
    init_confs_max = max([d.max() for d in init_confs])
    init_confs = [cmap(d/init_confs_max) for d in init_confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))
        imgs.append(rgb(init_confs[i]))

    # If two images, and the shape is same, we can compute the dynamic mask
    if len(rgbimg) == 2 and rgbimg[0].shape == rgbimg[1].shape:
        motion_mask_thre = 0.35
        error_map = get_dynamic_mask_from_pairviewer(scene, both_directions=True, output_dir=save_folder, motion_mask_thre=motion_mask_thre)
        # imgs.append(rgb(error_map))
        # Apply threshold on the error map
        normalized_error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
        error_map_max = normalized_error_map.max()
        error_map = cmap(normalized_error_map/error_map_max)
        imgs.append(rgb(error_map))
        binary_error_map = (normalized_error_map > motion_mask_thre).astype(np.uint8)
        imgs.append(rgb(binary_error_map*255))

    return scene, outfile, imgs

def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    # If inputfiles[0] is a video, set the num_files to 200
    if inputfiles is not None and len(inputfiles) == 1 and inputfiles[0].name.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV')):
        num_files = 200
    else:
        num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files-1)/2))
    if scenegraph_type in ["swin", "swin2stride", "swinstride"]:
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=min(max_winsize,5),
                                minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=True)
    else:
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    return winsize, refid

def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False, args=None):
    recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="MonST3R Demo") as demo:
        # Scene state is saved so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML(f'<h2 style="text-align: center;">MonST3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                schedule = gradio.Dropdown(["linear", "cosine"],
                                           value='linear', label="schedule", info="For global alignment!")
                niter = gradio.Number(value=300, precision=0, minimum=0, maximum=5000,
                                      label="num_iterations", info="For global alignment!")
                seq_name = gradio.Textbox(label="Sequence Name", placeholder="NULL", value=args.seq_name, info="For evaluation")
                scenegraph_type = gradio.Dropdown(["complete", "swin", "oneref", "swinstride", "swin2stride"],
                                                  value='swinstride', label="Scenegraph",
                                                  info="Define how to make pairs",
                                                  interactive=True)
                winsize = gradio.Slider(label="Scene Graph: Window Size", value=5,
                                        minimum=1, maximum=1, step=1, visible=False)
                refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0, maximum=0, step=1, visible=False)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                # Adjust the confidence threshold
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.1, minimum=0.0, maximum=20, step=0.1)  # Updated step
                # Adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1, step=0.01)  # Updated step
                # Adjust the temporal smoothing weight
                temporal_smoothing_weight = gradio.Slider(label="temporal_smoothing_weight", value=0.01, minimum=0.0, maximum=0.1, step=0.005)  # Updated step
                # Add translation weight
                translation_weight = gradio.Textbox(label="translation_weight", placeholder="1.0", value="1.0", info="For evaluation")
                # Change to another model (Removed from grid search if fixed)
                # new_model_weights = gradio.Textbox(label="New Model", placeholder=args.weights, value=args.weights, info="Path to updated model weights")
                # Assuming Option 2: Fixed model weights

            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # Two post-process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")
                # Not to show camera
                show_cam = gradio.Checkbox(value=True, label="Show Camera")
                shared_focal = gradio.Checkbox(value=True, label="Shared Focal Length")
                use_davis_gt_mask = gradio.Checkbox(value=False, label="Use GT Mask (DAVIS)")
            with gradio.Row():
                flow_loss_weight = gradio.Slider(label="Flow Loss Weight", value=0.01, minimum=0.0, maximum=1.0, step=0.01)  # Updated step
                flow_loss_start_iter = gradio.Slider(label="Flow Loss Start Iter", value=0.1, minimum=0, maximum=3, step=0.1)  # Updated step
                flow_loss_threshold = gradio.Slider(label="Flow Loss Threshold", value=25, minimum=1, maximum=3, step=0.1)  # Updated step
                # For video processing
                fps = gradio.Slider(label="FPS", value=12, minimum=0, maximum=60, step=1)
                num_frames = gradio.Slider(label="Num Frames", value=100, minimum=0, maximum=200, step=1)

            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(label='rgb,depth,confidence, init_conf', columns=4, height="100%")

            # Events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, winsize, refid, scenegraph_type],
                                   outputs=[winsize, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, winsize, refid, scenegraph_type],
                              outputs=[winsize, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                                  mask_sky, clean_depth, transparent_cams, cam_size, show_cam,
                                  scenegraph_type, winsize, refid, seq_name, 
                                  temporal_smoothing_weight, translation_weight, shared_focal, 
                                  flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask,
                                  fps, num_frames],
                          outputs=[scene, outmodel, outgallery])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, show_cam],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, show_cam],
                            outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, show_cam],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, show_cam],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, show_cam],
                               outputs=outmodel)
            transparent_cams.change(fn=model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, show_cam],
                                    outputs=outmodel)
        demo.launch(share=args.share, server_name=server_name, server_port=server_port)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_dir is not None:
        tmp_path = args.output_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None and os.path.exists(args.weights):
        weights_path = args.weights
    else:
        weights_path = args.model_name

    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    # Use the provided output_dir or create a temporary directory
    tmpdirname = args.output_dir if args.output_dir is not None else tempfile.mkdtemp(suffix='monst3r_gradio_demo')

    if not args.silent:
        print('Outputting stuff in', tmpdirname)

    if args.input_dir is not None:
        # Process images in the input directory with default parameters
        if os.path.isdir(args.input_dir):    # input_dir is a directory of images
            input_files = [os.path.join(args.input_dir, fname) for fname in sorted(os.listdir(args.input_dir))]
        else:   # input_dir is a video
            input_files = [args.input_dir]
        recon_fun = functools.partial(get_reconstructed_scene, args=args, device=args.device, silent=args.silent, image_size=args.image_size)
        
        # Define the hyperparameter grid
        hyperparameter_grid = {
            'schedule': ['linear', 'cosine'],                       # 2
            'niter': [300],                                     # 1
            'min_conf_thr': {'min': 1, 'max': 2, 'step_size': 0.5},      # 5
            'scenegraph_type': ['swinstride', 'complete', 'oneref', 'swin', 'swin2stride'],  # 5
            'winsize': {'min': 7, 'max': 9, 'step_size': 2},            # 2
            'temporal_smoothing_weight': {'min': 0.01, 'max': 0.05, 'step_size': 0.02},  # 2
            'translation_weight': ['1.0'],                # 1
            'flow_loss_weight': {'min': 0.05, 'max': 0.1, 'step_size': 0.05},  # 2
            # 'flow_loss_start_iter': {'min': 0.1, 'max': 0.3, 'step_size': 0.1},  # 3
            'flow_loss_start_iter': [0.1],
            # 'flow_loss_threshold': {'min': 10, 'max': 50, 'step_size': 10},     # 5
            'flow_loss_threshold': [25],
            'use_gt_mask': [False],                               # 2
            'fps': [12],                                                 # 1
            'num_frames': [100],                                         # 1
            # Fixed hyperparameters:
            'as_pointcloud': [True],
            'mask_sky': [False],
            'clean_depth': [True],
            'transparent_cams': [False],
            'cam_size': [0.05],
            'show_cam': [True],
            'refid': [0],
            'shared_focal': [True],
            # Total Combinations: 2 * 1 * 5 * 5 * 3 * 2 * 3 * 3 * 3 * 2 * 1 * 1 * 1 =  
        }

        # Create a directory to store results
        results_dir = "hyperparameter_search_results"
        os.makedirs(results_dir, exist_ok=True)

        # Expand the hyperparameter grid
        expanded_hyperparameter_grid = expand_hyperparameter_grid(hyperparameter_grid)

        # Initialize ParameterGrid
        grid = ParameterGrid(expanded_hyperparameter_grid)
        total_combinations = len(grid)
        print(f"Total hyperparameter combinations to evaluate: {total_combinations}")

        # Optionally, set up logging
        log_file = os.path.join(results_dir, "grid_search_log.txt")
        with open(log_file, 'w') as log:
            log.write(f"Hyperparameter Grid Search Log - {datetime.now()}\n")
            log.write(f"Total combinations: {total_combinations}\n\n")

        # Define which hyperparameters to include in the filename
        # Choose hyperparameters that significantly impact performance
        include_keys = [
            'schedule', 
            'niter', 
            'min_conf_thr', 
            'scenegraph_type', 
            'winsize', 
            'translation_weight',
            'flow_loss_weight',
            'flow_loss_start_iter',
            'flow_loss_threshold',
            'use_gt_mask'
        ]

        # Iterate over all combinations with a progress bar
        for idx, params in enumerate(tqdm(grid, desc="Grid Search"), 1):
            print(f"Evaluating combination {idx}/{total_combinations}: {params}")

            # Log the current combination
            with open(log_file, 'a') as log:
                log.write(f"Combination {idx}/{total_combinations}: {params}\n")

            # Generate a sanitized hyperparameter string
            hyperparam_str = generate_hyperparam_str(params, include_keys=include_keys)

            # Define 'seq_name' with hyperparameter info
            seq_name = f"combo_{idx}_{hyperparam_str}"

            # Define combination directory with hyperparameter string
            combination_dir = os.path.join(results_dir, seq_name)
            os.makedirs(combination_dir, exist_ok=True)

            try:
                # Convert string hyperparameters to appropriate types if necessary
                translation_weight = float(params['translation_weight']) if isinstance(params['translation_weight'], str) else params['translation_weight']

                # Call recon_fun with the current set of hyperparameters
                # Since 'seq_name' is now part of the parameters, pass it as follows:
                scene, outfile, imgs = recon_fun(
                    outdir=combination_dir,model=model,
                    filelist=input_files,
                    schedule=params['schedule'],
                    niter=params['niter'],
                    min_conf_thr=params['min_conf_thr'],
                    as_pointcloud=params['as_pointcloud'],
                    mask_sky=params['mask_sky'],
                    clean_depth=params['clean_depth'],
                    transparent_cams=params['transparent_cams'],
                    cam_size=params['cam_size'],
                    show_cam=params['show_cam'],
                    scenegraph_type=params['scenegraph_type'],
                    winsize=params['winsize'],
                    refid=params['refid'],
                    seq_name=seq_name,  # Pass the new seq_name
                    temporal_smoothing_weight=params['temporal_smoothing_weight'],
                    translation_weight=translation_weight,
                    shared_focal=params['shared_focal'],
                    flow_loss_weight=params['flow_loss_weight'],
                    flow_loss_start_iter=params['flow_loss_start_iter'],
                    flow_loss_threshold=params['flow_loss_threshold'],
                    use_gt_mask=params['use_gt_mask'],
                    fps=params['fps'],
                    num_frames=params['num_frames'],
                )





                # Log completion
                print(f"Combination {idx} completed. Output saved in {combination_dir}")
                with open(log_file, 'a') as log:
                    log.write(f"Combination {idx} completed successfully.\n\n")

            except Exception as e:
                # Handle exceptions, log them, and continue
                print(f"Combination {idx} failed with exception: {e}")
                with open(log_file, 'a') as log:
                    log.write(f"Combination {idx} failed with exception: {e}\n\n")
                torch.cuda.empty_cache()
                continue  # Proceed to the next combination
            finally:
                # Delete variables to free memory
                # Collect garbage
                gc.collect()
                torch.cuda.empty_cache()
                with open(log_file, 'a') as log:
                    log.write(torch.cuda.memory_summary(device=args.device))
                print("GPU memory cleared.")

        print("Hyperparameter grid search completed.")
    else:
        # Launch Gradio demo
        main_demo(tmpdirname, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent, args=args)
