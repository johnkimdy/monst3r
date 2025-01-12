# demo.py
# --------------------------------------------------------
# Modified to support both interactive and non-interactive runs
# --------------------------------------------------------

import argparse
import math
import gradio
import os
import torch
import numpy as np
import tempfile
import functools
import copy

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.viz_demo import convert_scene_output_to_glb, get_dynamic_mask_from_pairviewer
import matplotlib.pyplot as pl

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--server_port", type=int,
                        help=("will start gradio app on this port (if available). "
                              "If None, will search for an available port starting at 7860."),
                        default=None)

    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224],
                        help="image size")
    parser.add_argument("--weights", type=str, 
                        help="path to the model weights", 
                        default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument("--model_name", type=str, 
                        default='Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt',
                        help="model name")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--output_dir", type=str, default='./demo_tmp',
                        help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to input images directory or a single video file")
    parser.add_argument("--seq_name", type=str, default='NULL',
                        help="Sequence name for evaluation (also used for saving outputs)")
    parser.add_argument("--use_gt_davis_masks", action='store_true', default=False,
                        help="Use ground truth masks for DAVIS (if relevant)")

    # Non-interactive “override” hyperparameters for global aligner
    parser.add_argument("--schedule", type=str, default='linear', choices=['linear', 'cosine'],
                        help="Schedule for global alignment, used for non-interactive usage")
    parser.add_argument("--niter", type=int, default=300,
                        help="Number of iterations for global alignment, used for non-interactive usage")
    parser.add_argument("--min_conf_thr", type=float, default=1.1,
                        help="Minimum confidence threshold, used for non-interactive usage")
    parser.add_argument("--scenegraph_type", type=str,
                        default='swinstride',
                        choices=['complete','oneref','swin','swinstride','swin2stride'],
                        help="Scenegraph type, used for non-interactive usage")
    parser.add_argument("--winsize", type=int, default=5,
                        help="Win size for scenegraph, used for non-interactive usage")
    parser.add_argument("--temporal_smoothing_weight", type=float, default=0.01,
                        help="Temporal smoothing weight")
    parser.add_argument("--translation_weight", type=float, default=1.0,
                        help="Translation weight")
    parser.add_argument("--shared_focal", type=lambda x: (str(x).lower() == 'true'),
                        default=True,
                        help="Shared focal length for global aligner (True/False)")
    parser.add_argument("--flow_loss_weight", type=float, default=0.01,
                        help="Flow loss weight for global aligner")
    parser.add_argument("--flow_loss_start_iter", type=float, default=0.1,
                        help="Flow loss start iteration (fraction of niter) for global aligner")
    parser.add_argument("--flow_loss_threshold", type=float, default=25,
                        help="Flow loss threshold for global aligner")
    parser.add_argument("--use_gt_mask", type=lambda x: (str(x).lower() == 'true'),
                        default=False,
                        help="Use ground truth mask (DAVIS only) for global aligner (True/False)")
    parser.add_argument("--fps", type=int, default=12,
                        help="FPS for video processing (non-interactive usage)")
    parser.add_argument("--num_frames", type=int, default=100,
                        help="Maximum number of frames for video processing (non-interactive usage)")
    
    # Additional visualization / post-processing toggles
    parser.add_argument("--as_pointcloud", type=lambda x: (str(x).lower() == 'true'),
                        default=True,
                        help="Export results as pointcloud (True/False)")
    parser.add_argument("--mask_sky", type=lambda x: (str(x).lower() == 'true'),
                        default=False,
                        help="Mask sky (True/False)")
    parser.add_argument("--clean_depth", type=lambda x: (str(x).lower() == 'true'),
                        default=True,
                        help="Clean up depthmaps (True/False)")
    parser.add_argument("--transparent_cams", type=lambda x: (str(x).lower() == 'true'),
                        default=False,
                        help="Transparent cameras in 3D viewer (True/False)")
    parser.add_argument("--show_cam", type=lambda x: (str(x).lower() == 'true'),
                        default=True,
                        help="Show cameras in 3D viewer (True/False)")
    parser.add_argument("--cam_size", type=float, default=0.05,
                        help="Camera size in the 3D viewer")

    # Add "share" argument if you want to make the demo accessible on the public internet
    parser.add_argument("--share", action='store_true', default=False,
                        help="Share the demo")
    # Add this line:
    parser.add_argument('--not_batchify',
                        action='store_true',
                        default=False,
                        help='Use non-batchify mode for global optimization')

    return parser


def get_3D_model_from_scene(
    outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
    clean_depth=False, transparent_cams=False, cam_size=0.05, show_cam=True, save_name=None,
    thr_for_init_conf=True
):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
    scene.min_conf_thr = min_conf_thr
    scene.thr_for_init_conf = thr_for_init_conf
    msk = to_numpy(scene.get_masks())
    cmap = pl.get_cmap('viridis')
    cam_color = [cmap(i/len(rgbimg))[:3] for i in range(len(rgbimg))]
    cam_color = [(255*c[0], 255*c[1], 255*c[2]) for c in cam_color]
    return convert_scene_output_to_glb(
        outdir, rgbimg, pts3d, msk, focals, cams2world, 
        as_pointcloud=as_pointcloud,
        transparent_cams=transparent_cams, 
        cam_size=cam_size, 
        show_cam=show_cam, 
        silent=silent, 
        save_name=save_name,
        cam_color=cam_color
    )


def get_reconstructed_scene(
    args, outdir, model, device, silent, image_size, filelist, 
    schedule, niter, min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, 
    cam_size, show_cam, scenegraph_type, winsize, refid, seq_name, new_model_weights, 
    temporal_smoothing_weight, translation_weight, shared_focal,
    flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask,
    fps, num_frames
):
    """
    from a list of images, run dust3r inference, global aligner, then 3D model extraction
    """
    translation_weight = float(translation_weight)
    if new_model_weights != args.weights:
        model = AsymmetricCroCo3DStereo.from_pretrained(new_model_weights).to(device)
    model.eval()

    # If seq_name is set to a valid DAVIS seq, dynamic_mask_path would be used
    # but for custom data you probably won't have that path
    if seq_name != "NULL":
        dynamic_mask_path = f'data/davis/DAVIS/masked_images/480p/{seq_name}'
    else:
        dynamic_mask_path = None

    imgs = load_images(filelist, size=image_size, verbose=not silent,
                       dynamic_mask_root=dynamic_mask_path, fps=fps, num_frames=num_frames)
    if len(imgs) == 1:
        # hack so it doesn't fail in global aligner
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    # Adjust scenegraph type
    if scenegraph_type in ["swin", "swinstride", "swin2stride"]:
        scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)

    if len(imgs) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer
        scene = global_aligner(
            output, device=device, mode=mode, verbose=not silent,
            shared_focal=shared_focal, 
            temporal_smoothing_weight=temporal_smoothing_weight, 
            translation_weight=translation_weight,
            flow_loss_weight=flow_loss_weight, 
            flow_loss_start_epoch=flow_loss_start_iter, 
            flow_loss_thre=flow_loss_threshold, 
            use_self_mask=not use_gt_mask,
            num_total_iter=niter, 
            empty_cache=len(filelist) > 72, 
            batchify=not args.not_batchify
        )
    else:
        mode = GlobalAlignerMode.PairViewer
        scene = global_aligner(
            output, device=device, mode=mode, verbose=not silent
        )

    lr = 0.01
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    save_folder = f'{args.output_dir}/{seq_name}'  # default: 'demo_tmp/NULL'
    os.makedirs(save_folder, exist_ok=True)
    outfile = get_3D_model_from_scene(
        save_folder, silent, scene, min_conf_thr, 
        as_pointcloud, mask_sky, clean_depth,
        transparent_cams, cam_size, show_cam
    )

    # Save typical outputs
    poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
    K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
    depth_maps = scene.save_depth_maps(save_folder)
    dynamic_masks = scene.save_dynamic_masks(save_folder)
    conf = scene.save_conf_maps(save_folder)
    init_conf = scene.save_init_conf_maps(save_folder)
    rgbs = scene.save_rgb_imgs(save_folder)
    enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3)

    # Prepare gallery images (rgb, depth, conf, init_conf)
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    init_confs = to_numpy([c for c in scene.init_conf_maps])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths]) if depths else 1.0
    depths = [cmap(d/depths_max) for d in depths]
    confs_max = max([d.max() for d in confs]) if confs else 1.0
    confs = [cmap(d/confs_max) for d in confs]
    init_confs_max = max([d.max() for d in init_confs]) if init_confs else 1.0
    init_confs = [cmap(d/init_confs_max) for d in init_confs]

    imgs_out = []
    for i in range(len(rgbimg)):
        imgs_out.append(rgb(rgbimg[i]))
        imgs_out.append(rgb(depths[i]))
        imgs_out.append(rgb(confs[i]))
        imgs_out.append(rgb(init_confs[i]))

    # If only two images (PairViewer), also produce an error_map-based dynamic mask
    if len(rgbimg) == 2 and rgbimg[0].shape == rgbimg[1].shape:
        motion_mask_thre = 0.35
        error_map = get_dynamic_mask_from_pairviewer(
            scene, both_directions=True, output_dir=args.output_dir,
            motion_mask_thre=motion_mask_thre
        )
        normalized_error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-9)
        error_map_colored = cmap(normalized_error_map)
        imgs_out.append(rgb(error_map_colored))
        binary_error_map = (normalized_error_map > motion_mask_thre).astype(np.uint8)
        imgs_out.append(rgb(binary_error_map*255))

    return scene, outfile, imgs_out


def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    # if inputfiles[0] is a video, set the num_files to 200
    if inputfiles is not None and len(inputfiles) == 1 and \
       inputfiles[0].name.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV')):
        num_files = 200
    else:
        num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files-1)/2))
    if scenegraph_type in ["swin", "swin2stride", "swinstride"]:
        winsize = gradio.Slider(
            label="Scene Graph: Window Size",
            value=min(max_winsize, 5),
            minimum=1, maximum=max_winsize, step=1, visible=True
        )
        refid = gradio.Slider(
            label="Scene Graph: Id",
            value=0, minimum=0, maximum=num_files-1, step=1, visible=False
        )
    elif scenegraph_type == "oneref":
        winsize = gradio.Slider(
            label="Scene Graph: Window Size",
            value=max_winsize,
            minimum=1, maximum=max_winsize, step=1, visible=False
        )
        refid = gradio.Slider(
            label="Scene Graph: Id",
            value=0, minimum=0, maximum=num_files-1, step=1, visible=True
        )
    else:
        winsize = gradio.Slider(
            label="Scene Graph: Window Size",
            value=max_winsize,
            minimum=1, maximum=max_winsize, step=1, visible=False
        )
        refid = gradio.Slider(
            label="Scene Graph: Id",
            value=0, minimum=0, maximum=num_files-1, step=1, visible=False
        )
    return winsize, refid


def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False, args=None):
    recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)

    with gradio.Blocks(
        css=""".gradio-container {margin: 0 !important; min-width: 100%};""",
        title="MonST3R Demo"
    ) as demo:
        scene = gradio.State(None)
        gradio.HTML(f'<h2 style="text-align: center;">MonST3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                schedule = gradio.Dropdown(["linear", "cosine"],
                                           value='linear', label="schedule",
                                           info="For global alignment!")
                niter = gradio.Number(value=300, precision=0, minimum=0, maximum=5000,
                                      label="num_iterations", info="For global alignment!")
                seq_name = gradio.Textbox(label="Sequence Name", placeholder="NULL",
                                          value=args.seq_name, info="For evaluation")
                scenegraph_type = gradio.Dropdown(
                    ["complete", "swin", "oneref", "swinstride", "swin2stride"],
                    value='swinstride', label="Scenegraph", info="Define how to make pairs",
                    interactive=True
                )
                winsize = gradio.Slider(label="Scene Graph: Window Size",
                                        value=5,
                                        minimum=1, maximum=1, step=1, visible=False)
                refid = gradio.Slider(label="Scene Graph: Id",
                                      value=0, minimum=0, maximum=0, step=1, visible=False)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.1,
                                             minimum=0.0, maximum=20, step=0.01)
                cam_size = gradio.Slider(label="cam_size", value=0.05,
                                         minimum=0.001, maximum=0.1, step=0.001)
                temporal_smoothing_weight = gradio.Slider(
                    label="temporal_smoothing_weight", value=0.01,
                    minimum=0.0, maximum=0.1, step=0.001
                )
                translation_weight = gradio.Textbox(label="translation_weight",
                                                    placeholder="1.0",
                                                    value="1.0")
                new_model_weights = gradio.Textbox(label="New Model",
                                                   placeholder=args.weights,
                                                   value=args.weights)

            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")
                show_cam = gradio.Checkbox(value=True, label="Show Camera")
                shared_focal = gradio.Checkbox(value=True, label="Shared Focal Length")
                use_davis_gt_mask = gradio.Checkbox(value=False, label="Use GT Mask (DAVIS)")

            with gradio.Row():
                flow_loss_weight = gradio.Slider(label="Flow Loss Weight", value=0.01,
                                                 minimum=0.0, maximum=1.0, step=0.001)
                flow_loss_start_iter = gradio.Slider(label="Flow Loss Start Iter", value=0.1,
                                                     minimum=0, maximum=1, step=0.01)
                flow_loss_threshold = gradio.Slider(label="Flow Loss Threshold", value=25,
                                                    minimum=0, maximum=100, step=1)
                fps = gradio.Slider(label="FPS", value=0,
                                    minimum=0, maximum=60, step=1)
                num_frames = gradio.Slider(label="Num Frames", value=100,
                                           minimum=0, maximum=200, step=1)

            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(label='rgb,depth,confidence, init_conf',
                                        columns=4, height="100%")

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, winsize, refid, scenegraph_type],
                                   outputs=[winsize, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, winsize, refid, scenegraph_type],
                              outputs=[winsize, refid])

            run_btn.click(
                fn=recon_fun,
                inputs=[
                    inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                    mask_sky, clean_depth, transparent_cams, cam_size, show_cam,
                    scenegraph_type, winsize, refid, seq_name, new_model_weights,
                    temporal_smoothing_weight, translation_weight, shared_focal,
                    flow_loss_weight, flow_loss_start_iter, flow_loss_threshold,
                    use_davis_gt_mask, fps, num_frames
                ],
                outputs=[scene, outmodel, outgallery]
            )
            min_conf_thr.release(
                fn=model_from_scene_fun,
                inputs=[
                    scene, min_conf_thr, as_pointcloud, mask_sky, clean_depth,
                    transparent_cams, cam_size, show_cam
                ],
                outputs=outmodel
            )
            cam_size.change(
                fn=model_from_scene_fun,
                inputs=[
                    scene, min_conf_thr, as_pointcloud, mask_sky, clean_depth,
                    transparent_cams, cam_size, show_cam
                ],
                outputs=outmodel
            )
            as_pointcloud.change(
                fn=model_from_scene_fun,
                inputs=[
                    scene, min_conf_thr, as_pointcloud, mask_sky, clean_depth,
                    transparent_cams, cam_size, show_cam
                ],
                outputs=outmodel
            )
            mask_sky.change(
                fn=model_from_scene_fun,
                inputs=[
                    scene, min_conf_thr, as_pointcloud, mask_sky, clean_depth,
                    transparent_cams, cam_size, show_cam
                ],
                outputs=outmodel
            )
            clean_depth.change(
                fn=model_from_scene_fun,
                inputs=[
                    scene, min_conf_thr, as_pointcloud, mask_sky, clean_depth,
                    transparent_cams, cam_size, show_cam
                ],
                outputs=outmodel
            )
            transparent_cams.change(
                model_from_scene_fun,
                inputs=[
                    scene, min_conf_thr, as_pointcloud, mask_sky, clean_depth,
                    transparent_cams, cam_size, show_cam
                ],
                outputs=outmodel
            )

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

    # Decide which weights to load
    if args.weights is not None and os.path.exists(args.weights):
        weights_path = args.weights
    else:
        weights_path = args.model_name

    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    # Use the provided output_dir or create a temporary directory
    tmpdirname = args.output_dir if args.output_dir is not None else tempfile.mkdtemp(suffix='monst3r_gradio_demo')

    if not args.silent:
        print('Outputting stuff in', tmpdirname)

    # If user provided --input_dir, run NON-INTERACTIVE inference
    if args.input_dir is not None:
        # 1) Gather the input files (images or single video)
        if os.path.isdir(args.input_dir):
            input_files = [
                os.path.join(args.input_dir, fname)
                for fname in sorted(os.listdir(args.input_dir))
            ]
        else:
            input_files = [args.input_dir]

        # 2) Prepare the partial function
        recon_fun = functools.partial(
            get_reconstructed_scene,
            args, tmpdirname, model, args.device, args.silent, args.image_size
        )

        # 3) Run inference once with command-line hyperparameters
        scene, outfile, imgs = recon_fun(
            filelist=input_files,
            schedule=args.schedule,
            niter=args.niter,
            min_conf_thr=args.min_conf_thr,
            as_pointcloud=args.as_pointcloud,
            mask_sky=args.mask_sky,
            clean_depth=args.clean_depth,
            transparent_cams=args.transparent_cams,
            cam_size=args.cam_size,
            show_cam=args.show_cam,
            scenegraph_type=args.scenegraph_type,
            winsize=args.winsize,
            refid=0,  # or make this an arg
            seq_name=args.seq_name,
            new_model_weights=args.weights,
            temporal_smoothing_weight=args.temporal_smoothing_weight,
            translation_weight=str(args.translation_weight),
            shared_focal=args.shared_focal,
            flow_loss_weight=args.flow_loss_weight,
            flow_loss_start_iter=args.flow_loss_start_iter,
            flow_loss_threshold=args.flow_loss_threshold,
            use_gt_mask=args.use_gt_mask,
            fps=args.fps,
            num_frames=args.num_frames,
        )

        # 4) Save folder/files with *argument-labeled* names

        # create a folder name that includes argument values
        filename_suffix = (
            f"sched_{args.schedule}"
            f"_niter{args.niter}"
            f"_minConf{args.min_conf_thr}"
            f"_{args.scenegraph_type}"
            f"_ws{args.winsize}"
            f"_temporalW{args.temporal_smoothing_weight}"
            f"_transW{args.translation_weight}"
            f"_flowW{args.flow_loss_weight}"
            f"_flowStart{args.flow_loss_start_iter}"
            f"_flowThre{args.flow_loss_threshold}"
            f"_useGTMask{args.use_gt_mask}"
            f"_fps{args.fps}"
            f"_nframes{args.num_frames}"
        )

        # Instead of just "demo_tmp/NULL", embed the suffix (plus seq_name if you want)
        save_folder = (
            f"{args.output_dir}/"
            f"{args.seq_name}_{filename_suffix}"
        )
        os.makedirs(save_folder, exist_ok=True)

        # Save TUM poses with a suffix
        poses_file = f"{save_folder}/pred_traj.txt"
        scene.save_tum_poses(poses_file)

        # Similarly for intrinsics
        intrinsics_file = f"{save_folder}/pred_intrinsics.txt"
        scene.save_intrinsics(intrinsics_file)

        # Depth maps and other items can also be saved to unique subfolders
        depth_folder = f"{save_folder}"
        scene.save_depth_maps(depth_folder)

        # (Continue saving whatever else you like, similarly naming them with `filename_suffix`.)
        
        if not args.silent:
            print(f"Processing completed. Output saved in {save_folder}")

    else:
        # Otherwise, launch interactive Gradio demo
        main_demo(tmpdirname, model, args.device, args.image_size,
                  server_name, args.server_port, silent=args.silent, args=args)
