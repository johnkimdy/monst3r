import itertools
import subprocess
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run parameter sweep for demo_auto2.py')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input video or directory of frames')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to store all outputs')
    return parser.parse_args()

def main():
    args = parse_args()
    
    hyperparameter_grid = {
        'schedule': ['linear', 'cosine'],   
        'niter': [300],
        'min_conf_thr': [1.0, 2.0],  
        'scenegraph_type': ['swinstride', 'complete'],
        'winsize': [2, 3, 5],  
        'temporal_smoothing_weight': [0.01],
        'translation_weight': [1.0],
        'flow_loss_weight': [0.05, 0.1],
        'flow_loss_start_iter': [0.1],  
        'flow_loss_threshold': [25],
        'use_gt_mask': [False],  
        'fps': [12, 14, 16],
        'num_frames': [100],
    }

    all_keys = list(hyperparameter_grid.keys())
    all_combinations = itertools.product(*(hyperparameter_grid[k] for k in all_keys))

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    for combo in all_combinations:
        # Build dictionary of param -> value for this combination
        params = dict(zip(all_keys, combo))

        filename_suffix = (
            f"sched_{params['schedule']}"
            f"_niter{params['niter']}"
            f"_minConf{params['min_conf_thr']}"
            f"_{params['scenegraph_type']}"
            f"_ws{params['winsize']}"
            f"_temporalW{params['temporal_smoothing_weight']}"
            f"_transW{params['translation_weight']}"
            f"_flowW{params['flow_loss_weight']}"
            f"_flowStart{params['flow_loss_start_iter']}"
            f"_flowThre{params['flow_loss_threshold']}"
            f"_useGTMask{params['use_gt_mask']}"
            f"_fps{params['fps']}"
            f"_nframes{params['num_frames']}"
        )

        SEQ_NAME = filename_suffix

        save_folder = f"{args.output_dir}/{SEQ_NAME}_{filename_suffix}"

        final_loss_file = f"{save_folder}/final_loss.txt"
        if os.path.isfile(final_loss_file):
            print(f"Skipping parameters {params} because {final_loss_file} already exists.")
            continue

        # Build command line
        cmd = [
            "python", "demo_auto2.py",
            f"--input_dir={args.input_dir}",
            f"--output_dir={args.output_dir}",
            f"--schedule={params['schedule']}",
            f"--niter={params['niter']}",
            f"--min_conf_thr={params['min_conf_thr']}",
            f"--scenegraph_type={params['scenegraph_type']}",
            f"--winsize={params['winsize']}",
            f"--temporal_smoothing_weight={params['temporal_smoothing_weight']}",
            f"--translation_weight={params['translation_weight']}",
            f"--flow_loss_weight={params['flow_loss_weight']}",
            f"--flow_loss_start_iter={params['flow_loss_start_iter']}",
            f"--flow_loss_threshold={params['flow_loss_threshold']}",
            f"--use_gt_mask={params['use_gt_mask']}",
            f"--fps={params['fps']}",
            f"--num_frames={params['num_frames']}",
            f"--seq_name={SEQ_NAME}",
            "--silent"  # optionally mute logs
        ]
        print("Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            # The call failed; log the error and continue with the next combo
            print(f"WARNING: Subprocess failed with exit code {e.returncode}. Error: {e}")
            continue

    print("Parameter sweep complete!")

if __name__ == "__main__":
    main()