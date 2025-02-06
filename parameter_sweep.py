# parameter_sweep.py
import itertools
import subprocess

hyperparameter_grid = {
    'schedule': ['linear', 'cosine'],   
    'niter': [300],
    'min_conf_thr': [1.0, 1.5, 2.0],  
    'scenegraph_type': ['swinstride', 'complete'],
    'winsize': [5],  
    'temporal_smoothing_weight': [0.01, 0.03],
    'translation_weight': [1.0],  
    'flow_loss_weight': [0.05, 0.1],
    'flow_loss_start_iter': [0.1],  
    'flow_loss_threshold': [25],
    'use_gt_mask': [False],  
    'fps': [12],
    'num_frames': [100],
    # and so on...
}

# Input video or directory of frames
INPUT_DIR = "demo_data/lady-running.mp4"

# Where to dump all outputs
OUTPUT_DIR = "./grid_search_outputs"

all_keys = list(hyperparameter_grid.keys())
all_combinations = itertools.product(*(hyperparameter_grid[k] for k in all_keys))

for combo in all_combinations:
    # Build dictionary of param -> value for this combination
    params = dict(zip(all_keys, combo))

    # Build command line
    cmd = [
        "python", "demo_auto.py",
        f"--input_dir={INPUT_DIR}",
        f"--output_dir={OUTPUT_DIR}",
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
        "--silent"  # optionally mute logs
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

print("Parameter sweep complete!")
