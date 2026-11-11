"""
Optical flow based subsampling module.
Replaces the skeleton keypoint loading of the original subsampling.py with RAFT optical flow loading.

Input: .npy files extracted with RAFT (T-1, H, W, 2) — flow maps of consecutive frame pairs
Output: processed_keypoints.npy, sample_metadata.json (compatible with the existing pipeline)
"""

import numpy as np
import glob
import os
import sys
import json
import pandas as pd
from tqdm import tqdm
import cv2

# Add the project root to sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import concept_discovery.utils.cluster_utils as util


def load_optical_flow(flow_file, target_frames=None, target_size=None,
                       temporal_sampling='uniform'):
    """
    Load an optical flow file extracted with RAFT.
    If target_frames is given, temporal resampling adjusts the number of frames.
    If target_size is given, spatial resizing adjusts the resolution.

    Args:
        flow_file: path to the .npy file
        target_frames: (optional) target number of frames. If None, return the original as-is.
        target_size: (optional) target spatial size. An int (square) or an (H, W) tuple.
                     If None, return the original as-is. Flow values are rescaled accordingly.
        temporal_sampling: 'uniform' (evenly spaced sampling) or 'center' (contiguous center crop)
    Returns:
        flow_data: (T, H, W, 2) numpy array
        num_frames: number of frames
    """
    flow_data = np.load(flow_file)  # (T, H, W, 2)
    if target_frames is not None and flow_data.shape[0] != target_frames:
        orig_T = flow_data.shape[0]
        if orig_T < target_frames:
            # If there are too few frames, uniform upsample (repeat some frames)
            indices = np.linspace(0, orig_T - 1, target_frames, dtype=int)
            flow_data = flow_data[indices]
        elif temporal_sampling == 'center':
            # Contiguous center crop
            start = (orig_T - target_frames) // 2
            flow_data = flow_data[start:start + target_frames]
        else:
            # uniform: evenly spaced sampling
            indices = np.linspace(0, orig_T - 1, target_frames, dtype=int)
            flow_data = flow_data[indices]
    if target_size is not None:
        H, W = flow_data.shape[1], flow_data.shape[2]
        tH, tW = (target_size, target_size) if isinstance(target_size, int) else target_size
        if H != tH or W != tW:
            T = flow_data.shape[0]
            scale_w, scale_h = tW / W, tH / H
            resized = np.empty((T, tH, tW, 2), dtype=flow_data.dtype)
            for t in range(T):
                # cv2.resize takes (tW, tH) order and guarantees the exact target size
                resized[t, ..., 0] = cv2.resize(flow_data[t, ..., 0], (tW, tH), interpolation=cv2.INTER_LINEAR) * scale_w
                resized[t, ..., 1] = cv2.resize(flow_data[t, ..., 1], (tW, tH), interpolation=cv2.INTER_LINEAR) * scale_h
            flow_data = resized
    return flow_data, flow_data.shape[0]


def load_optical_flow_jpeg(flow_dir, target_frames=None, target_size=None,
                           temporal_sampling='uniform'):
    """
    Load optical flow stored as JPEG.
    Returns the data kept as-is in the uint8 [0,255] range (no float restoration).
    It is converted to [-1,1] later in the normalization step.

    Args:
        flow_dir: JPEG flow directory (a video_id folder). flow_x_00000.jpg, flow_y_00000.jpg, ...
        target_frames: (optional) target number of frames
        target_size: (optional) target spatial size
        temporal_sampling: 'uniform' (evenly spaced sampling) or 'center' (contiguous center crop)
    Returns:
        flow_data: (T, H, W, 2) numpy array (float32, [0,255])
        num_frames: number of frames
    """
    # Determine the number of frames from the list of flow_x files
    x_files = sorted(glob.glob(os.path.join(flow_dir, "flow_x_*.jpg")))
    if not x_files:
        return np.array([]), 0

    frames = []
    for x_file in x_files:
        y_file = x_file.replace("flow_x_", "flow_y_")
        dx = cv2.imread(x_file, cv2.IMREAD_GRAYSCALE)  # (H, W), uint8
        dy = cv2.imread(y_file, cv2.IMREAD_GRAYSCALE)   # (H, W), uint8
        if dx is None or dy is None:
            continue
        frames.append(np.stack([dx, dy], axis=-1))  # (H, W, 2)

    if not frames:
        return np.array([]), 0

    flow_data = np.stack(frames).astype(np.float32)  # (T, H, W, 2), float32 [0,255]

    # temporal sampling
    if target_frames is not None and flow_data.shape[0] != target_frames:
        orig_T = flow_data.shape[0]
        if orig_T < target_frames:
            # If there are too few frames, uniform upsample (repeat some frames)
            indices = np.linspace(0, orig_T - 1, target_frames, dtype=int)
            flow_data = flow_data[indices]
        elif temporal_sampling == 'center':
            # Contiguous center crop
            start = (orig_T - target_frames) // 2
            flow_data = flow_data[start:start + target_frames]
        else:
            # uniform: evenly spaced sampling
            indices = np.linspace(0, orig_T - 1, target_frames, dtype=int)
            flow_data = flow_data[indices]

    # spatial resize
    if target_size is not None:
        H, W = flow_data.shape[1], flow_data.shape[2]
        tH, tW = (target_size, target_size) if isinstance(target_size, int) else target_size
        if H != tH or W != tW:
            T = flow_data.shape[0]
            resized = np.empty((T, tH, tW, 2), dtype=flow_data.dtype)
            for t in range(T):
                resized[t, ..., 0] = cv2.resize(flow_data[t, ..., 0], (tW, tH), interpolation=cv2.INTER_LINEAR)
                resized[t, ..., 1] = cv2.resize(flow_data[t, ..., 1], (tW, tH), interpolation=cv2.INTER_LINEAR)
            flow_data = resized
            # uint8 values are in the [0,255] range, not pixel displacements,
            # so no flow value rescaling is needed on spatial resize (handled during normalization)

    return flow_data, flow_data.shape[0]


def normalize_jpeg_flow(clip):
    """
    Normalize JPEG flow (uint8 [0,255]).
    Convert directly to the [-1, 1] range without restoration.
    uint8_val=0 → -1, uint8_val=128 → 0, uint8_val=255 → +1

    Args:
        clip: (T, H, W, 2) float32 [0, 255]
    Returns:
        normalized: (T, H, W, 2) float32 [-1, 1]
    """
    return (clip / 255.0) * 2.0 - 1.0


def normalize_flow_raw(clip):
    """
    Approach A: Raw flow normalization.
    Normalize to the [-1, 1] range by dividing by the max magnitude of the entire clip.

    Args:
        clip: (T, H, W, 2)
    Returns:
        normalized clip: (T, H, W, 2)
    """
    magnitude = np.sqrt(clip[..., 0]**2 + clip[..., 1]**2)
    max_mag = magnitude.max() + 1e-8
    return clip / max_mag


def extract_flow_features(clip, grid_size=8):
    """
    Approach B: Flow map → low-dimensional vector conversion.
    Split into a spatial grid and compute the mean flow of each cell.

    Args:
        clip: (T, H, W, 2)
        grid_size: grid partition size (default: 8 → 8×8=64 cells → 128 dim/frame)
    Returns:
        features: (T, grid_size*grid_size*2)
    """
    T, H, W, C = clip.shape
    gh = H // grid_size
    gw = W // grid_size
    features = []

    for t in range(T):
        flow = clip[t]  # (H, W, 2)
        frame_feats = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell = flow[i*gh:(i+1)*gh, j*gw:(j+1)*gw]  # (gh, gw, 2)
                frame_feats.append(cell[..., 0].mean())  # mean dx
                frame_feats.append(cell[..., 1].mean())  # mean dy
        features.append(frame_feats)

    return np.array(features)  # (T, grid_size*grid_size*2)


def extract_patch_features(clip, patch_size=16, mag_threshold=0.0, patch_feat_mode='dxdy'):
    """
    Approach C: Patch-based flow statistics feature extraction (vectorized).
    Split each frame into patch_size×patch_size patches and compute per-patch statistics.

    Args:
        clip: (T, H, W, 2) — normalized flow
        patch_size: patch size (default: 16 → 14×14=196 patches for 224×224)
        mag_threshold: magnitude threshold. Flow at or below this value is set to 0 (default: 0.0=no threshold)
        patch_feat_mode: 'dxdy' (2 features: mean_dx, mean_dy) or 'full' (4 features: +mean_mag, std_mag)
    Returns:
        features: (T, n_patches_h * n_patches_w * F)
                  dxdy: F=2, full: F=4
    """
    T, H, W, _ = clip.shape
    n_ph = H // patch_size  # 14 for 224/16
    n_pw = W // patch_size  # 14

    # crop to exact patch grid (discard leftover pixels)
    clip = clip[:, :n_ph*patch_size, :n_pw*patch_size, :]

    dx = clip[..., 0]  # (T, H', W')
    dy = clip[..., 1]

    # Apply the threshold: set pixels with magnitude at or below the threshold to 0
    if mag_threshold > 0:
        mag = np.sqrt(dx**2 + dy**2)
        mask = mag <= mag_threshold
        dx = dx.copy()
        dy = dy.copy()
        dx[mask] = 0.0
        dy[mask] = 0.0

    # reshape → (T, n_ph, patch_size, n_pw, patch_size) → mean over patch dims
    dx_patches = dx.reshape(T, n_ph, patch_size, n_pw, patch_size).mean(axis=(2, 4))  # (T, n_ph, n_pw)
    dy_patches = dy.reshape(T, n_ph, patch_size, n_pw, patch_size).mean(axis=(2, 4))

    if patch_feat_mode == 'full':
        mag_all = np.sqrt(
            clip[..., 0]**2 + clip[..., 1]**2
        ).reshape(T, n_ph, patch_size, n_pw, patch_size)
        mag_mean = mag_all.mean(axis=(2, 4))  # (T, n_ph, n_pw)
        mag_std = mag_all.std(axis=(2, 4))
        # stack: (T, n_ph, n_pw, 4) → (T, n_ph*n_pw*4)
        features = np.stack([dx_patches, dy_patches, mag_mean, mag_std], axis=-1)
    else:
        # stack: (T, n_ph, n_pw, 2) → (T, n_ph*n_pw*2)
        features = np.stack([dx_patches, dy_patches], axis=-1)

    return features.reshape(T, -1)


def subsampling_flow_raw(args, flow_files):
    """
    Approach A: Use the raw optical flow as-is, flattened.
    Treat the entire flow of each video as a single sample.

    Args:
        args: argparse namespace
        flow_files: list of flow .npy file paths
    Returns:
        class_data: list of numpy arrays
        class_metadata: list of video IDs
    """
    class_data = []
    class_metadata = []
    target_frames = getattr(args, 'target_frames', None)
    target_size = getattr(args, 'target_size', None)
    temporal_sampling = getattr(args, 'temporal_sampling', 'uniform')

    for flow_file in tqdm(flow_files, desc="[Approach A] Loading raw flows"):
        video_id = os.path.basename(flow_file).replace(".npy", "")

        flow_data, num_frames = load_optical_flow(flow_file, target_frames=target_frames, target_size=target_size,
                                                   temporal_sampling=temporal_sampling)

        if num_frames == 0:
            print(f"Skipping {flow_file} (empty)")
            continue

        # The entire flow as a single sample — normalize, then spatial subsampling
        normalized = normalize_flow_raw(flow_data)
        spatial_stride = getattr(args, 'spatial_stride', 1)
        if spatial_stride > 1:
            normalized = normalized[:, ::spatial_stride, ::spatial_stride, :]
        class_data.append(normalized)
        class_metadata.append(video_id)

    return class_data, class_metadata


def subsampling_flow_feature(args, flow_files):
    """
    Approach B: Extract and use low-dimensional features from the optical flow.
    Extract grid-based features from the entire flow of each video.

    Args:
        args: argparse namespace (grid_size, etc.)
        flow_files: list of flow .npy file paths
    Returns:
        class_data: list of numpy arrays
        class_metadata: list of video IDs
    """
    class_data = []
    class_metadata = []
    grid_size = getattr(args, 'grid_size', 8)
    target_frames = getattr(args, 'target_frames', None)
    target_size = getattr(args, 'target_size', None)
    temporal_sampling = getattr(args, 'temporal_sampling', 'uniform')

    for flow_file in tqdm(flow_files, desc="[Approach B] Extracting flow features"):
        video_id = os.path.basename(flow_file).replace(".npy", "")

        flow_data, num_frames = load_optical_flow(flow_file, target_frames=target_frames, target_size=target_size,
                                                   temporal_sampling=temporal_sampling)

        if num_frames == 0:
            print(f"Skipping {flow_file} (empty)")
            continue

        features = extract_flow_features(flow_data, grid_size)  # (T, feat_dim)
        class_data.append(features)
        class_metadata.append(video_id)

    return class_data, class_metadata


def subsampling_flow_patch(args, flow_files):
    """
    Approach C: Patch-based flow statistics features.
    normalize (optional) → split into patches → extract per-patch statistics.

    Args:
        args: argparse namespace (patch_size, mag_threshold, norm_mode, patch_feat_mode, etc.)
        flow_files: list of flow .npy file paths
    Returns:
        class_data: list of numpy arrays, each element of shape: (T, n_patches * F)
        class_metadata: list of video IDs
    """
    class_data = []
    class_metadata = []
    patch_size = getattr(args, 'patch_size', 16)
    mag_threshold = getattr(args, 'mag_threshold', None)
    norm_mode = getattr(args, 'norm_mode', 'per_sample')
    patch_feat_mode = getattr(args, 'patch_feat_mode', 'dxdy')

    # global normalize: compute the max magnitude over the entire dataset in a 1st pass
    global_max_mag = None
    if norm_mode == 'global':
        print("[Approach C] Computing global max magnitude (1st pass)...")
        global_max_mag = 0.0
        for flow_file in tqdm(flow_files, desc="  scanning"):
            flow_data = np.load(flow_file)
            mag = np.sqrt(flow_data[..., 0]**2 + flow_data[..., 1]**2)
            sample_max = mag.max()
            if sample_max > global_max_mag:
                global_max_mag = sample_max
        global_max_mag = global_max_mag + 1e-8
        print(f"  global max magnitude: {global_max_mag:.4f}")

    target_frames = getattr(args, 'target_frames', None)
    target_size = getattr(args, 'target_size', None)
    temporal_sampling = getattr(args, 'temporal_sampling', 'uniform')

    for flow_file in tqdm(flow_files, desc="[Approach C] Extracting patch features"):
        video_id = os.path.basename(flow_file).replace(".npy", "")

        flow_data, num_frames = load_optical_flow(flow_file, target_frames=target_frames, target_size=target_size,
                                                   temporal_sampling=temporal_sampling)

        if num_frames == 0:
            print(f"Skipping {flow_file} (empty)")
            continue

        # normalize
        if norm_mode == 'per_sample':
            normalized = normalize_flow_raw(flow_data)
        elif norm_mode == 'global':
            normalized = flow_data / global_max_mag
        else:  # 'none'
            normalized = flow_data

        # Extract patch features
        thr = mag_threshold if mag_threshold is not None else 0.0
        features = extract_patch_features(normalized, patch_size, thr, patch_feat_mode)
        class_data.append(features)
        class_metadata.append(video_id)

    return class_data, class_metadata


def subsampling_flow_jpeg(args, flow_dirs):
    """
    Approach D: JPEG compressed flow.
    Load JPEG uint8 [0,255] → normalize to [-1,1] → use the raw flow as-is.
    Normalize directly without float restoration. Use the entire flow as the feature without patch splitting.

    Args:
        args: argparse namespace (spatial_stride, etc.)
        flow_dirs: list of JPEG flow directories (each directory = 1 video)
    Returns:
        class_data: list of numpy arrays
        class_metadata: list of video IDs
    """
    class_data = []
    class_metadata = []
    target_frames = getattr(args, 'target_frames', None)
    target_size = getattr(args, 'target_size', None)
    temporal_sampling = getattr(args, 'temporal_sampling', 'uniform')

    for flow_dir in tqdm(flow_dirs, desc="[Approach D] Loading JPEG flows"):
        video_id = os.path.basename(flow_dir)

        flow_data, num_frames = load_optical_flow_jpeg(
            flow_dir, target_frames=target_frames, target_size=target_size,
            temporal_sampling=temporal_sampling
        )

        if num_frames == 0:
            print(f"Skipping {flow_dir} (empty)")
            continue

        # Direct uint8 [0,255] → [-1, 1] conversion (no float restoration needed)
        normalized = normalize_jpeg_flow(flow_data)

        # spatial subsampling (optional)
        spatial_stride = getattr(args, 'spatial_stride', 1)
        if spatial_stride > 1:
            normalized = normalized[:, ::spatial_stride, ::spatial_stride, :]

        class_data.append(normalized)
        class_metadata.append(video_id)

    return class_data, class_metadata


def load_flow_files_jpeg(flow_path, anno_path, splits=("train", "val")):
    """
    Return the list of video directories under the JPEG flow directory that match the annotations.

    Args:
        flow_path: JPEG flow root directory (containing video_id/ subfolders)
        anno_path: directory containing train.csv and val.csv
        splits: tuple of splits to load
    Returns:
        flow_dirs: sorted list of JPEG flow directories
    """
    # Collect valid video IDs from the CSVs
    valid_ids = set()
    for split in splits:
        csv_f = os.path.join(anno_path, f"{split}.csv")
        if not os.path.exists(csv_f):
            print(f"Warning: {csv_f} not found")
            continue
        df = pd.read_csv(csv_f, header=None, names=["video_name", "class_label"],
                         sep=r'[\s,]+', engine='python')
        for name in df["video_name"].tolist():
            video_id = os.path.splitext(str(name))[0]
            valid_ids.add(video_id)

    print(f"[INFO] Valid video IDs from CSV ({','.join(splits)}): {len(valid_ids)}")

    # Match JPEG directories (each video_id folder; subfolders are searched too)
    all_dirs = sorted([
        d for d in glob.glob(os.path.join(flow_path, "*"))
        if os.path.isdir(d) and os.path.basename(d) not in ('train', 'val', 'test')
    ])
    # Also search video_id folders inside subfolders (train/, val/, etc.)
    for sub in ('train', 'val', 'test'):
        sub_path = os.path.join(flow_path, sub)
        if os.path.isdir(sub_path):
            all_dirs += sorted([
                d for d in glob.glob(os.path.join(sub_path, "*"))
                if os.path.isdir(d)
            ])
    all_dirs = sorted(set(all_dirs))
    print(f"[INFO] Total JPEG flow dirs found: {len(all_dirs)}")

    filtered = []
    for d in all_dirs:
        vid = os.path.basename(d)
        if vid in valid_ids:
            filtered.append(d)

    print(f"[INFO] Matched JPEG flow dirs: {len(filtered)}/{len(all_dirs)}")

    if not filtered:
        print("[WARN] No matching JPEG flow dirs found! Using all available dirs.")
        filtered = all_dirs

    return filtered


def load_flow_files(flow_path, anno_path, splits=("train", "val")):
    """
    Load only the flow files that correspond to videos listed in the annotation CSVs.

    Args:
        flow_path: directory containing the flow .npy files
        anno_path: directory containing train.csv and val.csv
        splits: tuple of splits to load (default: ("train", "val"); use ("train",) for train only)
    Returns:
        flow_files: sorted list of flow file paths
    """
    # Collect valid video IDs from the CSVs
    valid_ids = set()
    for split in splits:
        csv_f = os.path.join(anno_path, f"{split}.csv")
        if not os.path.exists(csv_f):
            print(f"Warning: {csv_f} not found")
            continue
        df = pd.read_csv(csv_f, header=None, names=["video_name", "class_label"],
                         sep=r'[\s,]+', engine='python')
        for name in df["video_name"].tolist():
            video_id = os.path.splitext(str(name))[0]  # strip the .mp4 extension
            valid_ids.add(video_id)

    print(f"[INFO] Valid video IDs from CSV ({','.join(splits)}): {len(valid_ids)}")

    # Match flow files (subfolders are searched too: flow_path/*.npy + flow_path/**/*.npy)
    all_flows = sorted(glob.glob(os.path.join(flow_path, "*.npy")))
    # Also search for files located in subfolders (train/, val/, etc.)
    all_flows += sorted(glob.glob(os.path.join(flow_path, "**", "*.npy"), recursive=True))
    # Remove duplicates
    all_flows = sorted(set(all_flows))
    print(f"[INFO] Total flow files found: {len(all_flows)}")

    # For datasets like UCF Sports: build a reverse mapping for the case where flow files
    # are named "Class__SeqNum.npy" while the CSV uses the "Class/SeqNum/filename.avi" form
    # CSV video_id "Diving-Side/004/2538-16_70032" → short key "Diving-Side__004"
    short_to_valid = {}
    for vid_id in valid_ids:
        parts = vid_id.split("/")
        if len(parts) >= 2:
            short_key = f"{parts[0]}__{parts[1]}"
            short_to_valid[short_key] = vid_id

    filtered = []
    for f in all_flows:
        vid = os.path.basename(f).replace(".npy", "")
        if vid in valid_ids:
            filtered.append(f)
        elif vid in short_to_valid:
            filtered.append(f)

    print(f"[INFO] Matched flow files: {len(filtered)}/{len(all_flows)}")

    if not filtered:
        print("[WARN] No matching flow files found! Using all available flows.")
        filtered = all_flows

    return filtered


def FlowDataset(args, output_path, splits=("train", "val")):
    """
    Main entry point: load the flow files, subsample them, and save the result.
    Counterpart of the original Keypointset() function.

    Args:
        args: argparse namespace
              - flow_path: flow .npy directory
              - anno_path: annotation CSV directory
              - approach: 'raw' or 'feature'
              - grid_size: (Approach B) grid size
        output_path: output directory
        splits: tuple of splits to load (default: ("train", "val"))
    """
    processed_path = os.path.join(output_path, "processed_keypoints.npy")
    if os.path.exists(processed_path):
        print(f"✅ {processed_path} exists, skipping FlowDataset()")
        return

    if args.approach == "jpeg":
        flow_dirs = load_flow_files_jpeg(args.flow_path, args.anno_path, splits=splits)
        class_data, class_metadata = subsampling_flow_jpeg(args, flow_dirs)
    else:
        flow_files = load_flow_files(args.flow_path, args.anno_path, splits=splits)

        if args.approach == "raw":
            class_data, class_metadata = subsampling_flow_raw(args, flow_files)
        elif args.approach == "feature":
            class_data, class_metadata = subsampling_flow_feature(args, flow_files)
        elif args.approach == "patch":
            class_data, class_metadata = subsampling_flow_patch(args, flow_files)
        else:
            raise ValueError(f"Unknown approach: {args.approach}. Use 'raw', 'feature', 'patch', or 'jpeg'.")

    print(f"Total samples: {len(class_data)}")
    print(f"Sample shape: {np.array(class_data[0]).shape}")

    util.save_data(output_path, class_data, class_metadata)
