"""
Main script for optical flow based motion concept discovery.
Follows the same pipeline structure as the original discover_motion_concepts.py,
but uses RAFT optical flow as input instead of skeleton keypoints.

Pipeline:
  flow loading → subsampling (raw / feature) → clustering → medoid selection → labeling

Usage:
  python -m concept_discovery.motion_discovery.discover_flow_concepts \
      --dataset KTH \
      --anno_path datasets/KTH \
      --flow_path /path/to/kth_flow \
      --output_path concept_discovery/output/kth_raw \
      --approach raw \
      --clustering_mode partition \
      --use_partition_num -1 \
      --train_only
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime

# Add the project root to sys.path (so the script can be run from anywhere)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import concept_discovery.motion_discovery.flow_subsampling as flow_subsampling
import concept_discovery.motion_discovery.clustering as clustering
import concept_discovery.motion_discovery.labeling as labeling
import concept_discovery.utils.cluster_utils as util


parser = argparse.ArgumentParser(description='Optical flow based motion concept discovery')

# Path settings
parser.add_argument('--anno_path', default='', help='annotation directory (train.csv, val.csv, class_list.txt)')
parser.add_argument('--flow_path', default='', help='directory containing RAFT flow .npy files')
parser.add_argument('--output_path', default='', help='output directory')
parser.add_argument('--class_list', default='', help='path to the class list file (class_list.txt)')

# Approach settings
parser.add_argument('--approach', type=str, default='raw', choices=['raw', 'feature', 'patch', 'jpeg'],
                    help='raw: use the flow map as-is / feature: convert to low-dimensional vectors / patch: patch-based statistical features / jpeg: JPEG-compressed flow')
parser.add_argument('--grid_size', type=int, default=8,
                    help='(Approach B) spatial grid size (default: 8 → 128dim/frame)')
parser.add_argument('--spatial_stride', type=int, default=1,
                    help='(Approach A) spatial subsampling stride (default: 1=no subsampling, 4=56x56)')
parser.add_argument('--patch_size', type=int, default=16,
                    help='(Approach C) patch size (default: 16 → 14×14=196 patches for 224×224)')
parser.add_argument('--mag_threshold', type=float, default=None,
                    help='(Approach C) magnitude threshold. If set, flow values at or below this threshold are zeroed out (default: None=no threshold)')
parser.add_argument('--norm_mode', type=str, default='per_sample',
                    choices=['per_sample', 'global', 'none'],
                    help='(Approach C) normalization mode. per_sample: per-sample max magnitude / global: max magnitude over the entire dataset / none: no normalization')
parser.add_argument('--patch_feat_mode', type=str, default='dxdy',
                    choices=['dxdy', 'full'],
                    help='(Approach C) dxdy: mean_dx and mean_dy only (2f) / full: +mean_mag, std_mag (4f)')

# JPEG flow settings
parser.add_argument('--jpeg_bound', type=float, default=20.0,
                    help='(Approach D) clipping bound used during JPEG encoding (for the output folder name, default: 20)')

# Clustering settings
parser.add_argument('--clustering_mode', type=str, default='partition',
                    choices=['req', 'partition', 'k-means', 'som'],
                    help='clustering algorithm')
parser.add_argument('--use_partition_num', type=int, default=-1,
                    help='(FINCH partition mode) index of the partition to use. If -1, the partition with the highest NMI is selected automatically')
parser.add_argument('--req_cluster', type=int, default=0,
                    help='(req/k-means/som modes) desired number of clusters')

# Arguments kept for backward compatibility
parser.add_argument('--dataset', default='single_object_direction',
                    choices=['Penn_action', 'KTH', 'HAA100', 'UCF101', 'UCF_Sports', 'single_object_direction', 'multi_object', 'round_trip', 'ssv2_5k', 'ssv2_chiral'],
                    type=str)
parser.add_argument('--num_subsequence', type=int, default=10,
                    help='(legacy compatibility) number of subsequences used during subsampling')
parser.add_argument('--len_subsequence', type=int, default=16,
                    help='(legacy compatibility) subsequence length')
parser.add_argument('--save_fps', type=int, default=20)

# Temporal resampling (for datasets with varying frame counts)
parser.add_argument('--target_frames', type=int, default=None,
                    help='target number of frames. If set, all flows are temporally resampled to this frame count')
parser.add_argument('--target_size', type=int, default=None,
                    help='target spatial size (square). If set, all flows are resized to this size (e.g., 224)')
parser.add_argument('--temporal_sampling', type=str, default='uniform',
                    choices=['uniform', 'center'],
                    help='temporal sampling method. uniform: evenly spaced / center: contiguous center crop (default: uniform)')

# Train-only discovery
parser.add_argument('--train_only', action='store_true',
                    help='Run concept discovery on train data only; val samples are assigned via nearest medoid')


def concept_discovery(args):
    # Build the output directory
    if args.approach == 'feature':
        sub_dir = f"grid{args.grid_size}"
    elif args.approach == 'patch':
        thr_str = f"_thr{args.mag_threshold}" if args.mag_threshold is not None else ""
        feat_str = "2f" if args.patch_feat_mode == 'dxdy' else "4f"
        sub_dir = f"patch{args.patch_size}_{feat_str}_{args.norm_mode}{thr_str}"
    elif args.approach == 'jpeg':
        bound_str = f"b{int(args.jpeg_bound)}" if hasattr(args, 'jpeg_bound') and args.jpeg_bound else "b20"
        stride_str = f"_s{args.spatial_stride}" if hasattr(args, 'spatial_stride') and args.spatial_stride > 1 else ""
        sub_dir = f"jpeg_{bound_str}{stride_str}"
    else:
        sub_dir = "raw"
    output_path = os.path.join(args.output_path, args.approach, sub_dir)
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")

    # Timing records
    timing = {}
    pipeline_start = time.time()

    # Step 1: Flow loading and subsampling
    print(f"=" * 60)
    print(f"[Step 1] Flow subsampling (approach: {args.approach})")
    print(f"=" * 60)
    t0 = time.time()
    flow_subsampling.FlowDataset(args, output_path)
    timing['subsampling'] = time.time() - t0
    print(f"  ⏱ Subsampling time: {timing['subsampling']:.2f}s ({timing['subsampling']/60:.1f}min)")

    # Step 2: Clustering
    print(f"\n{'=' * 60}")
    print(f"[Step 2] Clustering (mode: {args.clustering_mode})")
    print(f"=" * 60)
    t0 = time.time()
    data, result_gt, num_concept = clustering.clustering(args, output_path)
    timing['clustering'] = time.time() - t0
    print(f"  ⏱ Clustering time: {timing['clustering']:.2f}s ({timing['clustering']/60:.1f}min)")

    # Step 3: Medoid selection + Labeling
    print(f"\n{'=' * 60}")
    print(f"[Step 3] Medoid & Labeling ({num_concept} concepts)")
    print(f"=" * 60)
    t0 = time.time()
    save_path = os.path.join(output_path, f"{num_concept}concepts_{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    util.save_cluster_info(data, result_gt, num_concept, output_path, save_path)
    closest_sample_indices = labeling.labeling(args, data, result_gt, save_path, output_path, num_concept, use_medoid=True)
    timing['labeling'] = time.time() - t0
    print(f"  ⏱ Labeling time: {timing['labeling']:.2f}s ({timing['labeling']/60:.1f}min)")

    timing['total'] = time.time() - pipeline_start

    # Timing summary + save
    print(f"\n{'=' * 60}")
    print(f"[Timing] Concept Discovery Summary")
    print(f"{'=' * 60}")
    print(f"  Approach        : {args.approach}")
    print(f"  Subsampling     : {timing['subsampling']:8.2f}s ({timing['subsampling']/60:.1f}min)")
    print(f"  Clustering      : {timing['clustering']:8.2f}s ({timing['clustering']/60:.1f}min)")
    print(f"  Labeling        : {timing['labeling']:8.2f}s ({timing['labeling']/60:.1f}min)")
    print(f"  ─────────────────────────────")
    print(f"  Total           : {timing['total']:8.2f}s ({timing['total']/60:.1f}min)")
    print(f"{'=' * 60}")

    # Save timing.json
    timing_path = os.path.join(save_path, "timing.json")
    timing['approach'] = args.approach
    timing['num_concepts'] = num_concept
    timing['timestamp'] = timestamp
    with open(timing_path, 'w') as f:
        json.dump(timing, f, indent=2)
    print(f"  Timing saved to: {timing_path}")

    print(f"\n[Done] Results saved to: {save_path}")
    print(f"  - video_attributes.json")
    print(f"  - hard_label_train.pkl")
    print(f"  - hard_label_val.pkl")
    print(f"  - concept_index.txt")
    print(f"  - cluster_centroids.npy")
    print(f"  - sequence_cluster_map.json")
    print(f"  - timing.json")
    print(f"=" * 60)


def concept_discovery_train_only(args):
    """
    Train-only concept discovery.
    - Define concepts by clustering the train data only
    - Assign val data to clusters via nearest medoid
    - Removes data leakage
    """
    import numpy as np
    from scipy.spatial.distance import cdist

    # Build the output directory
    if args.approach == 'feature':
        sub_dir = f"grid{args.grid_size}"
    elif args.approach == 'patch':
        thr_str = f"_thr{args.mag_threshold}" if args.mag_threshold is not None else ""
        feat_str = "2f" if args.patch_feat_mode == 'dxdy' else "4f"
        sub_dir = f"patch{args.patch_size}_{feat_str}_{args.norm_mode}{thr_str}"
    elif args.approach == 'jpeg':
        bound_str = f"b{int(args.jpeg_bound)}" if hasattr(args, 'jpeg_bound') and args.jpeg_bound else "b20"
        stride_str = f"_s{args.spatial_stride}" if hasattr(args, 'spatial_stride') and args.spatial_stride > 1 else ""
        sub_dir = f"jpeg_{bound_str}{stride_str}"
    else:
        sub_dir = "raw"
    output_path = os.path.join(args.output_path, args.approach, sub_dir)
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")

    # Timing records
    timing = {}
    pipeline_start = time.time()

    # ─── Step 1a: Train flow loading ───
    print(f"{'=' * 60}")
    print(f"[Step 1a] Train flow subsampling (approach: {args.approach})")
    print(f"{'=' * 60}")
    t0 = time.time()
    # Remove any existing cache (a mixed train+val cache may exist)
    cache_path = os.path.join(output_path, "processed_keypoints.npy")
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"  Removed old cache: {cache_path}")
    flow_subsampling.FlowDataset(args, output_path, splits=("train",))
    timing['train_subsampling'] = time.time() - t0
    print(f"  ⏱ Train subsampling time: {timing['train_subsampling']:.2f}s ({timing['train_subsampling']/60:.1f}min)")

    # ─── Step 2: Cluster train data only ───
    print(f"\n{'=' * 60}")
    print(f"[Step 2] Clustering TRAIN data only (mode: {args.clustering_mode})")
    print(f"{'=' * 60}")
    t0 = time.time()
    data_train, result_gt_train, num_concept = clustering.clustering(args, output_path)
    timing['clustering'] = time.time() - t0
    print(f"  ⏱ Clustering time: {timing['clustering']:.2f}s ({timing['clustering']/60:.1f}min)")

    # Preserve train metadata
    with open(os.path.join(output_path, "sample_metadata.json"), "r") as f:
        train_metadata = json.load(f)
    n_train = len(train_metadata)
    print(f"[INFO] Train samples: {n_train}, Clusters: {num_concept}")

    # ─── Step 3: Extract train medoids ───
    print(f"\n{'=' * 60}")
    print(f"[Step 3] Finding medoids from train clusters")
    print(f"{'=' * 60}")
    t0 = time.time()
    medoid_indices = util.find_medoid(data_train, result_gt_train)
    cluster_ids_sorted = sorted(medoid_indices.keys())
    medoid_features = np.array([data_train[medoid_indices[c]] for c in cluster_ids_sorted])
    timing['medoid'] = time.time() - t0
    print(f"  Medoid features shape: {medoid_features.shape}")
    for c in cluster_ids_sorted:
        idx = medoid_indices[c]
        print(f"  Cluster {c}: medoid sample = {train_metadata[idx]}")
    print(f"  ⏱ Medoid time: {timing['medoid']:.2f}s ({timing['medoid']/60:.1f}min)")

    # ─── Step 4: Val flow loading ───
    print(f"\n{'=' * 60}")
    print(f"[Step 4] Val flow subsampling → nearest medoid assignment")
    print(f"{'=' * 60}")
    t0 = time.time()
    val_output_path = os.path.join(output_path, "_val_temp")
    os.makedirs(val_output_path, exist_ok=True)
    # Remove the val cache as well
    val_cache = os.path.join(val_output_path, "processed_keypoints.npy")
    if os.path.exists(val_cache):
        os.remove(val_cache)
    flow_subsampling.FlowDataset(args, val_output_path, splits=("val",))

    # Load and flatten val data
    data_val, val_metadata = util.load_data(val_output_path)
    data_val = data_val.reshape(data_val.shape[0], -1)
    n_val = len(val_metadata)
    print(f"  Val samples: {n_val}, shape: {data_val.shape}")

    # Assign val samples to the nearest medoid
    distances = cdist(data_val, medoid_features)  # (N_val, num_concept)
    result_gt_val = np.array([cluster_ids_sorted[i] for i in distances.argmin(axis=1)])
    val_unique, val_counts = np.unique(result_gt_val, return_counts=True)
    print(f"  Val cluster assignment:")
    for c, cnt in zip(val_unique, val_counts):
        print(f"    Cluster {c}: {cnt} samples")
    timing['val_assignment'] = time.time() - t0
    print(f"  ⏱ Val assignment time: {timing['val_assignment']:.2f}s ({timing['val_assignment']/60:.1f}min)")

    # ─── Step 5: Labeling on the combined data ───
    print(f"\n{'=' * 60}")
    print(f"[Step 5] Medoid & Labeling ({num_concept} concepts, train_only mode)")
    print(f"{'=' * 60}")
    t0 = time.time()

    # Combined: train + val
    data_combined = np.concatenate([data_train, data_val], axis=0)
    result_gt_combined = np.concatenate([result_gt_train, result_gt_val], axis=0)
    metadata_combined = train_metadata + val_metadata

    # Overwrite sample_metadata.json with the combined metadata
    with open(os.path.join(output_path, "sample_metadata.json"), "w") as f:
        json.dump(metadata_combined, f, indent=4)

    save_path = os.path.join(output_path, f"{num_concept}concepts_{timestamp}_trainonly")
    os.makedirs(save_path, exist_ok=True)

    util.save_cluster_info(data_combined, result_gt_combined, num_concept, output_path, save_path)
    closest_sample_indices = labeling.labeling(
        args, data_combined, result_gt_combined, save_path, output_path, num_concept, use_medoid=True
    )
    timing['labeling'] = time.time() - t0
    print(f"  ⏱ Labeling time: {timing['labeling']:.2f}s ({timing['labeling']/60:.1f}min)")

    # ─── Clean up the temporary val directory ───
    import shutil
    shutil.rmtree(val_output_path, ignore_errors=True)

    timing['total'] = time.time() - pipeline_start

    # Timing summary + save
    print(f"\n{'=' * 60}")
    print(f"[Timing] Train-Only Concept Discovery Summary")
    print(f"{'=' * 60}")
    print(f"  Approach          : {args.approach}")
    print(f"  Train subsampling : {timing['train_subsampling']:8.2f}s ({timing['train_subsampling']/60:.1f}min)")
    print(f"  Clustering        : {timing['clustering']:8.2f}s ({timing['clustering']/60:.1f}min)")
    print(f"  Medoid extraction : {timing['medoid']:8.2f}s ({timing['medoid']/60:.1f}min)")
    print(f"  Val assignment    : {timing['val_assignment']:8.2f}s ({timing['val_assignment']/60:.1f}min)")
    print(f"  Labeling          : {timing['labeling']:8.2f}s ({timing['labeling']/60:.1f}min)")
    print(f"  ─────────────────────────────────")
    print(f"  Total             : {timing['total']:8.2f}s ({timing['total']/60:.1f}min)")
    print(f"{'=' * 60}")

    # Save timing.json
    timing_path = os.path.join(save_path, "timing.json")
    timing['approach'] = args.approach
    timing['num_concepts'] = num_concept
    timing['n_train'] = n_train
    timing['n_val'] = n_val
    timing['timestamp'] = timestamp
    timing['mode'] = 'train_only'
    with open(timing_path, 'w') as f:
        json.dump(timing, f, indent=2)
    print(f"  Timing saved to: {timing_path}")

    print(f"\n[Done] Train-only discovery results saved to: {save_path}")
    print(f"  - Train: {n_train} samples (clustered)")
    print(f"  - Val: {n_val} samples (nearest medoid)")
    print(f"  - hard_label_train.pkl")
    print(f"  - hard_label_val.pkl")
    print(f"  - sequence_cluster_map.json")
    print(f"  - timing.json")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    args = parser.parse_args()
    if args.train_only:
        concept_discovery_train_only(args)
    else:
        concept_discovery(args)
