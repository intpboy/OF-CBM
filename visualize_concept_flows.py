"""
Concept Medoid Flow Visualization
==================================
Visualize each medoid concept's optical flow as a strip along the time axis.
HSV color wheel: hue = direction, value (brightness) = magnitude

Usage:
    python visualize_concept_flows.py \
        --dataset kth \
        --concept_dir concept_discovery/output/kth_raw/raw/raw/<concepts_dir_trainonly> \
        --flow_path flows/kth \
        --output_path figures/kth_concept_flows.png \
        --num_concepts 10 \
        --num_frames 8
"""

import argparse
import json
import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import hsv_to_rgb


def flow_to_rgb(flow_frame, max_mag=None, gamma=0.4, noise_thresh=0.05):
    """
    Optical flow (H, W, 2) → RGB image (H, W, 3)
    HSV color wheel: hue=direction, saturation=1, value=magnitude

    gamma < 1.0: non-linear mapping so that small motions also appear bright (default 0.4)
    Background (no motion) is black; stronger motion appears brighter
    """
    dx, dy = flow_frame[..., 0], flow_frame[..., 1]
    mag = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)  # -pi ~ pi

    # Normalize
    hue = (angle + np.pi) / (2 * np.pi)  # 0 ~ 1
    sat = np.ones_like(hue)
    if max_mag is None:
        max_mag = mag.max() if mag.max() > 0 else 1.0

    # Below the noise threshold: black; above it: amplified via gamma correction
    norm_mag = np.clip(mag / max_mag, 0, 1)
    # Treat values below noise_thresh as background (black)
    val = np.where(norm_mag < noise_thresh, 0.0, np.power(norm_mag, gamma))

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return (rgb * 255).astype(np.uint8), max_mag


def load_medoid_info(concept_dir):
    """concept_index.txt + sample_metadata.json → medoid video IDs"""
    concept_index_path = os.path.join(concept_dir, 'concept_index.txt')
    # sample_metadata.json lives in the parent directory of concept_dir
    meta_path = os.path.join(os.path.dirname(concept_dir), 'sample_metadata.json')
    if not os.path.exists(meta_path):
        # One more level up
        meta_path = os.path.join(os.path.dirname(os.path.dirname(concept_dir)), 'sample_metadata.json')
    if not os.path.exists(meta_path):
        # Top level of output_path
        parts = concept_dir.split('/')
        for i in range(len(parts), 0, -1):
            candidate = os.path.join('/'.join(parts[:i]), 'sample_metadata.json')
            if os.path.exists(candidate):
                meta_path = candidate
                break

    print(f"Loading concept_index: {concept_index_path}")
    print(f"Loading sample_metadata: {meta_path}")

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    medoids = {}
    with open(concept_index_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                concept_id, sample_idx = line.split(':')
                concept_id = int(concept_id.strip())
                sample_idx = int(sample_idx.strip())
                if sample_idx < len(metadata):
                    video_id = metadata[sample_idx]
                    # Strip the subsequence suffix (e.g., person01_boxing_d1_0 → person01_boxing_d1)
                    medoids[concept_id] = {
                        'sample_idx': sample_idx,
                        'video_id': video_id,
                        'full_id': video_id
                    }
    return medoids


def find_flow_file(video_id, flow_path):
    """Find the flow .npy file matching video_id.
    Returns: (flow_file_path, subseq_idx or None)
    """
    # Direct match (when per-subsequence files exist)
    direct = os.path.join(flow_path, f"{video_id}.npy")
    if os.path.exists(direct):
        return direct, None

    # Try again after stripping the subsequence suffix (e.g., person01_boxing_d1_0 → person01_boxing_d1)
    parts = video_id.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        base = parts[0]
        subseq_idx = int(parts[1])
        candidate = os.path.join(flow_path, f"{base}.npy")
        if os.path.exists(candidate):
            return candidate, subseq_idx

    # Search subdirectories
    for root, dirs, files in os.walk(flow_path):
        for fname in files:
            if fname.replace('.npy', '') == video_id:
                return os.path.join(root, fname), None

    return None, None


def load_and_subsample_flow(flow_file, num_display_frames=8,
                            pipeline_frames=30, pipeline_sampling='center',
                            target_size=112, subseq_idx=None):
    """Load flow → preprocess identically to the pipeline → sample frames for visualization

    Stage 1: reproduce the pipeline (center crop / uniform → pipeline_frames frames)
    Stage 2: uniform sampling for visualization (pipeline_frames → num_display_frames)
    """
    flow = np.load(flow_file)  # (T, H, W, 2)
    T = flow.shape[0]

    # If a subsequence index is given, extract that segment
    if subseq_idx is not None and T > pipeline_frames:
        seg_len = T // max(1, (T // pipeline_frames))
        start = subseq_idx * seg_len
        end = min(start + seg_len, T)
        if start < T:
            flow = flow[start:end]
            T = flow.shape[0]

    # Stage 1: temporal sampling identical to the pipeline
    if T > pipeline_frames:
        if pipeline_sampling == 'center':
            start = (T - pipeline_frames) // 2
            flow = flow[start:start + pipeline_frames]
        else:  # uniform
            indices = np.linspace(0, T - 1, pipeline_frames, dtype=int)
            flow = flow[indices]
    elif T < pipeline_frames:
        indices = np.linspace(0, T - 1, pipeline_frames, dtype=int)
        flow = flow[indices]

    # Stage 2: uniform sampling for visualization (30 frames → 8 frames)
    T = flow.shape[0]
    if T > num_display_frames:
        indices = np.linspace(0, T - 1, num_display_frames, dtype=int)
        flow = flow[indices]

    # Spatial resize
    if flow.shape[1] != target_size or flow.shape[2] != target_size:
        from scipy.ndimage import zoom as scipy_zoom
        h, w = flow.shape[1], flow.shape[2]
        scale_h = target_size / h
        scale_w = target_size / w
        flow = scipy_zoom(flow, (1, scale_h, scale_w, 1), order=1)

    return flow


def load_penn_action_label_map(anno_dir):
    """Penn Action: load the video_id → class name mapping"""
    class_list_path = os.path.join(anno_dir, 'class_list.txt')
    train_csv_path = os.path.join(anno_dir, 'train.csv')
    val_csv_path = os.path.join(anno_dir, 'val.csv')

    # class index → class name
    classes = {}
    with open(class_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                classes[len(classes)] = line

    # video_id → class name
    vid_to_class = {}
    for csv_path in [train_csv_path, val_csv_path]:
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    vid = parts[0].replace('.mp4', '').replace('.avi', '')
                    label_idx = int(parts[1])
                    if label_idx in classes:
                        vid_to_class[vid] = classes[label_idx]

    return vid_to_class


# Global cache (loaded only once)
_penn_label_cache = None


def get_class_from_video_id(video_id, dataset, penn_anno_dir=None):
    """Extract the class name from video_id"""
    global _penn_label_cache

    if dataset.lower() == 'kth':
        # person01_boxing_d1_0 → boxing
        parts = video_id.split('_')
        if len(parts) >= 2:
            return parts[1]
    elif dataset.lower() in ('penn_action', 'penn-action'):
        if _penn_label_cache is None and penn_anno_dir:
            _penn_label_cache = load_penn_action_label_map(penn_anno_dir)
            print(f"  Loaded Penn Action label map: {len(_penn_label_cache)} videos")
        if _penn_label_cache:
            # Strip the subsequence suffix from video_id: 0001_0 → 0001
            base_id = video_id.rsplit('_', 1)[0] if '_' in video_id and video_id.rsplit('_', 1)[1].isdigit() else video_id
            cls = _penn_label_cache.get(base_id) or _penn_label_cache.get(video_id)
            if cls:
                return cls
        return video_id
    return video_id


def select_diverse_concepts(medoids, dataset, num_concepts=10, sequence_cluster_map_path=None):
    """Select concepts spanning diverse classes"""
    # Group by class
    class_concepts = {}
    for cid, info in medoids.items():
        cls = get_class_from_video_id(info['video_id'], dataset)
        if cls not in class_concepts:
            class_concepts[cls] = []
        class_concepts[cls].append(cid)

    # Round-robin selection: one per class at a time
    selected = []
    class_list = sorted(class_concepts.keys())
    idx = 0
    while len(selected) < num_concepts and idx < 1000:
        for cls in class_list:
            concepts = class_concepts[cls]
            cls_idx = idx // len(class_list)
            if cls_idx < len(concepts) and len(selected) < num_concepts:
                selected.append(concepts[cls_idx])
        idx += len(class_list)

    return sorted(selected)


def create_flow_color_wheel(size=60, gamma=0.4):
    """Create the color wheel legend (same gamma setting as flow_to_rgb)"""
    Y, X = np.mgrid[-size//2:size//2, -size//2:size//2]
    angle = np.arctan2(Y, X)
    mag = np.sqrt(X**2 + Y**2)
    mask = mag <= size // 2

    hue = (angle + np.pi) / (2 * np.pi)
    sat = np.ones_like(hue)
    norm_mag = np.clip(mag / (size // 2), 0, 1)
    val = np.power(norm_mag, gamma)

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = hsv_to_rgb(hsv)
    rgb[~mask] = 0.0  # Black background
    return rgb


def visualize_concepts(medoids, selected_concepts, flow_path, dataset,
                       output_path, num_frames=8, target_size=224,
                       pipeline_frames=30, pipeline_sampling='center',
                       gamma=0.4, noise_thresh=0.05):
    """Main visualization: lay out each concept's flow as a strip along the time axis"""

    n_concepts = len(selected_concepts)
    fig_width = num_frames * 1.5 + 3  # Including extra margin
    fig_height = n_concepts * 1.5 + 1

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)

    # GridSpec: labels on the left, flow strips on the right
    gs = gridspec.GridSpec(n_concepts, num_frames + 1,
                           width_ratios=[0.3] + [1] * num_frames,
                           wspace=0.02, hspace=0.15,
                           left=0.02, right=0.98, top=0.95, bottom=0.05)

    # Compute the global max magnitude first (for a consistent color scale)
    global_max_mag = 0
    flow_cache = {}
    for cid in selected_concepts:
        info = medoids[cid]
        flow_file, subseq_idx = find_flow_file(info['video_id'], flow_path)
        if flow_file is None:
            print(f"  [WARN] Flow not found for concept {cid}: {info['video_id']}")
            continue
        flow = load_and_subsample_flow(flow_file, num_display_frames=num_frames,
                                       pipeline_frames=pipeline_frames,
                                       pipeline_sampling=pipeline_sampling,
                                       target_size=target_size, subseq_idx=subseq_idx)
        flow_cache[cid] = flow
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        global_max_mag = max(global_max_mag, np.percentile(mag, 98))

    if global_max_mag == 0:
        global_max_mag = 1.0

    print(f"Global max magnitude (98th pct): {global_max_mag:.2f}")
    print(f"Visualizing {len(flow_cache)} concepts...")

    row = 0
    for cid in selected_concepts:
        if cid not in flow_cache:
            continue
        flow = flow_cache[cid]
        info = medoids[cid]
        cls = get_class_from_video_id(info['video_id'], dataset)

        # Label: concept ID + class name (black background → white text)
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.set_facecolor('black')
        ax_label.text(0.9, 0.55, f"C{cid}",
                      ha='right', va='center', fontsize=8, fontweight='bold',
                      color='#FFFFFF', transform=ax_label.transAxes)
        ax_label.text(0.9, 0.30, f"{cls}",
                      ha='right', va='center', fontsize=6, fontstyle='italic',
                      color='#AAAAAA', transform=ax_label.transAxes)
        ax_label.axis('off')

        # Flow frames
        for t in range(min(num_frames, flow.shape[0])):
            ax = fig.add_subplot(gs[row, t + 1])
            rgb, _ = flow_to_rgb(flow[t], max_mag=global_max_mag,
                                 gamma=gamma, noise_thresh=noise_thresh)
            ax.imshow(rgb)
            ax.axis('off')
            if row == 0:
                ax.set_title(f"t={t}", fontsize=6, pad=2, color='white')

        row += 1

    # Title + timestamp
    dataset_name = dataset.upper() if dataset.lower() == 'kth' else 'Penn Action'
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.suptitle(f"OF-CBM Concept Medoid Flows — {dataset_name}\n"
                 f"(HSV: hue=direction, brightness=magnitude)  |  Generated: {now_str}",
                 fontsize=10, fontweight='bold', y=0.99, color='white')

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', facecolor='black')
    print(f"\n✅ Saved: {output_path}")
    plt.close(fig)


def save_individual_frames(medoids, selected_concepts, flow_path, dataset,
                           output_dir, num_frames=8, target_size=224,
                           pipeline_frames=30, pipeline_sampling='center',
                           gamma=0.4, noise_thresh=0.05):
    """Create a folder per concept and save the optical flow at each t as individual PNGs"""

    # Compute the global max magnitude first
    global_max_mag = 0
    flow_cache = {}
    for cid in selected_concepts:
        info = medoids[cid]
        flow_file, subseq_idx = find_flow_file(info['video_id'], flow_path)
        if flow_file is None:
            print(f"  [WARN] Flow not found for concept {cid}: {info['video_id']}")
            continue
        flow = load_and_subsample_flow(flow_file, num_display_frames=num_frames,
                                       pipeline_frames=pipeline_frames,
                                       pipeline_sampling=pipeline_sampling,
                                       target_size=target_size, subseq_idx=subseq_idx)
        flow_cache[cid] = flow
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        global_max_mag = max(global_max_mag, np.percentile(mag, 98))

    if global_max_mag == 0:
        global_max_mag = 1.0

    print(f"Global max magnitude (98th pct): {global_max_mag:.2f}")
    print(f"Saving individual frames for {len(flow_cache)} concepts → {output_dir}/")

    saved_count = 0
    for cid in selected_concepts:
        if cid not in flow_cache:
            continue
        flow = flow_cache[cid]
        info = medoids[cid]
        cls = get_class_from_video_id(info['video_id'], dataset)

        # Folder name: C{id}_{classname}
        folder_name = f"C{cid:03d}_{cls}"
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for t in range(min(num_frames, flow.shape[0])):
            rgb, _ = flow_to_rgb(flow[t], max_mag=global_max_mag,
                                 gamma=gamma, noise_thresh=noise_thresh)

            fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=150)
            fig.patch.set_facecolor('black')
            ax.imshow(rgb)
            ax.set_title(f"C{cid} ({cls})  t={t}", fontsize=9, fontweight='bold', color='white')
            ax.axis('off')

            frame_path = os.path.join(folder_path, f"t{t:02d}.png")
            fig.savefig(frame_path, bbox_inches='tight', facecolor='black', pad_inches=0.05)
            plt.close(fig)
            saved_count += 1

        print(f"  C{cid:03d}_{cls}: {min(num_frames, flow.shape[0])} frames saved")

    print(f"\n✅ Total {saved_count} frames saved → {output_dir}/")


def save_concept_gifs(medoids, selected_concepts, flow_path, dataset,
                      output_dir, num_frames=8, target_size=224,
                      pipeline_frames=30, pipeline_sampling='center',
                      gamma=0.4, noise_thresh=0.05,
                      gif_fps=4, gif_size=256, penn_anno_dir=None):
    """Save each concept's optical flow sequence as an animated GIF

    Args:
        gif_fps: GIF frame rate (default 4fps → 250ms per frame)
        gif_size: GIF output resolution (default 256x256)
    """
    from PIL import Image, ImageDraw, ImageFont

    # Compute the global max magnitude first
    global_max_mag = 0
    flow_cache = {}
    for cid in selected_concepts:
        info = medoids[cid]
        flow_file, subseq_idx = find_flow_file(info['video_id'], flow_path)
        if flow_file is None:
            print(f"  [WARN] Flow not found for concept {cid}: {info['video_id']}")
            continue
        flow = load_and_subsample_flow(flow_file, num_display_frames=num_frames,
                                       pipeline_frames=pipeline_frames,
                                       pipeline_sampling=pipeline_sampling,
                                       target_size=target_size, subseq_idx=subseq_idx)
        flow_cache[cid] = flow
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        global_max_mag = max(global_max_mag, np.percentile(mag, 98))

    if global_max_mag == 0:
        global_max_mag = 1.0

    os.makedirs(output_dir, exist_ok=True)
    print(f"Global max magnitude (98th pct): {global_max_mag:.2f}")
    print(f"Saving GIFs for {len(flow_cache)} concepts → {output_dir}/")

    duration_ms = int(1000 / gif_fps)
    saved_count = 0

    for cid in selected_concepts:
        if cid not in flow_cache:
            continue
        flow = flow_cache[cid]
        info = medoids[cid]
        cls = get_class_from_video_id(info['video_id'], dataset,
                                      penn_anno_dir=penn_anno_dir)

        frames_pil = []
        for t in range(min(num_frames, flow.shape[0])):
            rgb, _ = flow_to_rgb(flow[t], max_mag=global_max_mag,
                                 gamma=gamma, noise_thresh=noise_thresh)
            img = Image.fromarray(rgb).resize((gif_size, gif_size), Image.LANCZOS)

            # Text overlay: concept ID + class + timestep
            draw = ImageDraw.Draw(img)
            label = f"C{cid} ({cls})  t={t}"
            draw.text((4, 4), label, fill=(255, 255, 255))

            frames_pil.append(img)

        if frames_pil:
            gif_name = f"C{cid:03d}_{cls}.gif"
            gif_path = os.path.join(output_dir, gif_name)
            # Pause briefly on the last frame, then loop
            durations = [duration_ms] * len(frames_pil)
            durations[-1] = duration_ms * 3  # Hold the last frame 3x longer
            frames_pil[0].save(
                gif_path,
                save_all=True,
                append_images=frames_pil[1:],
                duration=durations,
                loop=0  # Loop forever
            )
            saved_count += 1
            print(f"  C{cid:03d}_{cls}: {len(frames_pil)} frames → {gif_name}")

    print(f"\n✅ Total {saved_count} GIFs saved → {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Visualize OF-CBM concept medoid flows')
    parser.add_argument('--dataset', type=str, required=True, choices=['kth', 'penn_action'])
    parser.add_argument('--concept_dir', type=str, required=True,
                        help='Path to concept discovery output (e.g., .../*_trainonly)')
    parser.add_argument('--flow_path', type=str, required=True,
                        help='Path to flow .npy directory')
    parser.add_argument('--output_path', type=str, default='figures/concept_flows.png')
    parser.add_argument('--num_concepts', type=int, default=12,
                        help='Number of concepts to visualize')
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Number of temporal frames to show per concept')
    parser.add_argument('--target_size', type=int, default=224,
                        help='Spatial size for flow visualization')
    parser.add_argument('--pipeline_frames', type=int, default=30,
                        help='Pipeline target_frames (must match concept discovery)')
    parser.add_argument('--pipeline_sampling', type=str, default='center',
                        choices=['center', 'uniform'],
                        help='Pipeline temporal_sampling (must match concept discovery)')
    parser.add_argument('--concept_ids', type=str, default=None,
                        help='Comma-separated concept IDs to visualize (overrides num_concepts)')
    parser.add_argument('--gamma', type=float, default=0.4,
                        help='Gamma correction (< 1.0: amplifies small motions, default 0.4)')
    parser.add_argument('--noise_thresh', type=float, default=0.05,
                        help='Noise threshold (below this: black background, default 0.05)')
    parser.add_argument('--video_anno_path', type=str, default=None,
                        help='Path to annotation dir (for Penn Action class name lookup)')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save individual frames per concept in separate folders')
    parser.add_argument('--frames_dir', type=str, default=None,
                        help='Output directory for individual frames (default: figures/<dataset>_concept_frames)')
    parser.add_argument('--save_gifs', action='store_true',
                        help='Save animated GIFs per concept')
    parser.add_argument('--gifs_dir', type=str, default=None,
                        help='Output directory for GIFs (default: figures/<dataset>_concept_gifs)')
    parser.add_argument('--gif_fps', type=int, default=4,
                        help='GIF frame rate (default 4fps)')
    parser.add_argument('--gif_size', type=int, default=256,
                        help='GIF output resolution (default 256x256)')
    args = parser.parse_args()

    # Auto-detect concept_dir if wildcard
    if '*' in args.concept_dir:
        import glob
        dirs = sorted(glob.glob(args.concept_dir))
        if dirs:
            args.concept_dir = dirs[-1]
            print(f"Auto-detected concept_dir: {args.concept_dir}")

    # Penn Action: preload the class name mapping
    if args.dataset == 'penn_action' and args.video_anno_path:
        global _penn_label_cache
        _penn_label_cache = load_penn_action_label_map(args.video_anno_path)
        print(f"Loaded Penn Action label map: {len(_penn_label_cache)} videos")

    # Load medoid info
    medoids = load_medoid_info(args.concept_dir)
    print(f"Loaded {len(medoids)} concepts")

    # Select concepts
    if args.concept_ids:
        selected = [int(x) for x in args.concept_ids.split(',')]
    else:
        selected = select_diverse_concepts(medoids, args.dataset, args.num_concepts)

    print(f"\nSelected {len(selected)} concepts:")
    print(f"{'ID':>6}  {'Class':<20}  {'Video ID'}")
    print(f"{'-'*6}  {'-'*20}  {'-'*30}")
    for cid in selected:
        if cid in medoids:
            info = medoids[cid]
            cls = get_class_from_video_id(info['video_id'], args.dataset)
            print(f"  C{cid:<4}  {cls:<20}  {info['video_id']}")

    print(f"\nVisualization settings: gamma={args.gamma}, noise_thresh={args.noise_thresh}")

    # 1) Single combined visualization figure
    visualize_concepts(medoids, selected, args.flow_path, args.dataset,
                       args.output_path, args.num_frames, args.target_size,
                       args.pipeline_frames, args.pipeline_sampling,
                       args.gamma, args.noise_thresh)

    # 2) Save individual frames
    if args.save_frames:
        frames_dir = args.frames_dir or f"figures/{args.dataset}_concept_frames"
        save_individual_frames(medoids, selected, args.flow_path, args.dataset,
                               frames_dir, args.num_frames, args.target_size,
                               args.pipeline_frames, args.pipeline_sampling,
                               args.gamma, args.noise_thresh)

    # 3) Save animated GIFs
    if args.save_gifs:
        gifs_dir = args.gifs_dir or f"figures/{args.dataset}_concept_gifs"
        save_concept_gifs(medoids, selected, args.flow_path, args.dataset,
                          gifs_dir, args.num_frames, args.target_size,
                          args.pipeline_frames, args.pipeline_sampling,
                          args.gamma, args.noise_thresh,
                          gif_fps=args.gif_fps, gif_size=args.gif_size,
                          penn_anno_dir=args.video_anno_path)


if __name__ == '__main__':
    main()
