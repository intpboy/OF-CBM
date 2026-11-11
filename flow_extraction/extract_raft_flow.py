"""
RAFT-based dense optical flow extraction.

For each video in the given annotation CSV, extract dense optical flow between
consecutive frames using a pretrained RAFT model (torchvision.models.optical_flow).
The flow for a video with L frames is stored as a single float16/float32 .npy file
of shape (L-1, H, W, 2), where the last dimension is (dx, dy) in pixels.

Usage
-----
    python flow_extraction/extract_raft_flow.py \
        --video_root /path/to/videos \
        --anno_path datasets/KTH \
        --output_dir /path/to/output_flows \
        --resize 224 224 \
        --dtype float16

Output layout
-------------
    {output_dir}/
        {video_id_without_extension}.npy   # shape (L-1, H, W, 2)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import decord
    from decord import VideoReader, cpu
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False

import cv2
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


def read_video_frames(video_path, resize=None):
    """
    Read all frames of a video.

    Returns:
        frames: uint8 numpy array of shape (L, H, W, 3) in RGB order
    """
    if _HAS_DECORD:
        vr = VideoReader(video_path, ctx=cpu(0))
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # (L, H, W, 3)
    else:
        cap = cv2.VideoCapture(video_path)
        frames_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames_list:
            return None
        frames = np.stack(frames_list, axis=0)

    if resize is not None:
        tH, tW = resize
        resized = np.empty((frames.shape[0], tH, tW, 3), dtype=frames.dtype)
        for t in range(frames.shape[0]):
            resized[t] = cv2.resize(frames[t], (tW, tH), interpolation=cv2.INTER_LINEAR)
        frames = resized

    return frames


def frames_to_tensor(frames, device):
    """
    Convert (L, H, W, 3) uint8 → (L, 3, H, W) float on `device`,
    normalized to the [-1, 1] range expected by RAFT.
    """
    x = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float()
    x = (x / 255.0) * 2.0 - 1.0
    return x


@torch.no_grad()
def compute_flow_for_video(model, frames, device, batch_size=4):
    """
    Compute dense optical flow for every consecutive frame pair.

    Args:
        model: RAFT model (eval mode)
        frames: (L, H, W, 3) uint8 RGB numpy array
        device: torch device
        batch_size: number of pairs processed simultaneously
    Returns:
        flow: (L-1, H, W, 2) numpy array
    """
    L = frames.shape[0]
    if L < 2:
        return None

    x = frames_to_tensor(frames, device)  # (L, 3, H, W)
    img1 = x[:-1]  # (L-1, 3, H, W)
    img2 = x[1:]

    # RAFT requires H and W to be divisible by 8
    H, W = img1.shape[-2], img1.shape[-1]
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h or pad_w:
        img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode='replicate')
        img2 = F.pad(img2, (0, pad_w, 0, pad_h), mode='replicate')

    flows = []
    for i in range(0, img1.shape[0], batch_size):
        p1 = img1[i : i + batch_size]
        p2 = img2[i : i + batch_size]
        # RAFT returns a list of refined flow predictions; take the last one
        flow = model(p1, p2)[-1]  # (B, 2, H_pad, W_pad)
        if pad_h or pad_w:
            flow = flow[:, :, :H, :W]
        flows.append(flow.cpu())

    flow = torch.cat(flows, dim=0)         # (L-1, 2, H, W)
    flow = flow.permute(0, 2, 3, 1).numpy()  # (L-1, H, W, 2)
    return flow


def load_video_list(anno_path):
    """
    Parse train.csv and val.csv under `anno_path` and return a unique list of
    video file names (without extension stripped).
    """
    names = []
    for split in ("train.csv", "val.csv"):
        csv_path = os.path.join(anno_path, split)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, header=None, names=["video_name", "label"],
                         sep=r'[\s,]+', engine='python')
        names.extend(df["video_name"].astype(str).tolist())
    # deduplicate while preserving order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def main():
    parser = argparse.ArgumentParser(description="RAFT optical flow extraction")
    parser.add_argument("--video_root", required=True,
                        help="root directory containing video files")
    parser.add_argument("--anno_path", required=True,
                        help="directory with train.csv / val.csv")
    parser.add_argument("--output_dir", required=True,
                        help="output directory for .npy flow files")
    parser.add_argument("--resize", type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="resize frames to HxW before flow extraction (default: keep original)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="number of frame pairs per forward call (default: 4)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"],
                        help="storage dtype for flow arrays (float16 halves disk usage)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_existing", action="store_true",
                        help="skip videos whose .npy output already exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[INFO] loading RAFT (large)...")
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()

    video_names = load_video_list(args.anno_path)
    print(f"[INFO] found {len(video_names)} videos in {args.anno_path}")

    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    n_ok, n_skip, n_fail = 0, 0, 0
    for name in tqdm(video_names, desc="extracting flow"):
        video_id = os.path.splitext(name)[0]
        out_path = os.path.join(args.output_dir, f"{video_id}.npy")
        if args.skip_existing and os.path.exists(out_path):
            n_skip += 1
            continue

        video_path = os.path.join(args.video_root, name)
        if not os.path.exists(video_path):
            print(f"  [WARN] missing video: {video_path}", file=sys.stderr)
            n_fail += 1
            continue

        try:
            frames = read_video_frames(video_path, resize=tuple(args.resize) if args.resize else None)
            if frames is None or frames.shape[0] < 2:
                print(f"  [WARN] cannot read {video_path}", file=sys.stderr)
                n_fail += 1
                continue

            flow = compute_flow_for_video(model, frames, device, batch_size=args.batch_size)
            if flow is None:
                n_fail += 1
                continue

            np.save(out_path, flow.astype(np_dtype))
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {name}: {e}", file=sys.stderr)
            n_fail += 1

    print(f"[DONE] ok={n_ok}, skipped={n_skip}, failed={n_fail}")


if __name__ == "__main__":
    main()
