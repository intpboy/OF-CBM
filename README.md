# OF-CBM: Optical Flow-Based Concept Bottleneck Models for Explainable Video Action Recognition

Reference implementation for the KCC 2026 paper
**"OF-CBM: Optical Flow-Based Concept Bottleneck Models for Explainable Video Action Recognition"**
by Junpyo Hong, Wooil Lee, Jongseo Lee, Taesoo Park, and Jinwoo Choi (Kyung Hee University).

> 🏆 **Encouragement Award (Undergraduate Division)** — 2026 Korea Computer Congress (KCC)
> Undergraduate & Junior Paper Competition, Korean Institute of Information Scientists and
> Engineers (KIISE), July 2026.
>
> 📄 **Paper:** [OF-CBM — KCC 2026 (camera-ready PDF)](docs/OF-CBM_KCC2026.pdf) · DBpia link coming soon.

OF-CBM extends the Concept Bottleneck Model (CBM) framework to video action
recognition by using **dense optical flow** as the basis for motion concept
discovery. Instead of relying on predefined concept banks or indirect motion
representations, concepts emerge directly from clustering normalized flow
fields, producing visually intuitive motion exemplars that can be inspected as
short flow clips.

The pipeline is a clean four-step process (plus an optional backbone-finetuning
step and a non-interpretable baseline):

```
  videos ──► (1) dense optical flow extraction ──► per-video .npy flow arrays
                                                            │
                                                            ▼
                               (2) flow-based motion concept discovery
                                   (FINCH clustering + medoid selection)
                                                            │
                                                            ▼
                               (3) frozen VideoMAE backbone feature extraction
                                                            │
                                                            ▼
                               (4) concept projection + sparse classifier
                                   (GLM-SAGA elastic-net)
```

## Results

Top-1 accuracy (%) reported in the paper. OF-CBM matches the non-interpretable
backbone while adding concept-based explanations.

| Method                   | KTH   | Penn Action |
|--------------------------|:-----:|:-----------:|
| VideoMAE (backbone)      | 91.43 | 97.38       |
| OF-CBM (Raw flow)        | 91.77 | 97.00       |
| OF-CBM (Patched flow)    | 91.77 | 97.19       |

## Repository layout

```
OF-CBM/
├── README.md
├── LICENSE                            # MIT
├── requirements.txt
├── flow_extraction/                   # step 1: dense optical flow
│   └── extract_raft_flow.py
├── concept_discovery/                 # step 2: motion concepts from flow
│   ├── motion_discovery/
│   │   ├── discover_flow_concepts.py      # main entry point
│   │   ├── flow_subsampling.py            # Raw / Patched flow representations
│   │   ├── clustering.py                  # FINCH / k-means / SOM
│   │   └── labeling.py                    # medoid + hard-label generation
│   └── utils/
│       ├── finch.py                       # FINCH algorithm
│       └── cluster_utils.py               # medoid helpers, IO, class mapping
├── cbm_training/                      # steps 3-4: CBM on VideoMAE features
│   ├── train_of_cbm.py                    # main training script
│   ├── feature_extraction/
│   │   └── save_features.py               # precompute VideoMAE features
│   ├── models/
│   │   ├── cbm.py
│   │   ├── concept_layer.py               # concept projection + classifier
│   │   ├── modeling_videomae.py           # VideoMAE ViT-B/16
│   │   └── glm_saga/                      # sparse linear solver
│   ├── video_dataloader/                  # video I/O and transforms
│   └── utils/                             # cbm / data / similarity utilities
├── datasets/                          # annotation files only (no videos)
│   ├── KTH/           { class_list.txt, train.csv, val.csv }
│   └── Penn_action/   { class_list.txt, train.csv, val.csv }
├── scripts/                           # one subdir per benchmark
│   ├── kth/
│   │   ├── 01_extract_flow.sh
│   │   ├── 02_discover_concepts.sh         # Raw flow
│   │   ├── 02_discover_concepts_patch.sh   # Patched flow
│   │   ├── 03_extract_backbone_features.sh
│   │   ├── 04_train_cbm.sh                 # Raw flow
│   │   ├── 04_train_cbm_patch.sh           # Patched flow
│   │   └── 05_baseline.sh                  # non-interpretable backbone
│   └── penn_action/                        # same seven scripts
└── visualize_concept_flows.py         # render discovered concepts as flow images/GIFs
```

## Environment

Tested with Python 3.10 + CUDA 11.8.

```bash
conda create -n ofcbm python=3.10 -y
conda activate ofcbm
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision` (provides RAFT for flow extraction),
`timm`, `decord`, `opencv-python`, `scikit-learn`, `scikit-image`, `pandas`,
`matplotlib`, `tqdm` — see `requirements.txt` for the full list. The FINCH
implementation is bundled in `concept_discovery/utils/finch.py`.

## Data

This repository ships **annotation files only** — download the raw videos from
the official sources and point the scripts at them:

- **KTH** — https://www.csc.kth.se/cvap/actions/ (6 action classes, 599 videos)
- **Penn Action** — http://dreamdragon.github.io/PennAction/ (15 classes, 2,326 videos)

Each `datasets/<name>/{train,val}.csv` has rows of `video_filename,label_index`,
and `class_list.txt` maps each label index (by line order) to a class name.

### Backbone checkpoint

Steps 3–5 require a **VideoMAE ViT-B/16** checkpoint. We use the Kinetics-400
pretrained model (optionally fine-tuned on the target dataset). Produce or
download a checkpoint with the official VideoMAE repository
(https://github.com/MCG-NJU/VideoMAE) and pass its path via `FINETUNE_PATH`.

## Running a full experiment

Every script is plain `bash` and reads its paths from environment variables, so
nothing needs to be edited for a new machine. KTH example:

```bash
# 1. dense optical flow for every video  → flows/kth/*.npy
VIDEO_ROOT=/path/to/KTH/videos bash scripts/kth/01_extract_flow.sh

# 2. flow-based motion concept discovery (FINCH, train-only)
bash scripts/kth/02_discover_concepts.sh         # Raw flow
bash scripts/kth/02_discover_concepts_patch.sh   # Patched flow

# 3. precompute frozen VideoMAE backbone features
VIDEO_ROOT=/path/to/KTH/videos \
FINETUNE_PATH=/path/to/videomae_vit_b_16.pth \
bash scripts/kth/03_extract_backbone_features.sh

# 4. train the concept projection + sparse classifier
bash scripts/kth/04_train_cbm.sh                 # Raw flow
bash scripts/kth/04_train_cbm_patch.sh           # Patched flow

# 5. (optional) non-interpretable backbone baseline
bash scripts/kth/05_baseline.sh
```

The same seven scripts exist under `scripts/penn_action/`. Common overrides:
`FLOW_DIR`, `FEAT_DIR`, `CONCEPT_DIR`, `BACKBONE_FEATS`, `SAVE_DIR` (see the
header of each script for its defaults).

## Flow representations

Two flow representations are implemented in
`concept_discovery/motion_discovery/flow_subsampling.py`:

- **Raw** — the full normalized flow tensor is flattened into a single
  high-dimensional vector per video. Preserves fine spatial detail; well suited
  to datasets with well-localized motion (e.g. KTH).
- **Patched** — the flow is divided into a `P×P` patch grid (default `P=16`,
  i.e. 14×14 patches at 224×224) and each patch is summarized by its mean
  `(dx, dy)`. This reduces the feature dimensionality by roughly `256×`,
  trading fine detail for robustness on larger / higher-resolution datasets.

Both strategies normalize each sample by its peak magnitude so that flow vectors
lie in `[-1, 1]`, making clustering invariant to absolute motion speed.

### Optical flow is pluggable

The paper extracts flow with **WAFT**. This repository ships a self-contained
**RAFT** extractor (`flow_extraction/extract_raft_flow.py`, via torchvision) so
the pipeline runs out of the box. Any dense-flow method can be substituted — the
only contract is that each video becomes one `.npy` array of shape
`(L-1, H, W, 2)` with the last dimension holding `(dx, dy)` in pixels.

## Concept discovery

`discover_flow_concepts.py` runs the discovery stage:

1. Load and subsample flow (`FlowDataset`).
2. Cluster train-set features with **FINCH**; with `--use_partition_num -1`
   (default) the partition with the highest NMI against action labels is picked
   automatically.
3. For each cluster, pick the **medoid** (the sample minimizing the sum of
   intra-cluster distances) as its visual representative.
4. Assign validation samples by nearest medoid (`--train_only`) to avoid data
   leakage.
5. Emit the per-video concept labels (`hard_label_train.pkl` /
   `hard_label_val.pkl`) along with `sequence_cluster_map.json`,
   `concept_index.txt`, and `video_attributes.json` under a
   `<N>concepts_<timestamp>_trainonly/` directory.

## CBM training

`train_of_cbm.py` learns a two-stage CBM on top of frozen VideoMAE features:

1. **Concept projection layer** — trained with a cosine-similarity loss that
   aligns backbone features with the discovered concept vectors.
2. **Sparse linear classifier** — trained with cross-entropy plus elastic-net
   regularization via `glm_saga`, producing a small, interpretable set of
   concept → class weights per action.

`--backbone_features` and `--pose_label` point to the artifacts produced by
steps 3 and 2 respectively. The OF-CBM configuration uses `--train_mode pose`.

## Visualizing concepts

`visualize_concept_flows.py` renders each discovered concept's medoid as a flow
image (and optionally a GIF), tracing a prediction back to the motion pattern it
relied on:

```bash
python visualize_concept_flows.py \
    --dataset kth \
    --concept_dir concept_discovery/output/kth_raw/raw/raw/<...>_trainonly \
    --flow_path flows/kth \
    --output_path figures/kth_concepts.png \
    --save_frames --save_gifs
```

## Relation to the DANCE codebase

OF-CBM builds on the CBM training framework from the **DANCE** codebase (Lee et
al., NeurIPS 2025). This repository has been pared down to exactly what OF-CBM
needs: a single frozen VideoMAE backbone, optical-flow motion concepts, and the
KTH / Penn Action benchmarks. The following DANCE components have been **removed**
(not just disabled), so the code reads as an OF-CBM-only project:

- CLIP / LaViLa / InternVid dual-encoder paths (text/vision concept branches)
- Skeleton-keypoint based motion discovery
- Data loaders and training branches for datasets other than KTH / Penn Action
  (Kinetics, SSV2, UCF, HMDB51, etc.)
- Error-analysis, t-SNE, and per-object analysis scripts

If you need any of these, start from the original DANCE repository. The data
loader (`cbm_training/video_dataloader/datasets.py`) now routes only `kth` and
`penn-action`; add a new `elif` branch there to support another dataset.

## Citation

```bibtex
@inproceedings{hong2026ofcbm,
  title     = {{OF-CBM}: Optical Flow-Based Concept Bottleneck Models for Explainable Video Action Recognition},
  author    = {Hong, Junpyo and Lee, Wooil and Lee, Jongseo and Park, Taesoo and Choi, Jinwoo},
  booktitle = {Proceedings of the Korea Computer Congress (KCC)},
  note      = {Encouragement Award, Undergraduate \& Junior Paper Competition (KIISE)},
  year      = {2026}
}
```

## License

Released under the MIT License — see [LICENSE](LICENSE).
