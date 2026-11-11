"""
No-CBM baseline: backbone feature → linear classifier (without concept projection).

Linear probe baseline that classifies VideoMAE backbone features directly into
action classes without a concept layer. It can be trained either with the
GLM-SAGA sparse solver (sparse mode) or with plain cross-entropy + Adam.

Call path:
    train_of_cbm.py  --no_cbm  →  train_non_cbm(args, backbone_feats, val_feats, save_name)
"""
import os
import time
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from cbm_training.video_dataloader import datasets
from .glm_saga.elasticnet import IndexedTensorDataset, glm_saga


# ──────────────────────────────────────────────────────────
# Label loading (same logic as concept_layer.py)
# ──────────────────────────────────────────────────────────

_CSV_DATASETS = (
    'single_object', 'single_object_fix_starting_point',
    'multi_object', 'round_trip', 'ssv2_5k', 'ssv2_chiral',
    'ucf_sports',
)


def _load_labels(args):
    """Return (train_y, val_y, classes) where train_y/val_y are LongTensor."""
    cls_file = os.path.join(args.video_anno_path, 'class_list.txt')
    with open(cls_file, "r") as f:
        classes = [c for c in f.read().split("\n") if c]
    assert args.nb_classes == len(classes), (
        f"nb_classes ({args.nb_classes}) != class_list ({len(classes)})"
    )

    if args.data_set in _CSV_DATASETS:
        train_csv = os.path.join(args.video_anno_path, 'train.csv')
        val_csv = os.path.join(args.video_anno_path, 'val.csv')
        train_df = pd.read_csv(train_csv, header=None,
                               names=['video_name', 'label'],
                               sep=r'[\s,]+', engine='python')
        val_df = pd.read_csv(val_csv, header=None,
                             names=['video_name', 'label'],
                             sep=r'[\s,]+', engine='python')
        train_y = torch.LongTensor(train_df['label'].values)
        val_y = torch.LongTensor(val_df['label'].values)
    else:
        train_ds, _ = datasets.build_dataset(True, False, args)
        val_ds, _ = datasets.build_dataset(False, False, args)
        train_y = torch.LongTensor(train_ds.label_array)
        val_y = torch.LongTensor(val_ds.label_array)

    return train_y, val_y, classes


# ──────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(linear, features, labels, device, batch_size=512):
    """Top-1 accuracy."""
    linear.eval()
    correct = 0
    total = 0
    ds = TensorDataset(features, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = linear(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────
# GLM-SAGA sparse linear probe
# ──────────────────────────────────────────────────────────

def _train_sparse(args, train_feats, val_feats, train_y, val_y, classes,
                  save_name, device):
    """Sparse linear probe using GLM-SAGA (same solver as the OF-CBM classification layer)."""
    # Normalize features (zero-mean, unit-std per dim)
    train_mean = train_feats.mean(dim=0, keepdim=True)
    train_std = train_feats.std(dim=0, keepdim=True).clamp(min=1e-8)
    train_normed = (train_feats - train_mean) / train_std
    val_normed = (val_feats - train_mean) / train_std

    indexed_train_ds = IndexedTensorDataset(train_normed, train_y)
    val_ds = TensorDataset(val_normed, val_y)
    train_loader = DataLoader(indexed_train_ds,
                              batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds,
                            batch_size=args.saga_batch_size, shuffle=False)

    linear = nn.Linear(train_normed.shape[1], len(classes)).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    STEP_SIZE = 0.05
    ALPHA = 0.99
    metadata = {'max_reg': {'nongrouped': args.lam}}

    output_proj = glm_saga(
        linear, train_loader, STEP_SIZE, args.n_iters, ALPHA,
        epsilon=1, k=1, val_loader=val_loader, do_zero=False,
        metadata=metadata, n_ex=len(train_feats),
        n_classes=len(classes), verbose=500,
    )

    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']

    # Save
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))

    # Final eval
    linear.load_state_dict({'weight': W_g, 'bias': b_g})
    train_acc = _evaluate(linear, train_normed, train_y, device)
    val_acc = _evaluate(linear, val_normed, val_y, device)

    return train_acc, val_acc


# ──────────────────────────────────────────────────────────
# Cross-entropy + Adam linear probe
# ──────────────────────────────────────────────────────────

def _train_ce(args, train_feats, val_feats, train_y, val_y, classes,
              save_name, device):
    """Linear probe trained with plain cross-entropy (non-sparse)."""
    # Normalize
    train_mean = train_feats.mean(dim=0, keepdim=True)
    train_std = train_feats.std(dim=0, keepdim=True).clamp(min=1e-8)
    train_normed = (train_feats - train_mean) / train_std
    val_normed = (val_feats - train_mean) / train_std

    linear = nn.Linear(train_normed.shape[1], len(classes)).to(device)
    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)

    optimizer = torch.optim.Adam(
        linear.parameters(),
        lr=args.no_cbm_lr,
        weight_decay=args.no_cbm_weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(train_normed, train_y)
    train_loader = DataLoader(train_ds, batch_size=args.no_cbm_batchsize,
                              shuffle=True)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.no_cbm_epochs):
        linear.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = linear(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = _evaluate(linear, val_normed, val_y, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in linear.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [Epoch {epoch+1:3d}/{args.no_cbm_epochs}] "
                  f"loss={total_loss/len(train_loader):.4f}  "
                  f"val_acc={val_acc*100:.2f}%  "
                  f"best={best_val_acc*100:.2f}%")

    # Save best
    if best_state is not None:
        linear.load_state_dict(best_state)
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(linear.weight.data, os.path.join(save_name, "W_g.pt"))
    torch.save(linear.bias.data, os.path.join(save_name, "b_g.pt"))

    train_acc = _evaluate(linear, train_normed, train_y, device)
    val_acc = best_val_acc

    return train_acc, val_acc


# ──────────────────────────────────────────────────────────
# Entry point (called from train_of_cbm.py)
# ──────────────────────────────────────────────────────────

def train_non_cbm(args, backbone_features, val_backbone_features, save_name):
    """
    Classify VideoMAE backbone features directly into action classes without concept projection.

    Depending on args.mode_no_cbm:
        'only_cls'             → CE + Adam linear probe
        'only_sparse_cls'      → GLM-SAGA sparse linear probe
        'dump_linear_cls'      → CE + Adam (dump_concept_num is ignored)
        'dump_linear_sparse_cls' → GLM-SAGA sparse
    """
    device = torch.device(args.device)
    t0 = time.time()

    print("=" * 60)
    print("  No-CBM baseline: backbone → linear classifier")
    print(f"  Mode       : {args.mode_no_cbm}")
    print(f"  data_set   : {args.data_set}")
    print(f"  nb_classes : {args.nb_classes}")
    print(f"  feat dim   : {backbone_features.shape}")
    print(f"  save_name  : {save_name}")
    print("=" * 60)

    train_y, val_y, classes = _load_labels(args)

    # Sanity check: feature count == label count
    assert backbone_features.shape[0] == train_y.shape[0], (
        f"Train feature count ({backbone_features.shape[0]}) != "
        f"train label count ({train_y.shape[0]})"
    )
    assert val_backbone_features.shape[0] == val_y.shape[0], (
        f"Val feature count ({val_backbone_features.shape[0]}) != "
        f"val label count ({val_y.shape[0]})"
    )

    use_sparse = args.mode_no_cbm in ('only_sparse_cls', 'dump_linear_sparse_cls')

    if use_sparse:
        print("\n[GLM-SAGA sparse linear probe]")
        train_acc, val_acc = _train_sparse(
            args, backbone_features, val_backbone_features,
            train_y, val_y, classes, save_name, device,
        )
    else:
        print(f"\n[CE + Adam linear probe]  "
              f"epochs={args.no_cbm_epochs}  lr={args.no_cbm_lr}  "
              f"wd={args.no_cbm_weight_decay}")
        train_acc, val_acc = _train_ce(
            args, backbone_features, val_backbone_features,
            train_y, val_y, classes, save_name, device,
        )

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  No-CBM baseline results")
    print(f"  Train accuracy : {train_acc * 100:.2f}%")
    print(f"  Val accuracy   : {val_acc * 100:.2f}%")
    print(f"  Elapsed        : {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 60)

    # Save summary
    summary = {
        'mode': args.mode_no_cbm,
        'data_set': args.data_set,
        'nb_classes': args.nb_classes,
        'feat_dim': backbone_features.shape[1],
        'train_samples': backbone_features.shape[0],
        'val_samples': val_backbone_features.shape[0],
        'train_acc': round(train_acc * 100, 2),
        'val_acc': round(val_acc * 100, 2),
        'elapsed_sec': round(elapsed, 1),
    }
    with open(os.path.join(save_name, "nocbm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {os.path.join(save_name, 'nocbm_summary.json')}")
