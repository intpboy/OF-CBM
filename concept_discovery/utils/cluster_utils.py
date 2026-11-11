import os
import glob
import numpy as np
import json
import random
import pandas as pd
import math

def load_class_list(anno_path):
    file_path = os.path.join(anno_path, "class_list.txt")
    with open(file_path, "r") as file:
        return [line.strip() for line in file]
    
def load_json_files(base_path, class_list, dataset):
    if dataset in ("Penn_action", "KTH", "UCF_Sports", "single_object_direction", "multi_object", "round_trip", "ssv2_5k", "ssv2_chiral"):
        return glob.glob(os.path.join(base_path, "*_result.json"))
    elif dataset in ("HAA100", "UCF101"):
        json_files = []
        for class_name in class_list:
            class_folder = os.path.join(base_path, class_name)
            if os.path.isdir(class_folder):
                json_files.extend(glob.glob(os.path.join(class_folder, "*_result.json")))
        return json_files

def load_data(base_path):
    data = np.load(os.path.join(base_path, 'processed_keypoints.npy'))
    with open(os.path.join(base_path, "sample_metadata.json"), "r") as f:
        json_data = json.load(f)
    return data, json_data

def expand_array(frames_data, F):
    frames_data = np.array(frames_data) 
    if frames_data.shape[0] == 0:
        return None  
    if frames_data.shape[0] == 1:
        return np.tile(frames_data, (F, 1, 1))  

    indices = np.linspace(0, frames_data.shape[0] - 1, F, dtype=int)
    expanded_array = frames_data[indices]
    return expanded_array

def save_data(output_path, class_data, class_metadata):
    os.makedirs(output_path, exist_ok=True)
    processed_keypoints_path = os.path.join(output_path, "processed_keypoints.npy")
    sample_metadata_path = os.path.join(output_path, "sample_metadata.json")
    
    np.save(processed_keypoints_path, np.array(class_data))
    with open(sample_metadata_path, "w") as f:
        json.dump(class_metadata, f, indent=4)
        
def save_cluster_info(data, result_gt, num_concept, output_path, save_path):
    # 1. Save centroids
    centroids = []
    for i in range(num_concept):
        members = data[result_gt == i]
        centroid = members.mean(axis=0)
        centroids.append(centroid)
    centroids = np.stack(centroids)
    np.save(os.path.join(save_path, "cluster_centroids.npy"), centroids)

    # 2. Load sample_metadata.json (list of pose sequence IDs)
    sample_metadata_path = os.path.join(output_path, "sample_metadata.json")
    with open(sample_metadata_path, "r") as f:
        json_data = json.load(f)

    # 3. Save pose_sequence_id → cluster mapping
    sequence_cluster_map = {
        json_data[i]: int(cluster_id)
        for i, cluster_id in enumerate(result_gt)
    }
    with open(os.path.join(save_path, "sequence_cluster_map.json"), "w") as f:
        json.dump(sequence_cluster_map, f, indent=4)

    print(f"[✓] Saved {num_concept} cluster centroids and sequence-cluster map with real IDs.")
    
def repeat_to_min_length(arr, min_len):
    n = len(arr)
    if n == 0:
        return arr  
    k = math.ceil(min_len / n)  
    repeated = np.repeat(arr, k, axis=0)
    return repeated
def find_closest_to_centroid(features, cluster_labels):
    unique_clusters = np.unique(cluster_labels)  
    closest_indices = {}  

    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]  
        cluster_points = features[cluster_indices]  

        centroid = np.mean(cluster_points, axis=0)  
        distances = np.linalg.norm(cluster_points - centroid, axis=1)  
        closest_idx = cluster_indices[np.argmin(distances)] 

        closest_indices[cluster] = closest_idx  
    
    return closest_indices

def find_medoid(features, cluster_labels):
    """
    Find the medoid of each cluster.
    Medoid: the sample whose sum of distances to all other samples in the cluster is minimal.

    Args:
        features: (N, D) numpy array — feature vectors of all samples
        cluster_labels: (N,) numpy array — cluster assignment of each sample
    Returns:
        medoid_indices: dict {cluster_id: sample_index}
    """
    from scipy.spatial.distance import cdist

    unique_clusters = np.unique(cluster_labels)
    medoid_indices = {}

    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_points = features[cluster_indices]

        if len(cluster_points) == 1:
            medoid_indices[cluster] = cluster_indices[0]
            continue

        # pairwise distances → the sample with the minimal total distance = medoid
        pairwise_dist = cdist(cluster_points, cluster_points)
        total_dist = pairwise_dist.sum(axis=1)
        medoid_idx = cluster_indices[np.argmin(total_dist)]
        medoid_indices[cluster] = medoid_idx

    return medoid_indices

def remove_missing_videos(csv_path, missing_videos, output_csv_path):
    df = pd.read_csv(csv_path, header=None, names=["video_name", "class_label"],
                     sep=r'[\s,]+', engine='python')
    df_filtered = df[~df["video_name"].isin(missing_videos)] 
    df_filtered.to_csv(output_csv_path, header=False, index=False, sep=",")
    print("--------Removed---------")

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def class_mapping(anno_path):
    class_list = load_class_list(anno_path)

    return {name : idx for idx,name in enumerate(class_list)}

def video_class_mapping(args):
    class_list = load_class_list(args.anno_path)
    train_csv = os.path.join(args.anno_path,"train.csv")
    val_csv = os.path.join(args.anno_path,"val.csv")
    train_df = pd.read_csv(train_csv, header=None, names=["video_name", "class_id"],
                           sep=r'[\s,]+', engine='python')
    val_df = pd.read_csv(val_csv, header=None, names=["video_name", "class_id"],
                         sep=r'[\s,]+', engine='python')
    df = pd.concat([train_df, val_df], ignore_index=True)
    if args.dataset == "UCF101":
        df["video_id"] = df["video_name"].str.replace(".avi", "", regex=False)
    else :
        df["video_id"] = df["video_name"].str.replace(".mp4", "", regex=False)

    df["class_name"] = df["class_id"].apply(lambda x: class_list[int(x)])
    return dict(zip(df["video_id"], df["class_name"]))


def compute_pose_cosine_similarity(xy_sequence: np.ndarray) -> np.ndarray:
    """
    Args:
    xy_sequence (np.ndarray): Keypoint sequence of shape (T, K, 2)

    Returns:
        similarities (np.ndarray): Cosine similarity array of shape (T-1,)
    """
    flat_poses = xy_sequence.reshape((xy_sequence.shape[0], -1))  # shape: (T, K*2)
    norms = np.linalg.norm(flat_poses, axis=1, keepdims=True) + 1e-8
    normalized = flat_poses / norms
    similarities = np.sum(normalized[1:] * normalized[:-1], axis=1)
    return similarities