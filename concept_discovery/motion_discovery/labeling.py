import numpy as np
import os
import json
from collections import defaultdict
import pandas as pd
import pickle
import concept_discovery.utils.cluster_utils as util


def make_attribute(args, result_gt,save_path,output_path,num_concept):
    with open(os.path.join(output_path, "sample_metadata.json"), "r") as f :
        json_data = json.load(f)
    labels = result_gt
    video_frames = defaultdict(list)
    for idx, item in enumerate(json_data):
        video_id = item.split('[')[0]
        video_frames[video_id].append(labels[idx])
    video_attributes = []
    for video_id, labels_ in video_frames.items():
        one_hot_vector = np.zeros(num_concept, dtype=int)
        for label in labels_:
            one_hot_vector[label] = 1 
        if args.dataset == "Penn_action" or args.dataset == "KTH":
            video_name = f"{video_id}.mp4"
        elif args.dataset in ("single_object_direction", "multi_object", "round_trip", "ssv2_5k", "ssv2_chiral"):
            video_name = f"{video_id}.mp4"
        elif args.dataset == "UCF_Sports":
            # The original UCF Sports extension is .avi. Keep it consistent with the video_name in the CSV.
            video_name = f"{video_id}.avi"
        elif args.dataset == "HAA100" :
            video_name = f"{video_id.rsplit('_', 1)[0]}/{video_id}.mp4"
        elif args.dataset == "UCF101":
            if video_id.startswith("v_"):
                no_prefix = video_id[2:]
                class_name = no_prefix.split('_')[0]
                if class_name == "HandStandPushups":
                    class_name = "HandstandPushups"
                video_name = f"{class_name}/{video_id}.avi"
            else:
                video_name = f"{video_id}.avi"
        video_entry = {
            "video_name": video_name,
            "attribute_label": one_hot_vector.tolist()
        }
        
        video_attributes.append(video_entry)

    output_annotation_path = os.path.join(save_path,"video_attributes.json")
    with open(output_annotation_path, "w") as f:
        json.dump(video_attributes, f, indent=4)
    return video_attributes


def hard_label(video_attributes, args, save_path, num_concept, mode):
    output_json_path = os.path.join(save_path, f"hard_label_{mode}.json")
    output_pkl_path = os.path.join(save_path, f"hard_label_{mode}.pkl")
    csv_path = os.path.join(args.anno_path, f"{mode}.csv")
    new_csv_path = os.path.join(save_path, f"{mode}.csv") 
    cnt = 0

    df = pd.read_csv(csv_path, header=None, names=["video_name", "class_label"],
                     sep=r'[\s,]+', engine='python')
    video_list = df["video_name"].tolist()

    annotation_dict = {
        item["video_name"]: item["attribute_label"]
        for item in video_attributes
    }
    # UCF Sports etc.: CSV video_name "Class/SeqNum/file.avi" → short key "Class__SeqNum.avi"
    # Reverse mapping for when the flow-based video_name has the "Class__SeqNum.avi" form
    csv_to_short = {}
    for vn in video_list:
        parts = os.path.splitext(vn)[0].split("/")
        if len(parts) >= 2:
            short_key = f"{parts[0]}__{parts[1]}.avi"
            csv_to_short[vn] = short_key

    sorted_annotations = []
    #missing_videos = []

    for video_name in video_list:
        # Try direct matching first; if that fails, match via the short key
        matched_label = annotation_dict.get(video_name)
        if matched_label is None:
            short_key = csv_to_short.get(video_name)
            if short_key:
                matched_label = annotation_dict.get(short_key)
        if matched_label is not None:
            sorted_annotations.append({
                "video_name": video_name,
                "attribute_label" : matched_label
            })
        else :
            # missing_videos.append(video_name)
            print(f"Missing video: {video_name}")
            cnt+=1
            sorted_annotations.append({
            "video_name": video_name,
            "attribute_label": [-1] * num_concept
        })
    print(f"Number of missing video : {cnt}")
    with open(output_json_path, "w") as f:
        json.dump(sorted_annotations, f, indent=4)

    with open(output_pkl_path, "wb") as f:
        pickle.dump(sorted_annotations, f)
    # util.remove_missing_videos(csv_path, missing_videos, new_csv_path)

    return sorted_annotations

def labeling(args, data, result_gt,save_path,output_path,num_concept, use_medoid=False):
    # data, result_gt = clustering.clustering(args)
    if use_medoid:
        closest_sample_indices = util.find_medoid(data, result_gt)
    else:
        closest_sample_indices = util.find_closest_to_centroid(data, result_gt)
    with open(os.path.join(save_path,'concept_index.txt'), "w", encoding="utf-8") as f:
        for key, value in closest_sample_indices.items():
            f.write(f"{key}: {value}\n")

    video_attributes = make_attribute(args, result_gt,save_path,output_path,num_concept)
    train_anno = hard_label(video_attributes, args, save_path, num_concept,mode = "train")
    if args.dataset == "UCF101":
        val_anno = hard_label(video_attributes, args, save_path,num_concept, mode = "test")
    else :
        val_anno = hard_label(video_attributes, args, save_path,num_concept, mode = "val")
    return closest_sample_indices

