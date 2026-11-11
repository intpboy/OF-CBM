import numpy as np
import os
from sklearn.metrics import normalized_mutual_info_score as nmi
import re
from concept_discovery.utils.finch import FINCH
import concept_discovery.utils.cluster_utils as util
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


def extract_label(args,json_data, class_map):
    video_ids = [item.split('[')[0] for item in json_data]
    if args.dataset == "Penn_action":
        label_dict = util.video_class_mapping(args)
        def get_label(video_id):
            class_name = label_dict.get(video_id, "unknown")  
            return class_map.get(class_name, -1)

        labels = np.array([get_label(video_id) for video_id in video_ids])
    
    elif args.dataset == "HAA100" :
        def extract_class_name(label):
            return re.sub(r"_\d+$", "", label)
        labels = np.array([class_map.get(extract_class_name(item), -1) for item in video_ids])
    
    elif args.dataset == "UCF101":
        def extract_class_name(label):
            match = re.match(r"v_(.+?)_g\d+_c\d+", label)
            if match:
                return match.group(1)
            else:
                return "unknown"
        labels = np.array([class_map.get(extract_class_name(item), -1) for item in video_ids])

    elif args.dataset == "KTH":
        def extract_class_name(item):
            parts = item.split('_')
            for part in parts:
                if part in class_map:
                    return class_map[part]
            return -1
        labels = np.array([extract_class_name(item) for item in video_ids])

    elif args.dataset == "single_object_direction":
        # Filename: {Object}_{direction}_{speed}_{position}
        # direction (up/down/left/right) → mapped to a label via class_map
        def extract_direction(video_id):
            parts = video_id.split('_')
            # direction is the second element (but the object name may contain '_')
            # Look for a direction keyword that exists in class_map
            for part in parts:
                if part in class_map:
                    return class_map[part]
            return -1
        labels = np.array([extract_direction(vid) for vid in video_ids])

    elif args.dataset == "multi_object":
        # Filename: pair_0000000 — the class cannot be inferred from the filename
        # Map video_name → label using train.csv / val.csv
        import pandas as pd
        label_map = {}
        for split in ['train', 'val']:
            csv_f = os.path.join(args.anno_path, f'{split}.csv')
            if os.path.exists(csv_f):
                df = pd.read_csv(csv_f, header=None, names=['video_name', 'label'],
                                 sep=r'[\s,]+', engine='python')
                for _, row in df.iterrows():
                    vid = os.path.splitext(str(row['video_name']))[0]
                    label_map[vid] = int(row['label'])
        labels = np.array([label_map.get(vid, -1) for vid in video_ids])

    elif args.dataset in ("round_trip", "ssv2_5k", "ssv2_chiral"):
        # CSV-based label mapping
        import pandas as pd
        label_map = {}
        for split in ['train', 'val']:
            csv_f = os.path.join(args.anno_path, f'{split}.csv')
            if os.path.exists(csv_f):
                df = pd.read_csv(csv_f, header=None, names=['video_name', 'label'],
                                 sep=r'[\s,]+', engine='python')
                for _, row in df.iterrows():
                    vid = os.path.splitext(str(row['video_name']))[0]
                    label_map[vid] = int(row['label'])
        labels = np.array([label_map.get(vid, -1) for vid in video_ids])

    elif args.dataset == "UCF_Sports":
        # UCF Sports: 10 classes. Use the (video_name, label) pairs from
        # train.csv / val.csv as-is, but match by video_id basename so that flow
        # filenames may be either "<class>_<clip_id>.npy" or bare "<id>.npy".
        import pandas as pd
        label_map = {}
        for split in ['train', 'val']:
            csv_f = os.path.join(args.anno_path, f'{split}.csv')
            if os.path.exists(csv_f):
                df = pd.read_csv(csv_f, header=None, names=['video_name', 'label'],
                                 sep=r'[\s,]+', engine='python')
                for _, row in df.iterrows():
                    raw = str(row['video_name'])
                    # Register all possible keys: full path, basename, extension-stripped variants
                    base = os.path.basename(raw)
                    label_map[os.path.splitext(raw)[0]] = int(row['label'])
                    label_map[os.path.splitext(base)[0]] = int(row['label'])
                    # UCF Sports etc.: also register the "Class/SeqNum/file.avi" → "Class__SeqNum" form
                    parts = os.path.splitext(raw)[0].split("/")
                    if len(parts) >= 2:
                        short_key = f"{parts[0]}__{parts[1]}"
                        label_map[short_key] = int(row['label'])
        labels = np.array([label_map.get(vid, -1) for vid in video_ids])

    return labels

def run_finch_clustering(data, labels, args, output_path):
    if args.clustering_mode == "partition":
        use_partition_num = args.use_partition_num
        c, num_clust, req_c = FINCH(
            data,
            req_clust=None,
            use_ann_above_samples=(data.shape[0]-1000),
            verbose=True,
            seed=655
        )
         

        nmi_scores = []
        for i in range(c.shape[1]):
            score = nmi(labels, c[:, i])
            nmi_scores.append(score)
            print('NMI Score {}: {:.2f}  (clusters: {})'.format(i, score * 100, num_clust[i]))

        # If use_partition_num == -1, automatically select the partition with the highest NMI
        if use_partition_num == -1:
            use_partition_num = int(np.argmax(nmi_scores))
            print(f'\n[AUTO] Best NMI partition selected: P{use_partition_num} '
                  f'(NMI={nmi_scores[use_partition_num]*100:.2f}, clusters={num_clust[use_partition_num]})')

        result_gt = c[:, use_partition_num]
        num_concept = num_clust[use_partition_num]
        selected_score = nmi(labels, result_gt)
        output_txt_path = os.path.join(output_path, f"L{args.len_subsequence}N{args.num_subsequence}_{num_concept}_output.txt")
        with open(output_txt_path, 'w') as f:
            text = 'Selected Partition: {}, num_concept :{} NMI Score: {:.2f}'.format(use_partition_num, num_concept, selected_score * 100)
            f.write(text + '\n')

        return result_gt, num_concept
    elif args.clustering_mode == "req":
        c, num_clust, req_c = FINCH(data, req_clust=args.req_cluster, use_ann_above_samples=(data.shape[0]-1000), verbose=True, seed=655)
        req_score = nmi(labels, req_c)
        output_txt_path = os.path.join(output_path, f"L{args.len_subsequence}N{args.num_subsequence}_{args.req_cluster}_output.txt")
        with open(output_txt_path, 'w') as f:
            text = 'num_concept :{} NMI Score: {:.2f}'.format(args.req_cluster, req_score * 100)
            f.write(text + '\n')
        
        return req_c, args.req_cluster


def run_kmeans_clustering(data, labels, args, output_path):
    kmeans = KMeans(n_clusters=args.req_cluster, random_state=655, n_init='auto')  
    cluster_labels = kmeans.fit_predict(data)

    score = nmi(labels, cluster_labels)

    output_txt_path = os.path.join(
        output_path, f"L{args.len_subsequence}N{args.num_subsequence}_{args.req_cluster}_output.txt"
    )
    with open(output_txt_path, 'w') as f:
        text = f"num_concept :{args.req_cluster} NMI Score: {score * 100:.2f}"
        f.write(text + '\n')
    
    return cluster_labels, args.req_cluster


def run_som_clustering(data, labels, args, output_path):
    # MiniSom is only needed for the optional SOM clustering mode, so it is
    # imported lazily to keep minisom an optional dependency.
    from minisom import MiniSom

    n_clusters = args.req_cluster

    som_side = int(np.ceil(np.sqrt(n_clusters)))
    som = MiniSom(x=som_side, y=som_side, input_len=data.shape[1], sigma=1.0, learning_rate=0.5, random_seed=655)
    
    som.random_weights_init(data)
    
    som.train_random(data, num_iteration=1000)
    
    bmu_list = np.array([som.winner(x) for x in data])  
    cluster_labels = np.ravel_multi_index((bmu_list[:,0], bmu_list[:,1]), dims=(som_side, som_side))  # 2D → 1D
    cluster_labels = LabelEncoder().fit_transform(cluster_labels)
    

    num_concept = len(np.unique(cluster_labels))

    score = nmi(labels, cluster_labels)

    output_txt_path = os.path.join(output_path, f"L{args.len_subsequence}N{args.num_subsequence}_{num_concept}_output.txt")
    with open(output_txt_path, 'w') as f:
        text = f"num_concept :{num_concept} NMI Score: {score * 100:.2f}"
        f.write(text + '\n')
    
    return cluster_labels, num_concept
    
    
    
def clustering(args,output_path):
    util.set_seed(42)  
    data, json_data = util.load_data(output_path)
    data = data.reshape(data.shape[0], -1)
    print(data.shape)
    print(len(json_data))
    class_map = util.class_mapping(args.anno_path)
    labels = extract_label(args, json_data, class_map)
    if args.clustering_mode == "req" or args.clustering_mode == "partition":
        result_gt, num_concept = run_finch_clustering(data, labels,args,output_path)
    elif args.clustering_mode == "k-means":
        result_gt, num_concept = run_kmeans_clustering(data, labels,args,output_path)
    elif args.clustering_mode == "som":
        result_gt, num_concept = run_som_clustering(data, labels,args,output_path)
    return data,result_gt,num_concept
