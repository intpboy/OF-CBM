from cbm_training.video_dataloader import datasets
import cbm_training.models.cbm as cbm
import pickle
import torch
from torch import nn
import cbm_training.models.cbm as cbm
import torch
import cbm_training.utils.debugging as debugging
import os
import random
import cbm_training.utils.cbm_utils as cbm_utils
import cbm_training.utils.data_utils as data_utils
import cbm_training.utils.similarity as similarity
import torch.optim
import json
from .glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset



def train_pose_cocept_layer(args,target_features, val_target_features,save_name):
    if args.loss_mode =='concept':
        similarity_fn = similarity.cos_similarity_cubed_single_concept
    elif args.loss_mode =='sample':
        similarity_fn = similarity.cos_similarity_cubed_single_sample
    elif args.loss_mode =='second':
        similarity_fn = similarity.cos_similarity_cubed_single_secondpower
    elif args.loss_mode =='first_concept':
        similarity_fn = similarity.cos_similarity_cubed_single_firstpower_concept
    elif args.loss_mode =='first_sample':
        similarity_fn = similarity.cos_similarity_cubed_single_firstpower_sample
    else:
        exit()
    # similarity_fn = similarity.cos_similarity_cubed_single_sample
    if args.pose_label is not None:
        file_path = args.pose_label
        n_feat = target_features.size(0)
        n_val_feat = val_target_features.size(0)
        with open(os.path.join(file_path,'hard_label_train.pkl'), "rb") as file:
            hard_label_train = pickle.load(file)  
        with open(os.path.join(file_path,'hard_label_val.pkl'), "rb") as file:
            hard_label_val = pickle.load(file) 
        train_attribute = [item['attribute_label'] for item in hard_label_train]
        

        train_result_tensor = torch.tensor(train_attribute,dtype=target_features.dtype)
    
        val_attribute = [item['attribute_label'] for item in hard_label_val]
        
        
        val_result_tensor = torch.tensor(val_attribute,dtype=target_features.dtype)
        print("▶ target_features:", target_features.shape)
        print("▶ val_target_features:", val_target_features.shape)
        print("▶ train_result_tensor:", train_result_tensor.shape)
        print("▶ val_result_tensor:", val_result_tensor.shape)
        print("▶ First 5 train labels:", train_result_tensor[:5])
        print("▶ First 5 val labels:", val_result_tensor[:5])
        
        if not args.use_mlp:
            train_result_tensor[train_result_tensor == 0.] = 0.05
            train_result_tensor[train_result_tensor == 1.] = 0.3
            val_result_tensor[val_result_tensor == 0.] = 0.05
            val_result_tensor[val_result_tensor == 1.] = 0.3
        if args.no_filter_out:
            train_result_tensor[train_result_tensor == -1.] = 1e-8
            val_result_tensor[val_result_tensor == -1.] = 1e-8
            print("----------------Using Noisy sample----------------------")

    
    train_valid_index = torch.where(train_result_tensor.max(dim=1).values != -1)[0]
    val_valid_index = torch.where(val_result_tensor.max(dim=1).values != -1)[0]
    print(f"▶ train_valid_index: {len(train_valid_index)}/{train_result_tensor.shape[0]}")
    print(f"▶ val_valid_index: {len(val_valid_index)}/{val_result_tensor.shape[0]}")
    print(f"▶ val max values (first 5): {val_result_tensor.max(dim=1).values[:5]}")
    
    target_features_indexed = target_features[train_valid_index]
    train_result_tensor_indexed = train_result_tensor[train_valid_index]
    
    val_target_features_indexed = val_target_features[val_valid_index]
    val_result_tensor_indexed = val_result_tensor[val_valid_index]

    if args.use_mlp:
        # proj_layer = cbm.ModelOracleCtoY(n_class_attr=2, input_dim=target_features.shape[1],num_classes=train_result_tensor.shape[1])
        proj_layer =torch.nn.Linear(target_features_indexed.shape[1],train_result_tensor_indexed.shape[1],bias=False).to(args.device)
        torch.nn.init.xavier_uniform_(proj_layer.weight)
        opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)
        criterion =nn.BCEWithLogitsLoss()# torch.nn.CrossEntropyLoss()
    else:
        proj_layer = torch.nn.Linear(in_features=target_features_indexed.shape[1], out_features=train_result_tensor_indexed.shape[1],
                                  bias=False).to(args.device)
        # proj_layer.weight.data.zero_()
        # proj_layer.bias.data.zero_()
        opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)    
    indices = [ind for ind in range(len(target_features_indexed))]
    
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features_indexed))
    # import pickle
    # result_tensor /= torch.norm(result_tensor,dim=1,keepdim=True)

    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features_indexed[batch].to(args.device).detach())
        # if args.pose_label is not None:
        if args.use_mlp:
            loss = criterion(outs, train_result_tensor_indexed[batch].to(args.device).detach())
        else:
            loss = -similarity_fn(train_result_tensor_indexed[batch].to(args.device).detach(), outs)
   
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%500==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features_indexed.to(args.device).detach())
                if args.use_mlp:
                    val_loss = criterion(val_output, val_result_tensor_indexed.to(args.device).detach())
                else:
                    val_loss = -similarity_fn(val_result_tensor_indexed.to(args.device).detach(), val_output)
                

                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                if args.use_mlp:
                    best_weights = proj_layer.weight.clone()
                else:
                    best_weights = proj_layer.weight.clone()

            elif val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_step = i
                if args.use_mlp:
                    best_weights = proj_layer.weight.clone()

                
                else:
                    best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                pass
            if args.use_mlp:
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(i, loss.cpu(),
                                                                                            val_loss.cpu()))
            else:
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(i, -loss.cpu(),
                                                                                            -val_loss.cpu()))
        opt.zero_grad()
    if args.use_mlp:
        proj_layer.load_state_dict({"weight":best_weights})
    else:
        proj_layer.load_state_dict({"weight":best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
    with torch.no_grad():
        proj_layer = proj_layer.cpu()
        train_c = proj_layer(target_features.detach())
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    W_c = proj_layer.weight[:]
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))

    return W_c, best_val_loss

def train_cocept_layer(args,concepts, target_features,val_target_features,clip_feature,val_clip_features,save_name):
    similarity_fn = similarity.cos_similarity_cubed_single_concept
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts),
                                bias=False).to(args.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)

    indices = [ind for ind in range(len(target_features))]

    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        loss = -similarity_fn(clip_feature[batch].to(args.device).detach(), outs)
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%500==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_clip_features.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                            -best_val_loss.cpu()))
                
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                break
            print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(i, -loss.cpu(),
                                                                                            -val_loss.cpu()))
        opt.zero_grad()
    proj_layer.load_state_dict({"weight":best_weights})
    print("**Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))

    #delete concepts that are not interpretable
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
        interpretable = sim > args.interpretability_cutoff
    if args.print:
        for i, concept in enumerate(concepts):
            if sim[i]<=args.interpretability_cutoff:
                print("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))
    original_n_concept = len(concepts)

    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]
    print(f"interpretability_cutoff: concept {original_n_concept}-> {len(concepts)}")
    W_c = proj_layer.weight[interpretable]
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))
    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    proj_layer = torch.nn.Linear(target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})
    with torch.no_grad():
        proj_layer = proj_layer.cpu()
        train_c = proj_layer(target_features.detach())
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    return W_c,concepts,best_val_loss

def train_classification_layer(args=None,W_c=None,pre_concepts=None,concepts=None, target_features=None,val_target_features=None,save_name=None,joint=None,best_val_loss=None):
    # cls_file = data_utils.LABEL_FILES[args.data_set]
    cls_file = os.path.join(args.video_anno_path, 'class_list.txt')
    with open(cls_file, "r") as f:
        classes = [c for c in f.read().split("\n") if c.strip()]
    assert args.nb_classes == len(classes), f"Error: args.nb_classes ({args.nb_classes}) != len(classes) ({len(classes)})"
    
    
    csv_datasets = ('single_object', 'single_object_fix_starting_point', 'multi_object', 'round_trip', 'ssv2_5k', 'ssv2_chiral', 'ucf_sports')
    if args.data_set in csv_datasets:
        # Load labels directly from CSV (no build_dataset needed)
        import pandas as pd
        train_csv = os.path.join(args.video_anno_path, 'train.csv')
        val_csv = os.path.join(args.video_anno_path, 'val.csv')
        train_df = pd.read_csv(train_csv, header=None, names=['video_name', 'label'],
                               sep=r'[\s,]+', engine='python')
        val_df = pd.read_csv(val_csv, header=None, names=['video_name', 'label'],
                             sep=r'[\s,]+', engine='python')
        train_y = torch.LongTensor(train_df['label'].values)
        val_y = torch.LongTensor(val_df['label'].values)
    else:
        train_video_dataset, _ = datasets.build_dataset(True, False, args)
        val_video_dataset,_ =   datasets.build_dataset(False, False, args)
        # d_train = args.data_set + "_train"
        # d_val = args.data_set + "_val"
        train_targets = train_video_dataset.label_array
        val_targets = val_video_dataset.label_array
        train_y = torch.LongTensor(train_targets)
        val_y = torch.LongTensor(val_targets)
        
    proj_layer = torch.nn.Linear(target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std

        
        


    indexed_train_ds = IndexedTensorDataset(train_c, train_y)
    val_ds = TensorDataset(val_c,val_y)
        
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_c.shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    STEP_SIZE = 0.05
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    #concept layer to classification
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
                    val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len(classes),verbose=500)
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    
    # if joint is None:
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))
 

    return train_c, val_c

def get_concept_features(
    args=None,W_c=None,concepts=None, 
    target_features=None,val_target_features=None):
            
    proj_layer = torch.nn.Linear(target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std
    return train_c, val_c

def train_aggregated_classification_layer(
    args=None,
    target_features=None, 
    val_target_features=None,
    W_c = None,
    concepts=None,
    save_name=None
):
    # Create integrated save folder
    save_name = os.path.join(save_name, 'aggregated')
    os.makedirs(save_name, exist_ok=True)
    aggregated_train_c_features = []
    aggregated_val_c_features = []
    for i,W in enumerate(W_c):
        train_c, val_c =  get_concept_features(args,
                                W_c=W,
                                concepts = concepts[i],
                                target_features=target_features,
                                val_target_features=val_target_features
                                )
        aggregated_train_c_features.append(train_c)
        aggregated_val_c_features.append(val_c)

    # Feature concat
    aggregated_train_c = torch.cat(aggregated_train_c_features, dim=1)
    aggregated_val_c = torch.cat(aggregated_val_c_features, dim=1)

    # Load classes
    cls_file = os.path.join(args.video_anno_path, 'class_list.txt')
    with open(cls_file, "r") as f:
        classes = [c for c in f.read().split("\n") if c.strip()]
    assert args.nb_classes == len(classes), f"Error: args.nb_classes ({args.nb_classes}) != len(classes) ({len(classes)})"

    # Load target
    train_video_dataset, _ = datasets.build_dataset(True, False, args)
    val_video_dataset,_ = datasets.build_dataset(False, False, args)
    train_y = torch.LongTensor(train_video_dataset.label_array)
    val_y = torch.LongTensor(val_video_dataset.label_array)

    # Create loader
    indexed_train_ds = IndexedTensorDataset(aggregated_train_c, train_y)
    val_ds = TensorDataset(aggregated_val_c, val_y)
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    # Initialize model
    linear = torch.nn.Linear(aggregated_train_c.shape[1], len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    # Configure GLM
    STEP_SIZE = 0.05
    ALPHA = 0.99
    metadata = {'max_reg': {'nongrouped': args.lam}}

    output_proj = glm_saga(
        linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
        val_loader=val_loader, do_zero=False, metadata=metadata,
        n_ex=len(aggregated_train_c), n_classes=len(classes), verbose=500
    )

    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']

    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))

    # ✅ Save concept list
    concepts = [item for sublist in concepts for item in sublist]
    if concepts:
        with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
            f.write(concepts[0])
            for concept in concepts[1:]:
                f.write('\n' + concept)

    # Save metrics
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {k: float(output_proj['path'][0][k]) for k in ('lam', 'lr', 'alpha', 'time')}
        out_dict['metrics'] = output_proj['path'][0]['metrics']
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {
            "Non-zero weights": nnz,
            "Total weights": total,
            "Percentage non-zero": nnz / total
        }
        json.dump(out_dict, f, indent=2)



def spatio_temporal_joint(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            save_name):
    save_spatial = os.path.join(save_name,'spatial')
    save_temporal = os.path.join(save_name,'temporal')
    save_spatio_temporal = os.path.join(save_name,'spatio_temporal')
    # save_spatio_temporal = os.path.join(save_name,'spatio_temporal')
    os.mkdir(save_spatial)
    os.mkdir(save_temporal)
    os.mkdir(save_spatio_temporal)
    s_W_c,s_concepts = train_cocept_layer(args,
                               s_concepts,
                               target_features,
                               val_target_features,
                               s_clip_features,
                               s_val_clip_features,
                               save_spatial)
    t_W_c,t_concepts = train_cocept_layer(args,
                               t_concepts,
                               target_features,
                               val_target_features,
                               t_clip_features,
                               t_val_clip_features,
                               save_temporal)

    
    del s_clip_features, s_val_clip_features,t_clip_features, t_val_clip_features
    

    st_W_c = torch.cat([s_W_c,t_W_c],dim=0)
    st_concepts = s_concepts+t_concepts
    train_classification_layer(args,
                               W_c=st_W_c,
                               pre_concepts=None,
                               concepts = st_concepts,
                               target_features=target_features,
                               val_target_features=val_target_features,
                                save_name=save_spatio_temporal
                               )

    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # d_val = args.data_set + "_test"
    # val_data_t = data_utils.get_data(d_val,args=args)
    # device = torch.device(args.device)
    
    # s_model,t_model = cbm.load_cbm_two_stream(save_name, device,args)

    # print("!****! Start test Spatio")
    # accuracy = cbm_utils.get_accuracy_cbm(s_model, val_data_t, device,32,10)
    # print("!****! Spatio Accuracy: {:.2f}%".format(accuracy*100))
    
    # print("?***? Start test Temporal")
    # accuracy = cbm_utils.get_accuracy_cbm(t_model, val_data_t, device,32,10)
    # print("?****? Temporal Accuracy: {:.2f}%".format(accuracy*100))
    
    return

def spatio_temporal_three_joint(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            p_concepts,
                            p_clip_features,
                            p_val_clip_features,
                            save_name):
    save_spatial = os.path.join(save_name,'spatial')
    save_temporal = os.path.join(save_name,'temporal')
    save_place = os.path.join(save_name,'place')
    save_spatio_temporal_place = os.path.join(save_name,'spatio_temporal_place')
    # save_spatio_temporal = os.path.join(save_name,'spatio_temporal')
    os.mkdir(save_spatial)
    os.mkdir(save_temporal)
    os.mkdir(save_place)
    os.mkdir(save_spatio_temporal_place)
    train_cs,val_cs = [],[]
    s_W_c,s_concepts = train_cocept_layer(args,
                               s_concepts,
                               target_features,
                               val_target_features,
                               s_clip_features, 
                               s_val_clip_features,
                               save_spatial)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(s_concepts), bias=False)
    proj_layer.load_state_dict({"weight":s_W_c})
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std
        
        
        train_cs.append(train_c)
        val_cs.append(val_c)
    torch.save(train_mean, os.path.join(save_spatial, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_spatial, "proj_std.pt"))
        
    t_W_c,t_concepts = train_cocept_layer(args,
                               t_concepts,
                               target_features,
                               val_target_features,
                               t_clip_features,
                               t_val_clip_features,
                               save_temporal)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(t_concepts), bias=False)
    proj_layer.load_state_dict({"weight":t_W_c})
    with torch.no_grad():

        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std

        
        train_cs.append(train_c)
        val_cs.append(val_c)
    torch.save(train_mean, os.path.join(save_temporal, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_temporal, "proj_std.pt"))
    p_W_c,p_concepts = train_cocept_layer(args,
                               p_concepts,
                               target_features,
                               val_target_features,
                               p_clip_features,
                               p_val_clip_features,
                               save_place)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(p_concepts), bias=False)
    proj_layer.load_state_dict({"weight":p_W_c})
    with torch.no_grad():

        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std
        
        
        train_cs.append(train_c)
        val_cs.append(val_c)
    torch.save(train_mean, os.path.join(save_place, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_place, "proj_std.pt"))
    train_c = torch.cat(train_cs,dim=1)
    val_c = torch.cat(val_cs,dim=1)
    del s_clip_features, s_val_clip_features,t_clip_features, t_val_clip_features,p_clip_features, p_val_clip_features
    

    stp_W_c = torch.cat([s_W_c,t_W_c,p_W_c],dim=0)
    stp_concepts = s_concepts+t_concepts+p_concepts
    train_classification_layer(args,
                               W_c=stp_W_c,
                               pre_concepts=None,
                               concepts = stp_concepts,
                               target_features=target_features,
                               val_target_features=val_target_features,
                                save_name=save_spatio_temporal_place,
                                joint=(train_c,val_c)
                               )

    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    # d_val = args.data_set + "_test"
    # val_data_t = data_utils.get_data(d_val,args=args)
    # val_data_t.end_point = 2
    # device = torch.device(args.device)
    
    if args.debug:
        debugging.debug(args,save_name)
    
    # model,_ = cbm.load_cbm_triple(save_name, device,args)
    # print("?***? Start test")
    # accuracy = cbm_utils.get_accuracy_cbm(model, val_data_t, device,128,10)
    # print("?****? Accuracy: {:.2f}%".format(accuracy*100))
    # d_val = args.data_set + "_test"
    # val_data_t = data_utils.get_data(d_val,args=args)
    # device = torch.device(args.device)
    
    # s_model,t_model = cbm.load_cbm_two_stream(save_name, device,args)

    # print("!****! Start test Spatio")
    # accuracy = cbm_utils.get_accuracy_cbm(s_model, val_data_t, device,32,10)
    # print("!****! Spatio Accuracy: {:.2f}%".format(accuracy*100))
    
    # print("?***? Start test Temporal")
    # accuracy = cbm_utils.get_accuracy_cbm(t_model, val_data_t, device,32,10)
    # print("?****? Temporal Accuracy: {:.2f}%".format(accuracy*100))
    
    return
def spatio_temporal_three_ensemble(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            p_concepts,
                            p_clip_features,
                            p_val_clip_features,
                            save_name):
    save_spatial = os.path.join(save_name,'spatial')
    save_temporal = os.path.join(save_name,'temporal')
    save_place = os.path.join(save_name,'place')
    save_spatio_temporal_place = os.path.join(save_name,'spatio_temporal_place')
    # save_spatio_temporal = os.path.join(save_name,'spatio_temporal')
    os.mkdir(save_spatial)
    os.mkdir(save_temporal)
    os.mkdir(save_place)
    os.mkdir(save_spatio_temporal_place)
    train_cs,val_cs = [],[]
    s_W_c,s_concepts = train_cocept_layer(args,
                               s_concepts,
                               target_features,
                               val_target_features,
                               s_clip_features, 
                               s_val_clip_features,
                               save_spatial)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(s_concepts), bias=False)
    proj_layer.load_state_dict({"weight":s_W_c})
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std
        
        
        train_cs.append(train_c)
        val_cs.append(val_c)
    torch.save(train_mean, os.path.join(save_spatial, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_spatial, "proj_std.pt"))
        
    t_W_c,t_concepts = train_cocept_layer(args,
                               t_concepts,
                               target_features,
                               val_target_features,
                               t_clip_features,
                               t_val_clip_features,
                               save_temporal)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(t_concepts), bias=False)
    proj_layer.load_state_dict({"weight":t_W_c})
    with torch.no_grad():

        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std

        
        train_cs.append(train_c)
        val_cs.append(val_c)
    torch.save(train_mean, os.path.join(save_temporal, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_temporal, "proj_std.pt"))
    p_W_c,p_concepts = train_cocept_layer(args,
                               p_concepts,
                               target_features,
                               val_target_features,
                               p_clip_features,
                               p_val_clip_features,
                               save_place)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(p_concepts), bias=False)
    proj_layer.load_state_dict({"weight":p_W_c})
    with torch.no_grad():

        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std
        
        
        train_cs.append(train_c)
        val_cs.append(val_c)
    torch.save(train_mean, os.path.join(save_place, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_place, "proj_std.pt"))
    # train_c = torch.cat(train_cs,dim=1)
    # val_c = torch.cat(val_cs,dim=1)
    del s_clip_features, s_val_clip_features,t_clip_features, t_val_clip_features,p_clip_features, p_val_clip_features
    cls_file = data_utils.LABEL_FILES[args.data_set]
    d_train = args.data_set + "_train"
    d_val = args.data_set + "_val"
    train_targets = data_utils.get_targets_only(d_train,args)
    val_targets = data_utils.get_targets_only(d_val,args)
    train_y = torch.LongTensor(train_targets)
    val_y = torch.LongTensor(val_targets)
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    s_train_c,t_train_c,p_train_c = train_cs
    s_val_c,t_val_c,p_val_c = val_cs
        
        


    indexed_train_ds = IndexedTensorDataset(train_c, train_y)
    val_ds = TensorDataset(val_c,val_y)
        
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_c.shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    STEP_SIZE = 0.05
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    #concept layer to classification
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
                    val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len(classes),verbose=10)
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {}
        for key in ('lam', 'lr', 'alpha', 'time'):
            out_dict[key] = float(output_proj['path'][0][key])
        out_dict['metrics'] = output_proj['path'][0]['metrics']
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        json.dump(out_dict, f, indent=2)

    # stp_W_c = torch.cat([s_W_c,t_W_c,p_W_c],dim=0)
    # stp_concepts = s_concepts+t_concepts+p_concepts


    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    # d_val = args.data_set + "_test"
    # val_data_t = data_utils.get_data(d_val,args=args)
    # val_data_t.end_point = 2
    # device = torch.device(args.device)
    
    
    debugging.debug(args,save_name)
    
    # model,_ = cbm.load_cbm_triple(save_name, device,args)
    # print("?***? Start test")
    # accuracy = cbm_utils.get_accuracy_cbm(model, val_data_t, device,128,10)
    # print("?****? Accuracy: {:.2f}%".format(accuracy*100))
    # d_val = args.data_set + "_test"
    # val_data_t = data_utils.get_data(d_val,args=args)
    # device = torch.device(args.device)
    
    # s_model,t_model = cbm.load_cbm_two_stream(save_name, device,args)

    # print("!****! Start test Spatio")
    # accuracy = cbm_utils.get_accuracy_cbm(s_model, val_data_t, device,32,10)
    # print("!****! Spatio Accuracy: {:.2f}%".format(accuracy*100))
    
    # print("?***? Start test Temporal")
    # accuracy = cbm_utils.get_accuracy_cbm(t_model, val_data_t, device,32,10)
    # print("?****? Temporal Accuracy: {:.2f}%".format(accuracy*100))
    
    return
def spatio_temporal_parallel(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            save_name):
    save_spatial = os.path.join(save_name,'spatial')
    save_temporal = os.path.join(save_name,'temporal')
    # save_spatio_temporal = os.path.join(save_name,'spatio_temporal')
    os.mkdir(save_spatial)
    os.mkdir(save_temporal)
    s_W_c,s_concepts = train_cocept_layer(args,
                               s_concepts,
                               target_features,
                               val_target_features,
                               s_clip_features,
                               s_val_clip_features,
                               save_spatial)
    t_W_c,t_concepts = train_cocept_layer(args,
                               t_concepts,
                               target_features,
                               val_target_features,
                               t_clip_features,
                               t_val_clip_features,
                               save_temporal)

    
    del s_clip_features, s_val_clip_features,t_clip_features, t_val_clip_features
    



    train_classification_layer(args,
                               W_c=s_W_c,
                               pre_concepts=None,
                               concepts = s_concepts,
                               target_features=target_features,
                               val_target_features=val_target_features,
                                save_name=save_spatial
                               )
    if not args.only_s:
        train_classification_layer(args,
                                W_c=t_W_c,
                                pre_concepts=None,
                                concepts = t_concepts,
                                target_features=target_features,
                                val_target_features=val_target_features,
                                    save_name=save_temporal
                                )
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    d_val = args.data_set + "_test"
    val_data_t = data_utils.get_data(d_val,args=args)
    val_data_t.end_point = 2
    device = torch.device(args.device)
    
    s_model,t_model = cbm.load_cbm_two_stream(save_name, device,args)

    print("!****! Start test Spatio")
    accuracy = cbm_utils.get_accuracy_cbm(s_model, val_data_t, device,32,10)
    print("!****! Spatio Accuracy: {:.2f}%".format(accuracy*100))
    
    print("?***? Start test Temporal")
    accuracy = cbm_utils.get_accuracy_cbm(t_model, val_data_t, device,32,10)
    print("?****? Temporal Accuracy: {:.2f}%".format(accuracy*100))
    
    return


def spatio_temporal_serial(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            save_name):
    save_spatial = os.path.join(save_name,'spatial')
    save_temporal = os.path.join(save_name,'temporal')
    save_classification = os.path.join(save_name,'classification')
    os.mkdir(save_spatial)
    os.mkdir(save_temporal)
    os.mkdir(save_classification)
    s_W_c,s_concepts = train_cocept_layer(args,
                               s_concepts,
                               target_features,
                               val_target_features,
                               s_clip_features,
                               s_val_clip_features,
                               save_spatial)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(s_concepts), bias=False)
    proj_layer.load_state_dict({"weight":s_W_c})
    s_target_features = proj_layer(target_features.detach())
    s_val_target_features = proj_layer(val_target_features.detach())
    s_train_mean = torch.mean(s_target_features, dim=0, keepdim=True)
    s_train_std = torch.std(s_target_features, dim=0, keepdim=True)
    s_target_features -= s_train_mean
    s_target_features /= s_train_std
    s_val_target_features -= s_train_mean
    s_val_target_features /= s_train_std
    torch.save(s_train_mean, os.path.join(save_name,'temporal', "proj_mean.pt"))
    torch.save(s_train_std, os.path.join(save_name,'temporal', "proj_std.pt"))
    t_W_c,t_concepts = train_cocept_layer(args,
                               t_concepts,
                               s_target_features,
                               s_val_target_features,
                               t_clip_features,
                               t_val_clip_features,
                               save_temporal)

    del s_clip_features, s_val_clip_features,t_clip_features, t_val_clip_features
    



    train_classification_layer(args,
                               W_c=t_W_c,
                               pre_concepts=s_concepts,
                               concepts = t_concepts,
                               target_features=s_target_features,
                               val_target_features=s_val_target_features,
                                save_name=save_classification
                               )
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    d_val = args.data_set + "_test"
    val_data_t = data_utils.get_data(d_val,args=args)
    device = torch.device(args.device)
    
    model = cbm.load_cbm_serial(save_name, device,args)
    
    # print("?****? Start test")
    # accuracy = cbm_utils.get_accuracy_cbm(model, val_data_t, device,32,10)
    # print("?****? Accuracy: {:.2f}%".format(accuracy*100))
def spatio_temporal_single(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            save_name):
    # save_spatial = os.path.join(save_name,'spatial')
    # save_temporal = os.path.join(save_name,'temporal')
    # save_classification = os.path.join(save_name,'classification')
    # os.mkdir(save_spatial)
    # os.mkdir(save_temporal)
    # os.mkdir(save_classification)
    s_W_c,s_concepts,best_val_loss = train_cocept_layer(args,
                               s_concepts,
                               target_features,
                               val_target_features,
                               s_clip_features,
                               s_val_clip_features,
                               save_name)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(s_concepts), bias=False)
    proj_layer.load_state_dict({"weight":s_W_c})
    s_target_features = proj_layer(target_features.detach())
    s_val_target_features = proj_layer(val_target_features.detach())
    s_train_mean = torch.mean(s_target_features, dim=0, keepdim=True)
    s_train_std = torch.std(s_target_features, dim=0, keepdim=True)
    s_target_features -= s_train_mean
    s_target_features /= s_train_std
    s_val_target_features -= s_train_mean
    s_val_target_features /= s_train_std
    torch.save(s_train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(s_train_std, os.path.join(save_name, "proj_std.pt"))
    # t_W_c,t_concepts = train_cocept_layer(args,
    #                            t_concepts,
    #                            s_target_features,
    #                            s_val_target_features,
    #                            t_clip_features,
    #                            t_val_clip_features,
    #                            save_temporal)

    del s_clip_features, s_val_clip_features
    



    train_classification_layer(args,
                               W_c=s_W_c,
                               pre_concepts=s_concepts,
                               concepts = s_concepts,
                               target_features=target_features,
                               val_target_features=val_target_features,
                                save_name=save_name,
                                best_val_loss=best_val_loss
                               )
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # d_val = args.data_set + "_test"
    # val_data_t = data_utils.get_data(d_val,args=args)
    # device = torch.device(args.device)
    
    # model = cbm.load_cbm_serial(save_name, device,args)
    
    # print("?****? Start test")
    # accuracy = cbm_utils.get_accuracy_cbm(model, val_data_t, device,32,10)
    # print("?****? Accuracy: {:.2f}%".format(accuracy*100))




def soft_label(args,target_features, val_target_features,save_name):
    if args.loss_mode =='concept':
        similarity_fn = similarity.cos_similarity_cubed_single_concept
    elif args.loss_mode =='sample':
        similarity_fn = similarity.cos_similarity_cubed_single_sample
    elif args.loss_mode =='second':
        similarity_fn = similarity.cos_similarity_cubed_single_secondpower
    elif args.loss_mode =='first_concept':
        similarity_fn = similarity.cos_similarity_cubed_single_firstpower_concept
    elif args.loss_mode =='first_sample':
        similarity_fn = similarity.cos_similarity_cubed_single_firstpower_sample
    else:
        exit()
    # similarity_fn = similarity.cos_similarity_cubed_single_sample
    if args.pose_label is not None:
        file_path = args.pose_label

        with open(os.path.join(file_path,'soft_label_train.pkl'), "rb") as file:
            hard_label_train = pickle.load(file)  
        with open(os.path.join(file_path,'soft_label_val.pkl'), "rb") as file:
            hard_label_val = pickle.load(file)  
        train_attribute = [item['attribute_label'] for item in hard_label_train]
        
        train_result_tensor = torch.tensor(train_attribute,dtype=target_features.dtype)
        
        val_attribute = [item['attribute_label'] for item in hard_label_val]
        
        
        val_result_tensor = torch.tensor(val_attribute,dtype=target_features.dtype)

    if args.use_mlp:
        proj_layer = cbm.ModelOracleCtoY(n_class_attr=2, input_dim=target_features.shape[1],
                                    num_classes=train_result_tensor.shape[1])
        proj_layer = proj_layer.cuda()
        opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=train_result_tensor.shape[1],
                                  bias=False).to(args.device)
        # proj_layer.weight.data.zero_()
        # proj_layer.bias.data.zero_()
        opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)    
    indices = [ind for ind in range(len(target_features))]
    
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    # import pickle
    # result_tensor /= torch.norm(result_tensor,dim=1,keepdim=True)

    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        # if args.pose_label is not None:
        if args.use_mlp:
            loss = criterion(outs, train_result_tensor[batch].to(args.device).detach())
        else:
            loss = -similarity_fn(train_result_tensor[batch].to(args.device).detach(), outs)
   
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%10==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                if args.use_mlp:
                    val_loss = criterion(val_output, val_result_tensor.to(args.device).detach())
                else:
                    val_loss = -similarity_fn(val_result_tensor.to(args.device).detach(), val_output)
                

                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                if args.use_mlp:
                    best_weights = {
                    "linear.weight": proj_layer.linear.weight.clone(),
                    "linear.bias": proj_layer.linear.bias.clone(),
                    "linear2.weight": proj_layer.linear2.weight.clone(),
                    "linear2.bias": proj_layer.linear2.bias.clone()
                }
                else:
                    best_weights = proj_layer.weight.clone()

            elif val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_step = i
                if args.use_mlp:
                    best_weights = {
                    "linear.weight": proj_layer.linear.weight.clone(),
                    "linear.bias": proj_layer.linear.bias.clone(),
                    "linear2.weight": proj_layer.linear2.weight.clone(),
                    "linear2.bias": proj_layer.linear2.bias.clone()
                }
                else:
                    best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                pass
            print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(i, -loss.cpu(),
                                                                                            -val_loss.cpu()))
        opt.zero_grad()
    if args.use_mlp:
        proj_layer.load_state_dict(best_weights)
    else:
        proj_layer.load_state_dict({"weight":best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
    W_c = proj_layer.weight[:]
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))

    # save_classification = os.path.join(save_name,'classification')

    # os.mkdir(save_classification)
    train_classification_layer(args,
                               W_c=W_c,
                               pre_concepts=None,
                               concepts = train_result_tensor[0],
                               target_features=target_features,
                               val_target_features=val_target_features,
                                save_name=save_name,
                                best_val_loss=best_val_loss
                               )







