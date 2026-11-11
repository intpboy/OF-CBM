# Standard Library
import os
import json
from PIL import Image
from IPython.display import display, Image as IPImage

# Numerical & Scientific Libraries
import numpy as np

# PyTorch
import torch
from torch.utils.data import DataLoader

# NOTE: OF-CBM uses only the frozen VideoMAE backbone. The CLIP / LaViLa /
# InternVid dual-encoder helpers from the original DANCE repo are not included.

# Evaluation Metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# Progress Bar
from tqdm import tqdm
PM_SUFFIX = {"max":"_max", "avg":""}

def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.mean(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.amax(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    return hook

    
def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    s_concept, t_concept,p_concept = concept_set    
    # if target_name.startswith("clip_"):
    target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    # else:
    #     target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
    #                                              PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    s_concept_set_name = (s_concept.split("/")[-1]).split(".")[0] if s_concept is not None else None
    t_concept_set_name = (t_concept.split("/")[-1]).split(".")[0] if t_concept is not None else None
    p_concept_set_name = (p_concept.split("/")[-1]).split(".")[0] if p_concept is not None else None
    s_text_save_name = "{}/{}_{}.pt".format(save_dir, s_concept_set_name, clip_name.replace('/', ''))
    t_text_save_name = "{}/{}_{}.pt".format(save_dir, t_concept_set_name, clip_name.replace('/', ''))
    p_text_save_name = "{}/{}_{}.pt".format(save_dir, p_concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, s_text_save_name, t_text_save_name,p_text_save_name
def get_save_backbone_name(clip_name, target_name, d_probe, save_dir):
    target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    # else:
    #     target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
    #                                              PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name
    
def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

def get_accuracy_cbm(model, dataset, device, batch_size=250, num_workers=10):
    correct = 0
    total = 0
    model.eval()
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=10,
                                           pin_memory=True)):
        with torch.no_grad():
            #outs = target_model(images.to(device))
            outs, concept_activation = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu()==labels)
            total += len(labels)
    return correct/total



def get_off_diagonal_confusion_rate(y_true, y_pred, class_names, subset_labels):
    """
    Args:
        y_true (List[int] or np.array): Ground-truth labels
        y_pred (List[int] or np.array): Predicted labels
        class_names (List[str]): List of all class names
        subset_labels (List[str]): List of class names to display

    Returns:
        dict: {
            'subset_labels': [...],
            'off_diagonal_sum': int,
            'total': int,
            'confusion_rate': float,
            'confusion_matrix': np.ndarray
        }
    """
    subset_indices = [class_names.index(cls) for cls in subset_labels]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = np.isin(y_true, subset_indices)
    y_true_subset = y_true[mask]
    y_pred_subset = y_pred[mask]

    cm = confusion_matrix(y_true_subset, y_pred_subset, labels=subset_indices)

    correct = np.trace(cm)
    total = cm.sum()
    off_diagonal = total - correct
    confusion_rate = off_diagonal / total if total > 0 else 0.0

    return {
        'subset_labels': subset_labels,
        'off_diagonal_sum': off_diagonal,
        'total': total,
        'confusion_rate': confusion_rate,
        'confusion_matrix': cm
    }
def get_class_subset_confusion(y_true, y_pred, class_names, target_classes):
    """
    
    Args:
    y_true, y_pred: Complete prediction results
    class_names: List of all class names
    target_classes: Target classes of interest (e.g., ['pushup', 'pullup', 'situp'])
    
    Returns:
        cm (N x N confusion matrix), report (str)
    """
    target_indices = [class_names.index(cls) for cls in target_classes]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = np.isin(y_true, target_indices)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=target_indices)
    report = classification_report(y_true_filtered, y_pred_filtered, labels=target_indices, target_names=target_classes, digits=3)

    return cm, report

def get_detailed_metrics_cbm(model, dataset, device, batch_size=250, num_workers=10, class_names=None, return_raw=False):
    all_preds = []
    all_labels = []

    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    for images, labels in tqdm(dataloader):
        with torch.no_grad():
            outputs, _ = model(images.to(device))
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    correct = sum([pred == label for pred, label in zip(all_preds, all_labels)])
    total = len(all_labels)
    accuracy = correct / total

    print(f"Accuracy: {accuracy:.4f}")
    report = classification_report(all_labels, all_preds,labels=[i for i in range(dataloader.dataset.nb_classes)],target_names=class_names, digits=3)

    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    class_accuracy_dict = {}
    if class_names is not None:
        for idx, class_name in enumerate(class_names):
            class_accuracy_dict[class_name] = class_accuracies[idx]
    else:
        for idx in range(len(class_accuracies)):
            class_accuracy_dict[f"Class_{idx}"] = class_accuracies[idx]

    if return_raw:
        return report, cm, all_labels, all_preds, class_accuracy_dict
    else:
        return report, cm, class_accuracy_dict

def get_accuracy_and_concept_distribution_cbm(model,k,dataset, device, batch_size=250, num_workers=10,save_name=None):
    correct = 0
    total = 0
    num_object,num_action,num_scene=model.s_proj_layer.weight.shape[0],model.t_proj_layer.weight.shape[0],model.p_proj_layer.weight.shape[0]
    num_concept = num_object + num_action + num_scene  

    object_range = (0, num_object)  
    action_range = (num_object, num_object + num_action)
    scene_range = (num_object + num_action, num_concept)

    total_object_count = 0
    total_action_count = 0
    total_scene_count = 0
    model.eval()

    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=10,
                                           pin_memory=True)):
        with torch.no_grad():
            outs, concept_activation = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu()==labels)
            total += len(labels)
            pred_weights = model.final.weight[pred]  # shape: (batch, num_concept)
            concept_contribution = concept_activation * pred_weights  
            topk_indices = torch.topk(concept_contribution, k, dim=1).indices  # shape: (batch, topk)

            for i in range(topk_indices.size(0)):
                object_count = action_count = scene_count = 0
                for idx in topk_indices[i]:
                    if object_range[0] <= idx < object_range[1]:
                        object_count += 1
                    elif action_range[0] <= idx < action_range[1]:
                        action_count += 1
                    elif scene_range[0] <= idx < scene_range[1]:
                        scene_count += 1

                total_object_count += object_count
                total_action_count += action_count
                total_scene_count += scene_count
                
    
    return correct/total,[total_object_count,total_action_count,total_scene_count]


def get_preds_cbm(model, dataset, device, batch_size=250, num_workers=2):
    preds = []
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                           pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    return preds

def get_concept_act_by_pred(model, dataset, device):
    preds = []
    concept_acts = []
    for images, labels in tqdm(DataLoader(dataset, 500, num_workers=8, pin_memory=True)):
        with torch.no_grad():
            outs, concept_act = model(images.to(device))
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)
    concept_acts_by_pred=[]
    for i in range(torch.max(pred)+1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds==i], dim=0))
    concept_acts_by_pred = torch.stack(concept_acts_by_pred, dim=0)
    return concept_acts_by_pred


def save_vmae_video_features(model, dataloader, save_name, batch_size=1000 , device = "cuda",args = None):
    _make_save_dir(save_name)
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # for end_point in range(5):
    all_features = []
    #     dataset.end_point = end_point
    with torch.no_grad():
        for videos, labels in (dataloader):
            # t = (images.shape)[2]
            # if args.center_frame:
            #     images = images.squeeze(2)
            features = model.forward_features(videos.to(device))
            all_features.append(features.cpu())
    torch.save(torch.cat(all_features), save_name)

    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def analysis_backbone_dim(model,dataset,args,number_act=5,view_num=50):
    import numpy as np
    distribution = np.zeros(768, dtype=int)
    for i in range(len(dataset)):
        if i%100==0:
            print(f'iteration ** {i}/{len(dataset)}')
        # image,label ,path= val_pil_data[i]
        x, _ = dataset[i]
        x = x.unsqueeze(0).to(args.device)
        backbone_feat,concept_activation,outputs = model.get_feature(x)
        # outputs, _ = s_model(x)
        _, top_classes = torch.topk(outputs[0], dim=0, k=2)
        contributions = concept_activation[0]*model.final.weight[top_classes[0], :]
        high_spatial_concept = torch.topk(contributions, dim=0, k=2)[1][0]
        # top_logit_vals, top_classes = torch.topk(concept_activation[0], dim=0, k=5)
        contributions = backbone_feat[0]*model.proj_layer.weight[high_spatial_concept, :]
        _,top_dim = torch.topk(contributions, dim=0, k=number_act)
        dims = top_dim.cpu().numpy()
        distribution += np.bincount(dims, minlength=768) 
        sorted_indices = np.argsort(distribution)
        N = view_num  
        bottom_N_indices = sorted_indices[:N]  
        top_N_indices = sorted_indices[-N:][::-1]  

        bottom_N_values = distribution[bottom_N_indices]
        top_N_values = distribution[top_N_indices]

    print(f"Top {N} most activated dimensions: {top_N_indices}")
    print(f"Activation counts for most activated dimensions: {top_N_values}\n")

    print(f"Top {N} least activated dimensions: {bottom_N_indices}")
    print(f"Activation counts for least activated dimensions: {bottom_N_values}")

def save_args(args,save_name):
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def visualize_gif(image,label,path,index,img_ind):
    tensor = image
    if len(tensor.shape)>4:
        tensor = tensor[index]
    video_name = path.split('/')[-1].split('.')[0]
    gif_path = f'./gif/{img_ind}_{video_name}_{index}.gif'

    tensor = tensor.permute(1, 2, 3, 0)  # (T, H, W, C)

    tensor_np = tensor.numpy()

    images = []
    for i in range(tensor_np.shape[0]):
        frame = ((tensor_np[i] - tensor_np[i].min()) / (tensor_np[i].max() - tensor_np[i].min()) * 255).astype(np.uint8)
        frame_image = Image.fromarray(frame)
        images.append(frame_image)

    

    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
    display(IPImage(filename=gif_path))