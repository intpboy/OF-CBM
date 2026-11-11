import numpy as np
import torch
from . import cbm_utils
from . import data_utils

# CBM_training.model.cbm is a path that exists only in the original DANCE repo and is not included in OF-CBM.
# debug() / visualize_gif() in this file are never called by the OF-CBM pipeline, so
# the import failure is ignored.
try:
    import CBM_training.model.cbm as cbm
except ImportError:
    cbm = None

try:
    from PIL import Image
    from IPython.display import display, Image as IPImage
except ImportError:
    Image = None

def visualize_gif(image,label,path,index,img_ind,boring):
    tensor = image
    video_name = path.split('/')[-1].split('.')[0]
    gif_path = f'./gif/{img_ind}_{video_name}_{index}.gif'
    # if not os.path.exists(gif_path):
# Convert tensor to (T, H, W, C)
    tensor = tensor.permute(1, 2, 3, 0)  # (T, H, W, C)

    # Convert tensor to NumPy array
    tensor_np = tensor.numpy()

    # Create image list
    images = []
    for i in range(tensor_np.shape[0]):
        # Convert each frame to (H, W, C) format and scale to the 0–255 range
        frame = ((tensor_np[i] - tensor_np[i].min()) / (tensor_np[i].max() - tensor_np[i].min()) * 255).astype(np.uint8)
        image = Image.fromarray(frame)
        images.append(image)

    # Save as GIF (duration = time between frames, 100ms = 0.1 sec)
    

    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)

    # Display GIF in Jupyter
    display(IPImage(filename=gif_path))
# def dual_encoder(args,save_name)

def debug(args,save_name):
    d_val = args.data_set + "_test"
    val_data_t = data_utils.get_data(d_val,args=args)
    val_data_t.end_point = 2
    device = torch.device(args.device)

    model,_ = cbm.load_cbm_triple(save_name, device,args)
    num_object,num_action,num_scene=model.s_proj_layer.weight.shape[0],model.t_proj_layer.weight.shape[0],model.p_proj_layer.weight.shape[0]
    total_concepts = num_object + num_action + num_scene
    k=3
    print("?***? Start test")
    accuracy,concept_counts = cbm_utils.get_accuracy_and_concept_distribution_cbm(model, k, val_data_t, device,64,10,save_name)
    counted_object,counted_action,counted_scene = concept_counts
    total_samples = len(val_data_t)
    print("=== Basic Information ===")
    print(f"Total Samples: {total_samples}")
    print(f"Number of Object Concepts: {num_object}")
    print(f"Number of Action Concepts: {num_action}")
    print(f"Number of Scene Concepts: {num_scene}")
    print(f"Total Number of Concepts: {total_concepts}")
    print("=========================\n\n")
    # Calculate the average number of concepts used per sample for each attribute
    # Calculate the average usage ratio of each attribute
    avg_usage_object = counted_object / (total_samples*k)
    avg_usage_action = counted_action / (total_samples*k)
    avg_usage_scene = counted_scene / (total_samples*k)

    # Normalize each attribute's usage ratio by the number of concepts in that attribute
    normalized_usage_object = avg_usage_object / num_object
    normalized_usage_action = avg_usage_action / num_action
    normalized_usage_scene = avg_usage_scene / num_scene

    # Adjust usage ratios so that their sum equals 1
    total_normalized_usage = normalized_usage_object + normalized_usage_action + normalized_usage_scene
    ratio_object = normalized_usage_object / total_normalized_usage
    ratio_action = normalized_usage_action / total_normalized_usage
    ratio_scene = normalized_usage_scene / total_normalized_usage

    
    
    
    print(f"Average Usage of Object Concepts per Sample: {avg_usage_object:.3f}")
    print(f"Average Usage of Action Concepts per Sample: {avg_usage_action:.3f}")
    print(f"Average Usage of Scene Concepts per Sample: {avg_usage_scene:.3f}")

    print("\n=== Proportion of Concept Usage ===")
    print(f"Proportion of Object Concepts Usage: {ratio_object:.3f}")
    print(f"Proportion of Action Concepts Usage: {ratio_action:.3f}")
    print(f"Proportion of Scene Concepts Usage: {ratio_scene:.3f}")

    
    print("?****? Accuracy: {:.2f}%".format(accuracy*100))