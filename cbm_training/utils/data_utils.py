import os
import torch
from torchvision import datasets, transforms, models

# NOTE: OF-CBM uses only the VideoMAE backbone. CLIP / AIM backbone support is
# intentionally not included in this repository.
# import cbm_training.models.clip as clip
from cbm_training.video_dataloader.datasets import build_dataset
import cbm_training.models.modeling_videomae as modeling_finetune
from collections import OrderedDict
from timm.models import create_model
# from cbm_training.models.modeling_aim import AIM
from cbm_training.video_dataloader import video_utils
DATASET_ROOTS = {
    "imagenet_train": "YOUR_PATH/CLS-LOC/train/",
    "imagenet_val": "YOUR_PATH/ImageNet_val/",
    "cub_train":"data/CUB/train",
    "cub_val":"data/CUB/test"
}

# OF-CBM class-list files.  All paths are RELATIVE to the repo root so that
# the scripts work out-of-the-box on any machine as long as the user runs
# them from the top-level OF-CBM directory.
LABEL_FILES = {
    "kth":          "datasets/KTH/class_list.txt",
    "penn-action":  "datasets/Penn_action/class_list.txt",
    "ssv2_chiral":  "datasets/ssv2_chiral/class_list.txt",
    "multi_object": "datasets/multi_object/class_list.txt",
}

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess


def get_data(dataset_name, preprocess=None,args = None):
    if dataset_name == "UCF101_train":
        data, _= build_dataset(is_train=True, test_mode=False, args=args)
    elif dataset_name == "UCF101_val":
        data, _= build_dataset(is_train=False, test_mode=False, args=args)
    elif dataset_name == "UCF101_test":
        data, _= build_dataset(is_train=False, test_mode=True, args=args)
    elif dataset_name == "SSV2_train":
        data, _= build_dataset(is_train=True, test_mode=False, args=args)
    elif dataset_name == "SSV2_val":
        data, _= build_dataset(is_train=False, test_mode=False, args=args)
    elif dataset_name == "kth_train":
        data, _= build_dataset(is_train=True, test_mode=False, args=args)
    elif dataset_name == "kth_val":
        data, _= build_dataset(is_train=False, test_mode=False, args=args)
    elif dataset_name == "kth-5_train":
        data, _= build_dataset(is_train=True, test_mode=False, args=args)
    elif dataset_name == "kth-5_val":
        data, _= build_dataset(is_train=False, test_mode=False, args=args)
    elif dataset_name == "kth-2_train":
        data, _= build_dataset(is_train=True, test_mode=False, args=args)
    elif dataset_name == "kth-2_val":
        data, _= build_dataset(is_train=False, test_mode=False, args=args)
    elif dataset_name == "penn-action_train":
        data, _= build_dataset(is_train=True, test_mode=False, args=args)
    elif dataset_name == "penn-action_val":
        data, _= build_dataset(is_train=False, test_mode=False, args=args)
    elif dataset_name == "haa100_train":
        data, _= build_dataset(is_train=True, test_mode=False, args=args)
    elif dataset_name == "haa100_val":
        data, _= build_dataset(is_train=False, test_mode=False, args=args)
    elif dataset_name == "haa100_test":
        data, _= build_dataset(is_train=False, test_mode=False, args=args)
    elif dataset_name == "mini-SSV2_train":
        data, _= build_dataset(is_train=True, test_mode=False, args=args)
    elif dataset_name == "mini-SSV2_val":
        data, _= build_dataset(is_train=False, test_mode=False, args=args)
    elif dataset_name == "kinetics400_train":
        data, _= build_dataset(is_train=True, test_mode=False, args=args)
    elif dataset_name == "kinetics400_val":
        data, _= build_dataset(is_train=False, test_mode=False, args=args)
    elif dataset_name == "kinetics400_test":
        data, _= build_dataset(is_train=False, test_mode=True, args=args)
    elif dataset_name == "kinetics400_scratch_train":
        data, _= build_dataset(is_train=True, test_mode=False, args=args)
    elif dataset_name == "kinetics400_scratch_val":
        data, _= build_dataset(is_train=False, test_mode=False, args=args)
    elif dataset_name == "kinetics400_scratch_test":
        data, _= build_dataset(is_train=False, test_mode=True, args=args)
    elif dataset_name == "kinetics100_train":
        data, _= build_dataset(is_train=True, test_mode=False, args=args)
    elif dataset_name == "kinetics100_val":
        data, _= build_dataset(is_train=False, test_mode=False, args=args)
    elif dataset_name == "kinetics100_test":
        data, _= build_dataset(is_train=False, test_mode=True, args=args)
    elif dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
    return data

def get_targets_only(dataset_name,args=None):
    pil_data = get_data(dataset_name,args=args)
    return pil_data.label_array

def get_target_model(target_name, device,args=None):

    # NOTE: OF-CBM only supports the VideoMAE backbone; CLIP / AIM branches
    # have been removed.
    if target_name.startswith("vmae_"):
        target_model = create_model(
            target_name,
            pretrained=False,
            num_classes=args.nb_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            fc_drop_rate=args.fc_drop_rate,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_checkpoint=args.use_checkpoint,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
        )
        patch_size = target_model.patch_embed.patch_size
        print("Patch size = %s" % str(patch_size))
        args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
        args.patch_size = patch_size
        if args.finetune:
            target_model = load_vmae_weight(target_model,args)

        preprocess=None
        target_model.to(device)
    elif target_name == 'resnet18_places':
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
        
    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    
    return target_model, preprocess





def load_vmae_weight(model,args=None):
    
    if args.finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.finetune, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.finetune, map_location='cpu')

    print("Load ckpt from %s" % args.finetune)
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    # interpolate position embedding
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
        num_patches = model.patch_embed.num_patches # 
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

        # height (== width) for the checkpoint position embedding 
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size) )** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
            pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

    video_utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    print("\n\n\n*********** VMAE Load ***************")
    
    return model