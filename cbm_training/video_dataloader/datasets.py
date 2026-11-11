import os
from .ssv2 import SSVideoClsDataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'kth':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 6
    elif args.data_set == 'penn-action':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv')

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 15
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes