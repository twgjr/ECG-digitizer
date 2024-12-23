import os
import sys
import argparse

# map defining allowed titles of image folders
titles = {
    "base": "base",
}

def record_path():
    rel_path = os.path.join('..','..','data','ptb_xl','records100')
    abs_path = os.path.abspath(rel_path)
    return abs_path

def group_dir(group_num: int):
    record_path_str = record_path()
    if(group_num < 0 or group_num > 21):
        raise ValueError(f"Invalid ECG number: {ecg_num}")
    group_str = f"{group_num:02}000"
    path = os.path.join(record_path_str, group_str)
    return path

# function that retrieves the image path within each augmentation
# sub group.  'base' is the sub group without any augmentations.
def sub_group_dir(group_path_str:str, title: str):
    if(title not in titles):
        raise ValueError(f"Invalid title: {title}")

    path = os.path.join(group_path_str, title)

    # if the directory does not exist, create it
    if not os.path.exists(path):
        os.makedirs(path)

    return path


# create images within the sub group directory
# title str determines the sub group directory and the augmentations
# 'base' is the sub group without any augmentations
# https://github.com/alphanumericslab/ecg-image-kit/tree/main/codes/ecg-image-generator
def image_args_batch(sub_group_dir_str: str, group_path_str: str):
    args = ["--input_directory" , group_path_str] 
    args += ["--output_directory", sub_group_dir_str]
    args += ["--se" , "10"]  # random seed
    args += ["--store_config", "2"] # image information stored to json
    args += ["-r", "50"] # resolution 50 dpi
    args += ["--random_grid_present", "0"] # no grid
    args += ["--remove_lead_names"] # no lead names
    args += ["--calibration_pulse", "0"] # no calibration pulse
    args += ["--random_bw", "1"] # all images in black and white
    return args

if __name__=="__main__":
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('--group', type=int, help="the folder integer number (0-21)", choices=range(0, 22, 1))
    main_parser.add_argument('--title', type=str, help="the folder title (e.g. 'base')", choices=titles.keys())
    main_args = main_parser.parse_args()

    print(f'begin group: {main_args.group} title: {main_args.title}')
    group_dir_str = group_dir(main_args.group)
    print(f"group path: {group_dir_str}")
    sub_group_dir_str = sub_group_dir(group_dir_str, main_args.title)
    print(f"sub group directory: {sub_group_dir_str}")
    args = image_args_batch(sub_group_dir_str, group_dir_str)

    sys.path.append(os.path.join(os.getcwd(),'ecg_image_kit','codes','ecg_image_generator'))
    cwd_save= os.getcwd()
    os.chdir(os.path.join(cwd_save,'ecg_image_kit','codes','ecg_image_generator'))
    from ecg_image_kit.codes.ecg_image_generator.gen_ecg_images_from_data_batch import get_parser, run
    run(get_parser().parse_args(args))
    os.chdir(cwd_save)
    