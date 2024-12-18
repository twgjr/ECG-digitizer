import sys
import os
import json
import argparse

def ecg_num_to_folder_num(ecg_num: int):
    return f"{int(ecg_num / 1000):02}000"

def image_args_common():
    args = ["--se" , "10"]  # random seed
    args += ["--lead_name_bbox"]  # store lead name bounding box coordinates to jsob
    args += ["--lead_bbox"]  # store lead waveform bounding box coordinates to json
    args += ["--store_config", "2"] # image information stored to json
    args += ["-r", "50"] # resolution 50 dpi

    # random attributes or distortions
    args += ["--random_print_header" , "0.8"]  # header
    args += ["--calibration_pulse", "0.5"] # part of waveform 
    args += ["--random_grid_color", "--random_grid_present", "0.8"] # grids
    args += ["--wrinkles", "-ca", "45"] # wrinkles
    args += ["--augment", "-rot", "5", "-noise", "5"]  # img augs ON, others default
    args += ["--random_padding", "--pad_inches", "1"]  # padding

    return args

def image_args_single(ecg_num:int):
    folder_num = ecg_num_to_folder_num(ecg_num)
    args = ["--input_file" , f"../../../ptb_xl/records500/{folder_num}/{ecg_num:05}_hr.dat"] 
    args += ["--header_file", f"../../../ptb_xl/records500/{folder_num}/{ecg_num:05}_hr.hea"]
    args += ["--output_directory" , f"../../../ptb_xl/records500/{folder_num}"]
    args += ["--start_index" , "-1"]
    args += image_args_common()

    return args

def image_args_batch(folder_num:int):
    folder_num_str = f"{folder_num:02}000"
    args = ["--input_directory" , f"../../../ptb_xl/records500/{folder_num_str}"] 
    args += ["--output_directory" , f"../../../ptb_xl/records500/{folder_num_str}"]
    args += image_args_common()

    return args

def image_single(ecg_num_list: list[int]):
    for ecg_num in ecg_num_list:
        print(f"Processing single image with ECG number: {ecg_num}")
        run_single_file(get_parser().parse_args(image_args_single(ecg_num)))
        # Now handle JSON files to remove plotted_pixels key
        folder_num = ecg_num_to_folder_num(ecg_num)  # This will determine the correct folder for each ECG number

def image_batch(folder_num: int):
    
    # image gen attributes or distortions
    args = image_args_batch(folder_num)
    
    run(get_parser().parse_args(args))
    
    
if __name__=="__main__":
    main_parser = argparse.ArgumentParser()
    group = main_parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--batch', type=int, help="the folder integer number")
    group.add_argument("--single", nargs='*', type=int, help="A list of ecg id integers")
    main_args = main_parser.parse_args()
    sys.path.append(f"{os.getcwd()}/ecg_image_kit/codes/ecg_image_generator")
    cwd_save= os.getcwd()
    os.chdir(os.getcwd()+"/ecg_image_kit/codes/ecg_image_generator")
    print("begin")
    print(main_args.batch)
    if(main_args.batch is not None):
        from gen_ecg_images_from_data_batch import *
        image_batch(main_args.batch)
    elif main_args.single:
        from gen_ecg_image_from_data import *
        image_single(main_args.single)
    os.chdir(cwd_save)