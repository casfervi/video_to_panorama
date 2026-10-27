# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:25:40 2026

@author: vinicius.ferreira
"""

import cv2
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="Pipeline to crop images via cv2 and Python. \n\n"
                                    "Sugestion of use to vertical image: \n python crop_image.py --image path_to_image\image.jpg --crop_x 200 --crop_w 960 --crop_y 150 --crop_h 3400 --output_path folder_here. \n\n"
                                    "Sugestion of use to horizontal image: \n python crop_image.py --image path_to_image\image.jpg --crop_x 150 --crop_w 3400 --crop_y 200 --crop_h 960 --output_path folder_here. \n")
    parser.add_argument("--image", required=True, type=Path, help="Photos to be croped.")
    parser.add_argument("--output_path", default=Path("output"), type=Path, help="Output path (default: ./output).")
    parser.add_argument("--crop_x", required=True, type=int, help="Start crop in X (from left to right).")
    parser.add_argument("--crop_w", required=True, type=int, help="End crop in X.")
    parser.add_argument("--crop_y", required=True, type=int, help="Start crop in Y (from top to bottom).")
    parser.add_argument("--crop_h", required=True, type=int, help="End crop in Y.")

    args = parser.parse_args()

    
    output_folder = args.output_path
    output_folder.mkdir(parents=True, exist_ok=True)

    
    path_image = args.image    
    img = cv2.imread(path_image)
    
    file_name = str(path_image).replace("\\", "/")
    file_name = file_name.split("/")[-1]
    
    print(f"Cropping image {file_name}. Original size = {img.size}. Image shape = {img.shape[:2]}")
    
    file_name = file_name.split(".")[0]


    # # [rows, columns]
    crop = img[args.crop_y:args.crop_h, args.crop_x:args.crop_w]  
    print(f"Cropped image size = {crop.size} and shape = {crop.shape[:2]}. This ended up being {round(100-((crop.size*100)/img.size),2)}% of the original size. ")


    output_file_name = file_name + "_croped.jpg"       
    output_path = Path(f"{output_folder}/{output_file_name}")       
    
    status = cv2.imwrite(output_path, crop)

    if status:
        print(f"Image successfully saved in {output_path}")
    else:
        print("Error saving image.")


if __name__ == "__main__":
    main()


