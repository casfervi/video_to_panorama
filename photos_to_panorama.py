# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:49:35 2026

@author: vinicius.ferreira
"""



import cv2
import glob
import matplotlib.pyplot as plt
import math
import argparse
from pathlib import Path



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="Pipeline to stitch images via cv2 and Python. \n\n"
                                    """Sugestion of use: \n python photos_to_panorama.py --image_folder path/to/image_folder --show_img True \n\n""")
    parser.add_argument("--image_folder", required=True, type=Path, help="Folder with photos to be stitched.")
    parser.add_argument("--output_path", default=Path("output"), type=Path, help="Output path (default: ./output).")
    parser.add_argument("--show_img", default=False, help="Print the images while the script is running (default: False)" ) 

    args = parser.parse_args()

    
    output_folder = args.output_path
    output_folder.mkdir(parents=True, exist_ok=True)

    show_img = args.show_img

    image_folder = args.image_folder
    imagefiles = glob.glob(f"{image_folder}/*")
    imagefiles.sort()

    images = []
    for filename in imagefiles:
        print(f"Reading image {filename}.")
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    

    
    output_file = str(filename).replace("\\", "/")
    output_file = output_file.split("/")[-2]
        
    output_file_name = output_file + "_stitched.jpg"       
    output_path = Path(f"{output_folder}/{output_file_name}")  
    
    num_images = len(images)
    print(f"I found {num_images} to be stitched.")


    plt.figure(figsize = [30,10])
    num_cols = 4
    num_rows = math.ceil(num_images / num_cols)

    for i in range(0, num_images):
        plt.subplot(num_rows, num_cols, i+1)
        plt.axis("off")
        plt.imshow(images[i])
        # plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    
    if show_img:
        plt.tight_layout()
        plt.show() 


    # stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
    stitcher = cv2.Stitcher_create(cv2.STITCHER_SCANS)
    status, result = stitcher.stitch(images)
    
    if status == 0:
        plt.figure(figsize = [30,10])
        plt.imshow(result)
        plt.imsave(output_path, result)
        print(f"Image save in {output_path}.")
        
        if show_img:
            plt.tight_layout()
            plt.show() 
        
    else:
        print("Error!")
        
    


if __name__ == "__main__":
    main()
