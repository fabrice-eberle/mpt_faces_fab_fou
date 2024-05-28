import os
import shutil
import cv2 as cv
import argparse
import random

# Argument parser for the terminal prompt
parser = argparse.ArgumentParser()
parser.add_argument("--border", type=float, required=True)
parser.add_argument("--split", type=float, required=True)
args = parser.parse_args()

def add_border_to_image(image, border_factor):
    # Calculate the border size in pixels
    pixels = int(border_factor * min(image.shape[:2]))
    
    # Add mirrored border to the image
    bordered_image = cv.copyMakeBorder(image, pixels, pixels, pixels, pixels, cv.BORDER_REFLECT)
    return bordered_image, pixels

def crop_faces():
    # Walk through all subfolders of the objects folder
    for folder_name in os.listdir("objects"):
        folder_path = os.path.join("objects", folder_name)
        
        # Only take the folders without 'cropped' in its name and foldername != train/val
        # Just walk through the folder with the raw pictures data
        if os.path.isdir(folder_path) and "cropped" not in folder_name and folder_name != "train" and folder_name != "val":
            
            # Create new folder for cropped pictures  
            cropped_folder_name = folder_name + "_cropped"
            cropped_folder_path = os.path.join("objects", cropped_folder_name)
            
            # Check if a cropped folder (e.g., objects/till_cropped) already exists; if yes, delete and create new one
            if os.path.exists(cropped_folder_path):
                # Delete
                shutil.rmtree(cropped_folder_path)
            # Create new 
            os.makedirs(cropped_folder_path)
            
            # Take all files of subfolder (e.g. Till) to search for pictures
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".png"):
                    # Load picture
                    picture_path = os.path.join(folder_path, file_name)
                    picture = cv.imread(picture_path)
                    # Load CSV for each picture as well
                    csv_path = os.path.join(folder_path, file_name.replace(".png", ".csv"))

                    # Read CSV
                    if os.path.exists(csv_path):
                        with open(csv_path, "r") as csvfile:
                            lines = csvfile.readlines()
                            # Check that CSV file is not empty (first line: header; second line: face coordinates)
                            if len(lines) > 0:  
                                # Read all values of line 2 and transform them into individual int-variables
                                x, y, w, h = map(int, lines[1].strip().split(","))
                                
                                # Calculate border and add it to the picture
                                bordered_picture, border_pixels = add_border_to_image(picture, args.border)
                                
                                # Adjust the coordinates with the added border
                                x += border_pixels
                                y += border_pixels

                                # Crop face with added border
                                cropped_face = bordered_picture[y - border_pixels : y + h + border_pixels, 
                                                                x - border_pixels : x + w + border_pixels]
                                
                                # Create new name, path and save cropped picture
                                cropped_pic_name = file_name.replace(".png", "_cropped.png")
                                cropped_pic_path = os.path.join(cropped_folder_path, cropped_pic_name)
                                cv.imwrite(cropped_pic_path, cropped_face)
                                print(f"Face of {folder_name} detected in '{file_name}'; cropped successfully and saved to '{cropped_pic_path}'.")

def train_val_split():
    split_value = args.split
    # Current working directory
    cwd = os.getcwd()

    objects_folder_path = os.path.join(cwd, "objects")
    os.chdir(objects_folder_path)

    # Create path for new train/val folder
    folder_path_train = os.path.join(objects_folder_path, "train")
    folder_path_val = os.path.join(objects_folder_path, "val")

    # Check if train folder exist, if not create
    if not os.path.exists(folder_path_train):
        os.makedirs(folder_path_train)
        print("--> train_folder created")
   
   # Check if val folder exist, if not create
    if not os.path.exists(folder_path_val):
        os.makedirs(folder_path_val)
        print("--> validation_folder created")

    # Walk through all subfolders of objects
    for folder_name in os.listdir(objects_folder_path):
        folder_path = os.path.join(objects_folder_path, folder_name)
        # Only take folders with cropped in name
        if os.path.isdir(folder_path) and "cropped" in folder_name:
            # Create new (label)folder in train/val 
            label_train_folder_name = os.path.join(folder_path_train, folder_name)

            # Check if train folder already exist
            if not os.path.exists(label_train_folder_name):
                os.makedirs(label_train_folder_name)
                
            # Check if val folder already exist
            label_val_folder_name = os.path.join(folder_path_val, folder_name)
            if not os.path.exists(label_val_folder_name):
                os.makedirs(label_val_folder_name)
                

            # List all pictures and shuffle path
            cropped_pictures_list = os.listdir(folder_path)
            random.shuffle(cropped_pictures_list) 

            # Train/val split value as int
            num_val_ratio = int(len(cropped_pictures_list) * split_value)

            # Select first x pic for val folder
            val_pictures = cropped_pictures_list[:num_val_ratio] 

            # Divide pictures into folders
            for pic in cropped_pictures_list:
                current_pic_path = os.path.join(folder_path, pic)
                if pic in val_pictures:
                    new_path = os.path.join(label_val_folder_name, pic)
                else:
                    new_path = os.path.join(label_train_folder_name, pic)
                
                shutil.move(current_pic_path, new_path)

def rename():
    # Current working directory
    cwd = os.getcwd()
    folder_path_train = os.path.join(cwd, "train")
    folder_path_val = os.path.join(cwd, "val")

    # Walk through all subfolders of train and val
    for folder_path in [folder_path_train, folder_path_val]:
            for folder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, folder_name)
                # Check if cropped is in the name
                if os.path.isdir(subfolder_path) and "_cropped" in folder_name:
                    # Split name by "_"
                    name_parts = folder_name.split('_')
                    if len(name_parts) >= 2:
                        # Only take first part of the folder name
                        new_folder_name = name_parts[0]
                        # Create new folder name
                        new_subfolder_path = os.path.join(folder_path, new_folder_name)
                        # Check if folder with correct name already exist, if yes, delete (necessary for renaming)
                        if os.path.exists(new_subfolder_path):
                            shutil.rmtree(new_subfolder_path)
                        # Rename
                        os.rename(subfolder_path, new_subfolder_path)
                        
    print(f"--> Cutting and splitting all pictures successfully <-- ")

if __name__ == "__main__":
    crop_faces()
    train_val_split()
    rename()
