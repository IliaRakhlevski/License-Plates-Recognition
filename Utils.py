# Some useful functions for image processing
import os
import random
import cv2

# directory containing images to be processed
IMAGE_ROOT = 'images'

# image sizes
ROWS = 500
COLS = 900


# rename all the file in the given directory
# input_dir - directory containing images
# base_name - base name of the files
def rename_files_images_dir(input_dir, base_name, i_start = 0):
    
    new_file = ""
    i = i_start
    files_list = os.listdir(input_dir)
    random.shuffle(files_list)
    
    for f in files_list:
        old_file_path = os.path.join(input_dir, f)
        if len(base_name) > 0 :
            new_file = "{0}_{1}.jpg".format(base_name, i)
        else:
            new_file = "{0}.jpg".format(i)
        new_file_path = os.path.join(input_dir, new_file)
        os.rename(old_file_path, new_file_path)
        i += 1
 

# find maximal sizes (height and width) from all the images
# input_dir - directory containing images
def get_max_sizes(input_dir):
    max_height = 0
    max_width = 0
    files_list = os.listdir(input_dir)
    for file_name in files_list:
        im_path = input_dir + "/" + file_name
        img_cv2 = cv2.imread(im_path)
        im_cur_height = img_cv2.shape[0]
        im_cur_width = img_cv2.shape[1]
        if im_cur_height > ROWS or im_cur_width > COLS:
            print(f'File: {file_name}, sizes: {im_cur_height}, {im_cur_width}  <<---')
        else:
            print(f'File: {file_name}, sizes: {im_cur_height}, {im_cur_width}')
        if im_cur_height > max_height:
            max_height = im_cur_height
        if im_cur_width > max_width:
            max_width = im_cur_width
            

    print(f'Max Height: {max_height}, Max Width: {max_width}')


if __name__ == '__main__':  
    
    get_max_sizes("C:/Users/rakhl/Documents/Python Scripts/License_Plate_Recognition/Datasets/New Data")
      
    #rename_files_images_dir("C:/Users/rakhl/Documents/Python Scripts/License_Plate_Recognition/Datasets/New Data", "img", 590)
