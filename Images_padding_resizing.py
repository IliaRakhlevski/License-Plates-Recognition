# Resizing images by padding
import cv2
import os


# directories
IMAGES_DIR = "images" # original images
PROCESSED_IMAGES_DIR = "processed_images" # processed images

# image sizes
ROWS = 500
COLS = 900
WHITE = [255, 255, 255]     # white color
PAD_COLOR = WHITE   # color of the added pads


# Add pads to image in order it to have the given size (img_width, img_height)
# image - image (frame) object
# img_width - final image width
# img_height - final image height
def image_padding(image, img_width, img_height):
    im_cur_height = image.shape[0]
    im_cur_width = image.shape[1]
    
    top = 0
    bottom = 0
    left = 0
    right = 0
    
    if im_cur_height < img_height:
        deltha = (int)((img_height - im_cur_height) // 2)
        top = deltha
        bottom = deltha
        if (top + bottom + im_cur_height) < img_height:
            top += 1
        
    if im_cur_width < img_width:
        deltha = (int)((img_width - im_cur_width) // 2)
        left = deltha
        right = deltha
        if (left + right + im_cur_width) < img_width:
            left += 1
    
    pad_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_COLOR)
    
    return pad_image



# Process the image (frame)
# image - image (frame) object
# img_width - final image width
# img_height - final image height
def frame_preprocessing(image, img_width, img_height):
    im_cur_height = image.shape[0]
    im_cur_width = image.shape[1]
    
    if im_cur_height > img_height or im_cur_width > img_width:
        height_factor = im_cur_height / img_height
        width_factor  = im_cur_width  / img_width
        
        factor = width_factor
        if height_factor > width_factor:
            factor = height_factor
            
        im_cur_height = int(im_cur_height / factor)
        im_cur_width  = int(im_cur_width  / factor)
        
        image = cv2.resize(image, (im_cur_width, im_cur_height))
    
    # padding
    image = image_padding(image, img_width, img_height)
  
    return image


# process the images
# files - list of files to be processed
def images_processing(files):
    
    for file_name in files:
        im_path = IMAGES_DIR+"/"+file_name
        img_cv2 = cv2.imread(im_path)
        img_cv2 = frame_preprocessing(img_cv2, COLS, ROWS)
        #img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        pr_file = os.path.join(PROCESSED_IMAGES_DIR, f"{file_name}")  
        cv2.imwrite(pr_file, img_cv2)


# create the directory for processed images (if missing)
if not os.path.exists(PROCESSED_IMAGES_DIR):
    os.makedirs(PROCESSED_IMAGES_DIR)
    
# get list of files in the given directory
images_files = os.listdir(IMAGES_DIR)    
 
# process (resize) images   
images_processing(images_files)















