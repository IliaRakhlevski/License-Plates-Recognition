# Testing images, prediction
import os
import sys
import ast
import cv2
import pandas as pd
from torch_snippets import *
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision
import glob
from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from difflib import SequenceMatcher


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# image sizes
ROWS = 500
COLS = 900
WHITE = [255, 255, 255]     # white color
PAD_COLOR = WHITE           # color of the added pads
IMAGE_ROOT = 'images'

# read the data received from the VGG Image Annotator and converted 
# to another more appropriated format
DF_RAW = df = pd.read_csv('df.csv')


# create classes (targets) and their labels
label2target = {l:t+1 for t,l in enumerate(DF_RAW['LabelName'].unique())}
label2target['background'] = 0
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)


# create Faster R-CNN model
def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = None, weights_backbone = None, num_classes = num_classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Add pads to image in order it to have the given size (img_width, img_height)
# image - image (frame) object
# img_width - final image width
# img_height - final image height
def image_padding(image, img_width = COLS, img_height = ROWS):
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



# Change the image in order it to have the given size (img_width, img_height)
# image - image (frame) object
# img_width - final image width
# img_height - final image height
def frame_preprocessing(image, img_width = COLS, img_height = ROWS):
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
    
    image = image_padding(image, img_width, img_height)
  
    return Image.fromarray(image)


# perform image preprocessing 
def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    return img.to(device).float()

# load image from the given file
def load_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = frame_preprocessing(np.asarray(img))
    img = np.array(img.resize((COLS, ROWS), resample=Image.BILINEAR))/255.
    img = preprocess_image(img)
    return img

# get from the "output" boxes and labels
def decode_output(output):
    'convert tensors to numpy arrays'
    bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([target2label[i] for i in output['labels'].cpu().detach().numpy()])
    confs = output['scores'].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()


# compare two charakters secuences
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
    
# create model
model = get_model().to(device)
# load weights from the "model.pt" file
model.load_state_dict(torch.load(f'model.pt', map_location=torch.device('cpu')))

similarity_sum = 0.0    # sum of all the similarities 
count = 0               # counter of images
files_list = os.listdir(IMAGE_ROOT) # list of files in the directory


for file_name in files_list:
    
    # load a new image from the given file
    car_num = file_name[:-len(".jpg")]
    img_path = IMAGE_ROOT +'/' + file_name
    image = load_image(img_path)
    tensor = image.unsqueeze(0)
    
    # evaluate the image (make prediction)
    model.eval()
    outputs = model(tensor) 
    
    text = ''
    for ix, output in enumerate(outputs):
        print("\n===============================================================================================================\n")
        
        # get from the "output" boxes and labels
        bbs, confs, labels = decode_output(output)
        print("Predicted figures: ", labels)
        
        #info = [f'{l}@{c:.2f}' for l,c in zip(labels, confs)]
        
        # plot the image with the boxes and the labels
        show(image.cpu().permute(1,2,0), bbs=bbs, texts=labels, sz=25)
        
        # sort the boxes and labels
        zipped = zip(bbs, labels)
        s = sorted(zipped, key = lambda t: t[0][0])
        sorted_bbs, sorted_labels = zip(*s)
        text = text.join([str(value) for value in sorted_labels])
        
        print("Predicted sorted figures: ", sorted_labels)
        print("Predicted car number: ", text)
        print("Real car number:      ", car_num)
        
        # count and print the similarity of the current image
        sim = similar(text, car_num)
        print("\nSimilarity: ", sim)
        similarity_sum += sim
        count += 1

aver_similarity = similarity_sum / count  # average of all the similarities
print("\n\nSimilarity Average: ", aver_similarity)    
   

    
