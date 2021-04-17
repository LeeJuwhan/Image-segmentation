import argparse # parser
import os #path
from os.path import basename #path
import numpy as np #numpy
import torch #torch
from PIL import Image #PIL
import cv2 # cv
import torchvision #torch vision
import colorsys
import random
from random import randint
import glob
import time
import natsort # sort file list by number
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as transforms

from scipy import ndimage
import shutil



print("torch vision version:" +   torchvision.__version__)
print("torch version:" + torch.__version__)
print("torch cuda version:" + torch.version.cuda)
from datasets import ImageDataset



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--root', type = str, default = './', help = 'root dir')
parser.add_argument('--data_path', type = str, default = './images/' , help = 'dataset dir')
parser.add_argument('--save_path', type = str, default = './output/' , help = 'result path')


parser.add_argument('--cuda', action = 'store_false', help = 'use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
opt = parser.parse_args()


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']



#######################################
# Networks Definition
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# if torch.cuda.is_available():    
    # print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    # model.cuda()
#######################################


#####################################################
# Set model's test mode
model.eval()

def init(data_path):
    #Dataset Loader
    transforms_ = [ transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_), 
                            batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
    return dataloader

def human_remove(image,mask,labels,file_name,save_path):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)    
    image = cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
    
    image[:,:,3] = 255.0
    channel = image.shape[2]
    area = 0    
    _,_,w,h = mask.shape
    for n in range(mask.shape[0]):
            if labels[n] == 1:                
                mask[n] = np.where(mask[n] >0.5, 255,0)                                                                                         
            else :
                continue        
    for n in range(mask.shape[0]):
            if labels[n] == 1:
                for c in range(channel):                
                    image[:,:,c] = np.where(mask[n] == 255, 0, image[:,:,c])                                                        
            else :
                continue                   
    cv2.imwrite(save_path +'seg_' + file_name ,image)    
def apply_mask(image,mask,labels,boxes,file_name,save_path):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)    
    # image = cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
    
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))    
    
    # image[:,:,3] = 255.0
    channel = image.shape[2]
    area = 0    
    _,_,w,h = mask.shape    
    segmentation_map = np.zeros((w,h,3),np.uint8)
    
    for n in range(mask.shape[0]): 
        if labels[n] == 0:
            continue
        else:
            color = COLORS[random.randrange(0,len(COLORS))]
            segmentation_map[:,:,0] = np.where(mask[n] > 0.5, COLORS[labels[n]][0], 0)
            segmentation_map[:,:,1] = np.where(mask[n] > 0.5, COLORS[labels[n]][1], 0)
            segmentation_map[:,:,2] = np.where(mask[n] > 0.5, COLORS[labels[n]][2], 0)            
            image = cv2.addWeighted(image,alpha,segmentation_map,beta,gamma,dtype = cv2.CV_8U)
        # draw the bounding boxes around the objects        
        cv2.rectangle(image, boxes[n][0], boxes[n][1],color = color ,thickness = 2)
        
        print(class_names[labels[n]])
        # put the label text above the objects
        cv2.putText(image , class_names[labels[n]], (boxes[n][0][0], boxes[n][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(save_path +'seg_2' + file_name ,image)  

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors    
    
        
def tensor2im(tensor):
    # if opt.cuda == True:
        # tensor = 127.5*(tensor[0].data.cpu().float().numpy() +1.0)
    # else:
    tensor = 127.5 *(tensor[0].data.float().numpy() +1.0)
    img = tensor.astype(np.uint8)
    img = np.transpose(tensor,(1,2,0))    
    return img

def mask_rcnn(file_list,dataloader,save_path):
    for i,batch in enumerate(dataloader):        
    
        img = batch['A']        
        print(file_list[i])
    
        #inference time using cuda
        # torch.cuda.synchronize()        
        # start = time.time()
        # if opt.cuda == True:
            # result = model(img.to('cuda'))            
        # else :        
        #use cuda
        #result = model(img.to('cuda'))
        #use cpu
        result = model(img)
            
        # torch.cuda.synchronize()                        
        # end = time.time()        
        # print("time : " + str(end-start))
        image = tensor2im(img)
        scores = list(result[0]['scores'].detach().numpy())
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > 0.965]
        thresholded_preds_count = len(thresholded_preds_inidices)        
        mask = result[0]['masks']
        mask = mask[:thresholded_preds_count]
        labels = result[0]['labels']        
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in result[0]['boxes']]
        boxes = boxes[:thresholded_preds_count]
        
        mask = mask.data.float().numpy()        
        
        # human_remove(image,mask,labels,file_list[i],save_path)
        apply_mask(image,mask,labels,boxes,file_list[i],save_path)
        


    
        
def main():
    if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
    if not os.path.exists(opt.data_path):
        os.makedirs(opt.data_path)
    file_list = os.listdir(opt.data_path)
    file_list = natsort.natsorted(file_list)    
    dataloader = init(opt.data_path)
    mask_rcnn(file_list,dataloader,opt.save_path)        

if __name__ == "__main__":    
    main()

        
