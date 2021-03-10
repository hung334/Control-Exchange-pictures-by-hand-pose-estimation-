import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import torchvision.transforms as tfs
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from train_hands_pose_ import Net
import cv2
from PIL import Image 

hand_pose_label=['five', 'four', 'good', 'gun', 'loveU', 'ok', 'one', 'rock', 'three', 'two']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)
net.eval()
    
net.load_state_dict(torch.load('new_save_train/best_save_net_36_0.0125.pkl'))#('save_train/best_train_save_net_36_0.0125.pkl'))



def detect(img):
    #crop_size=(128,128)
    #img = cv2.imread(image_path)
    H,W,C = img.shape
    re_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))#cv2.resize(img, crop_size)
    im_tfs = tfs.Compose([
        tfs.Resize(size=(128,128)),
        tfs.ToTensor(),
        #tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = im_tfs(re_img)
    batch = img_tensor.unsqueeze(0)
    test = Variable(batch).to(device)
    outputs = net(test)
    
    
    label_pred = outputs.max(1)[1].squeeze().cpu().data.numpy()
    ans = hand_pose_label[label_pred]
    
    #print(label_pred)
    #print(outputs)
    #print(F.softmax(outputs))
    #print(F.sigmoid(outputs))
    
    return ans



if __name__ == '__main__':
    
    image_path=r"./save_image/rock/10.jpg"
    img = cv2.imread(image_path)
    print(detect(img))
    # hand_pose_label=['five', 'four', 'good', 'ok', 'one', 'rock', 'three', 'two']
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # net = Net().to(device)
    # net.eval()
    
    # net.load_state_dict(torch.load('save_train/save_net_40_0.0125.pkl'))
    
    # image_path=r"./save_image/rock/10.jpg"
    
    
    # crop_size=(64,64)
    # img = cv2.imread(image_path)
    # H,W,C = img.shape
    # re_img = cv2.resize(img, crop_size)
    # im_tfs = tfs.Compose([
    #     #tfs.Resize(size=(64,64)),
    #     tfs.ToTensor(),
    #     #tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # img_tensor = im_tfs(re_img)
    # batch = img_tensor.unsqueeze(0)
    # test = Variable(batch).to(device)
    # outputs = net(test)
    
    
    # label_pred = outputs.max(1)[1].squeeze().cpu().data.numpy()
    # ans = hand_pose_label[label_pred]
    
    # print(ans)
    