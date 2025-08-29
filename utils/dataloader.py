"""
paper: 
file: dataloader.py
about: build the training dataset
author: 
date: 31/07/24
"""
# --- Imports --- #
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np

class TrainData(Dataset):
    def __init__(self,annotation_lines,transform=None):
        super(TrainData, self).__init__()
        self.annotation_lines = annotation_lines
        self.length             = len(self.annotation_lines)
        self.transform = transform
    
    def __getitem__(self, index):
        index       = index % self.length
        image_data,line  = self.get_random_data(self.annotation_lines[index])
        image      = self.transform(image_data)
        #print(image.shape,len(line))
        return image, \
               line
    def get_random_data(self,annotation_line):
        path = annotation_line.replace('\n','')+'.jpg'
        line_path = annotation_line.replace('\n','')+'.txt'
        image   = Image.open(path)
        image = image.convert('RGB') 
        with open(line_path, 'r', encoding='utf-8') as file: 
            line = file.read()  
        return image, line

    def __len__(self):
        return len(self.annotation_lines)
    
class TrainCLIPData(Dataset):
    def __init__(self,annotation_lines,transform=None):
        super(TrainCLIPData, self).__init__()
        self.annotation_lines = annotation_lines
        self.length             = len(self.annotation_lines)
        self.transform = transform
        self.noisy_inx = np.arange(self.length)
        self.im_div = 1
    def __getitem__(self, index):
        index       = index % self.length
        img_id = self.noisy_inx[int(index / self.im_div)]   
        image_data,line  = self.get_random_data(self.annotation_lines[index])
        image      = self.transform(image_data)
        if img_id == list(self.noisy_inx).index(img_id):
            label = 1
        else:
            label = 0
        return image, line, len(line), img_id, label
    
    def get_random_data(self,annotation_line):
        path = annotation_line.replace('\n','')+'.jpg'
        line_path = annotation_line.replace('\n','')+'.txt'
        image   = Image.open(path)
        image = image.convert('RGB') 
        with open(line_path, 'r', encoding='utf-8') as file: 
            line = file.read()  
        return image, line    

    def __len__(self):
        return len(self.annotation_lines)
