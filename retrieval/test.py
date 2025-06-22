import argparse
import math
import random
import shutil
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
import math
import clip
sys.path.append("..")
print(sys.path)
from models.hy_cond import Hy2018
from config.config import model_config

'''Calculate PSNR'''
def PSNR(data1, data2):
    mse = torch.mean((data1 - data2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()
'''Calculate BPP'''
def bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
               for likelihoods in out_net['likelihoods'].values()).item()

'''generate images'''
def get_image(net,preprocess,data_root,path):
    psnr1 = 0
    psnr2 = 0
    bpp1 = 0 
    bpp2 = 0
    with torch.no_grad():
        with open(path, 'r') as f:
            dataset = f.readlines()
            data_len = len(dataset)//5
            for i in range(data_len):
                data = dataset[5*i]
                image_name = data.split(',')[0]
                print(image_name)
                image = Image.open(data_root + image_name)
                d = preprocess(image).unsqueeze(0).to(device)
                bs = d.size(0)
                betas1 = torch.zeros((bs,))
                betas2 = torch.ones((bs,))
                betas1 = betas1.to(device)
                betas2 = betas2.to(device)
                out_net1 = net((d, betas1))
                out_net2 = net((d, betas2))

                out_net1['x_hat'] = out_net1['x_hat'].clamp(0.0, 1.0) 
                out_net2['x_hat'] = out_net2['x_hat'].clamp(0.0, 1.0) 

                bpp1 += bpp(out_net1)
                bpp2 += bpp(out_net2)

                psnr1 += PSNR(d,out_net1['x_hat'])
                psnr2 += PSNR(d,out_net2['x_hat'])
                noise_image = out_net2['x_hat'][0].permute(1, 2, 0).detach().cpu().numpy()
                noise_image = (noise_image * 255).astype('uint8')
                r_image = Image.fromarray(noise_image)
                r_image.save('data/flickr8/protectedImages/'+image_name) #save
                out_image = out_net1['x_hat'][0].permute(1, 2, 0).detach().cpu().numpy()
                out_image = (out_image * 255).astype('uint8')
                r_image = Image.fromarray(out_image)
                r_image.save('data/flickr8/reImages/'+image_name)#save 
        print ("PSNR between original images and encoded images:", psnr1/data_len)
        print ("PSNR between original images and protected images:", psnr2/data_len)
        print ("BPP of encoded images:", bpp1/data_len)
        print ("BPP of protected Images:", bpp2/data_len) 
'''extract image features'''
def get_image_feature(data_root,path):
    image_list = []
    image_feature_list = []
    with torch.no_grad():
        with open(path, 'r') as f:
            dataset = f.readlines()
            data_len = len(dataset)
            for i in range(data_len//5):
                print(i)
                #1 image corresponding to 5 captions
                data = dataset[5*i]
                image_name = data.split(',')[0]
                image_list.append(image_name)
                print(image_name)
                image = Image.open(data_root + image_name)
                image = preprocess(image).unsqueeze(0).to(device)
                image_feature = model.encode_image(image).to('cpu')
                image_feature_list.append(image_feature)
                torch.cuda.empty_cache()
                del image_feature, image
            image_feature = torch.concatenate(image_feature_list, dim=0)
    return image_list, image_feature
'''extract text features'''
def get_text_feature(path):
    text_list = []  
    feature_list = []  
    with torch.no_grad():
        with open(path, 'r') as f:
            dataset = f.readlines()
            for data in dataset:
                image = data.split(',')[0]
                text = data.split(',')[1]
                text_list.append(text)
        len_list = len(text_list)
    with torch.no_grad():
        for i in range(20):
            text = text_list[i*len_list//20: (i+1)*len_list//20]
            text = clip.tokenize(text, truncate=True).to(device)
            feature_list.append(model.encode_text(text).to('cpu'))
    text_feature = torch.concatenate(feature_list, dim=0)
    return text_list, text_feature 
def get_accuracy_t2i(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        pred_true = 0
        if image_feature.dtype != text_feature.dtype:  
            if image_feature.dtype == torch.float32:  
                text_feature = text_feature.to(torch.float32)  
            else:  
                image_feature = image_feature.to(torch.float16) 
        image_feature = image_feature.float()  
        text_feature = text_feature.float() 
        sim = (text_feature @ image_feature.T).softmax(dim=-1)
        for i in range(text_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            true_index = i//5
            if true_index in topk:
                pred_true = pred_true + 1
        return pred_true/text_feature.shape[0]

def get_accuracy_i2t(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        pred_true = 0
        if image_feature.dtype != text_feature.dtype:  
            if image_feature.dtype == torch.float32:  
                text_feature = text_feature.to(torch.float32)  
            else:  
                image_feature = image_feature.to(torch.float16)
        image_feature = image_feature.float()  
        text_feature = text_feature.float() 
        sim = (image_feature @ text_feature.T).softmax(dim=-1)
        for i in range(image_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            for j in range(5):
                true_index = 5*i + j
                if true_index in topk:
                    pred_true = pred_true + 1
                    break
        return pred_true/image_feature.shape[0]

def get_asr_t2i(text_feature, image_feature,pimage_feature,k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        pimage_feature /= pimage_feature.norm(dim=-1, keepdim=True)
        asr = 0
        acc = 0
        if image_feature.dtype != text_feature.dtype:  
            if image_feature.dtype == torch.float32 or pimage_feature.dtype == torch.float32:  
                text_feature = text_feature.to(torch.float32)  
            else:  
                image_feature = image_feature.to(torch.float16) 
                pimage_feature = pimage_feature.to(torch.float16) 
        image_feature = image_feature.float()  
        text_feature = text_feature.float() 
        pimage_feature  = pimage_feature.float()   
        sim = (text_feature @ image_feature.T).softmax(dim=-1)
        psim = (text_feature @ pimage_feature.T).softmax(dim=-1)
        for i in range(text_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            ppred = psim[i]
            pvalues, ptopk = ppred.topk(k)
            true_index = i//5
            if true_index in topk:
                acc += 1
                if true_index not in ptopk:
                    asr += 1
        return asr/acc

 
def get_asr_i2t(text_feature, image_feature,pimage_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        pimage_feature /= pimage_feature.norm(dim=-1, keepdim=True)
        if image_feature.dtype != text_feature.dtype:  
            if image_feature.dtype == torch.float32 or pimage_feature.dtype == torch.float32:  
                text_feature = text_feature.to(torch.float32)  
            else:  
                image_feature = image_feature.to(torch.float16) 
                pimage_feature = pimage_feature.to(torch.float16) 
        image_feature = image_feature.float()  
        text_feature = text_feature.float() 
        pimage_feature  = pimage_feature.float() 
        sim = (image_feature @ text_feature.T).softmax(dim=-1)
        psim = (pimage_feature @ text_feature.T).softmax(dim=-1)
        acc = 0
        asr = 0
        for i in range(pimage_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            ppred = psim[i]
            pvalues, ptopk = ppred.topk(k)
            flag = 0
            for j in range(5):
                true_index = 5*i + j
                if true_index in topk:
                    acc = acc + 1
                    flag = 1
                    break
            if flag == 1:
                for j in range(5):
                    true_index = 5*i + j
                    if true_index in ptopk:
                        flag = 2
                        break
            if flag == 2:
                asr += 1
        return  1- asr/acc


if __name__ == "__main__":
    device = "cuda" if True and torch.cuda.is_available() else "cpu"
    preprocess = transforms.Compose([
        transforms.Resize([256,256], interpolation=Image.BICUBIC), 
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
    ])
    #device = "cuda" if False else "cpu"
    '''Loader Model'''
    net = Hy2018(N=192, M=320)
    checkpoint = "experiments/q1checkpoint_best_loss.pth.tar"#loader checkpoint 
    checkpoint = torch.load(checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net = net.eval()
    net = net.to(device)
    preprocess = transforms.Compose([
        transforms.Resize([256,256], interpolation=Image.BICUBIC), 
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
    ])
    data_root = "data/flickr8/Images/"#loader dataset
    path = "data/flickr8/captions.txt"#loader captions
    #get_image(net,preprocess,data_root,path)#If it's the first time, please run this code. If it's not the first time, please comment this code to speed up inference

    '''loader CLIP model'''
    model, preprocess = clip.load("ViT-B/32", device=device)
    #loder text feature
    text_list, text_feature = get_text_feature(path)
    text_feature = text_feature.to(device) 
    print(text_feature.shape)

    #loder encoded image feature
    data_root = "data/flickr8/reImages/"
    rimage_list, rimage_feature = get_image_feature(data_root,path)
    print(rimage_feature.shape)
    
    #loder protected image feature
    data_root ="data/flickr8/protectedImages/"
    pimage_list, pimage_feature = get_image_feature(data_root,path)
    print(pimage_feature.shape)
    #loder original image feature
    data_root ="data/flickr8/resizeImages/"
    image_list, image_feature = get_image_feature(data_root,path)

    image_feature = image_feature.to(device)  
    print("Original Images Accuracy(i2t):",get_accuracy_i2t(text_feature, image_feature, 1))
    print("Original Images Accuracy(t2i):",get_accuracy_t2i(text_feature, image_feature, 1))    

    rimage_feature = rimage_feature.to(device)  
    print("Encoded Images Accuracy(i2t):",get_accuracy_i2t(text_feature, rimage_feature, 1))
    print("Encoded Images Accuracy(t2i):",get_accuracy_t2i(text_feature, rimage_feature, 1)) 

    pimage_feature = pimage_feature.to(device)    
    print("Protected Images Accuracy(i2t):",get_accuracy_i2t(text_feature, pimage_feature, 1)) 
    print("Protected Images Accuracy(t2i):",get_accuracy_t2i(text_feature, pimage_feature, 1)) 
    
    print("Encoded Images and Protected Images ASR(i2t):",get_asr_i2t(text_feature, rimage_feature,pimage_feature, 1)) 
    print("Encoded Images and Protected Images ASR(t2i):",get_asr_t2i(text_feature, rimage_feature,pimage_feature, 1)) 











