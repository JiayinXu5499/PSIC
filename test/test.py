# python rd.py
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
sys.path.append('../')
import torch.nn.functional as F
from PIL import Image
from models.hy_cond import Hy2018
import math
import os
import torch
import torch.nn as nn
from utils.metrics import compute_metrics, compute_metrics2
from utils.utils import *


def compress_one_image(model, x, stream_path, H, W, img_name):
    with torch.no_grad():
        out = model.compress(x)

    shape = out["shape"]
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    return bpp


def decompress_one_image(model, stream_path, img_name, beta):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)

    with torch.no_grad():
        # Convert beta to a tensor and move it to the same device as the model
        beta_tensor = torch.tensor([beta], dtype=torch.float32, device=next(model.parameters()).device)
        out = model.decompress(strings, shape, beta_tensor)

    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    return x_hat


def compute(path,net,preprocess,device,save_dir, save_dir_encode, save_dir_protect):
    bpp1 = 0
    bpp2 = 0
    psnr1 = 0
    psnr2 = 0
    count = 0
    avg_bpp= AverageMeter()
    avg_psnr_encode = AverageMeter()
    avg_psnr_protect = AverageMeter()
    net = net.to(device)
    with torch.no_grad():
        with open(path, 'r') as f:
            dataset = f.readlines()
            for line in dataset:
                image = Image.open(line.replace('\n',''))
                img = preprocess(image).unsqueeze(0)
                B, C, H, W = img.shape
                pad_h = 0
                pad_w = 0
                if H % 64 != 0:
                    pad_h = 64 * (H // 64 + 1) - H
                if W % 64 != 0:
                    pad_w = 64 * (W // 64 + 1) - W
                img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
                img_pad = img_pad.to(device)
                if count == 0:
                    bpp = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(count))
                bpp = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(count))
                avg_bpp.update(bpp)
                x_hat_encode = decompress_one_image(model=net, stream_path=save_dir, img_name=str(count), beta=0)
                rec_encode = torch2img(x_hat_encode)
                img = torch2img(img)
                rec_encode.save(os.path.join(save_dir_encode, str(count)) + ".png")
                p = compute_metrics2(rec_encode, img)
                avg_psnr_encode.update(p)
                x_hat_protect = decompress_one_image(model=net, stream_path=save_dir, img_name=str(count), beta=1)
                rec_protect = torch2img(x_hat_protect)
                rec_protect.save(os.path.join(save_dir_protect, str(count)) + ".png")
                p2 = compute_metrics2(rec_protect, img)
                avg_psnr_protect.update(p2)
                count += 1
    print(
        f"For Encode | "
        f"Avg Bpp: {avg_bpp.avg:.4f} | "
        f"Avg PSNR: {avg_psnr_encode.avg:.4f} | "
    )
    print(
        f"For Protect | "
        f"Avg Bpp: {avg_bpp.avg:.4f} | "
        f"Avg PSNR: {avg_psnr_protect.avg:.4f} | "
    )
            
if __name__ == "__main__":
    device = "cuda" if True and torch.cuda.is_available() else "cpu"
    preprocess = transforms.Compose([
        transforms.Resize([256,256], interpolation=Image.BICUBIC), 
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
    ])
    path ='data.txt'#Save the file containing the address of the dataset
    root_path = "test"#This directory
    save_dir = os.path.join(root_path, "bitstream")
    print(save_dir)
    save_dir_encode = os.path.join(root_path, "encode")
    save_dir_protect = os.path.join(root_path, "protect")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_encode):
        os.makedirs(save_dir_encode)
    if not os.path.exists(save_dir_protect):
        os.makedirs(save_dir_protect)
    
    '''Loader Model'''
    net = Hy2018(N=192, M=320)
    net = net.to(device)
    checkpoint = "experiments/q1checkpoint_best_loss.pth.tar"#Weight
    checkpoint = torch.load(checkpoint)
    #net = CustomDataParallel(net)
    net.load_state_dict(checkpoint['state_dict'])
    net = net.eval()
    print(f"Start testing!" )
    compute(path,net,preprocess,device,save_dir, save_dir_encode, save_dir_protect)

    