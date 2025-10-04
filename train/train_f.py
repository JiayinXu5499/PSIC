"""
file: train_f.py
"""
import os
import random
import logging
from PIL import ImageFile, Image
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import random
import numpy as np
sys.path.append("..")
print(sys.path)
from config.args import train_options
from config.config import model_config
from utils.dataloader import TrainCLIPData
from utils.logger import setup_logger
from utils.utils import CustomDataParallel, save_checkpoint
from utils.optimizers import configure_optimizers, configure_optimizer
from models.hy_cond import Hy2018
from loss.rd_loss import RateDistortionLoss
from loss.clip_loss import CLIP_Loss
from utils.training import train_conditional_f_epoch
from utils.testing import test_conditional_epoch 
from clip import clip
def main():
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = train_options()
    config = model_config()

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        seed = args.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU

        random.seed(seed)
        np.random.seed(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(os.path.join('../experiments', args.experiment)):
        os.makedirs(os.path.join('../experiments', args.experiment))

    setup_logger('train', os.path.join('../experiments', args.experiment), 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', os.path.join('../experiments', args.experiment), 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    tb_logger = SummaryWriter(log_dir='../tb_logger/' + args.experiment)

    if not os.path.exists(os.path.join('../experiments', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('../experiments', args.experiment, 'checkpoints'))
     #加载数据集
    train_transforms =  transforms.Compose([
        transforms.Resize(args.patch_size,interpolation=Image.NEAREST), 
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )
  
    train_annotation_path = args.data_root + 'train_dataset.txt'
    #train_annotation_path = args.data_root + 'test_dataset.txt'
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    train_dataset   = TrainCLIPData(train_lines,transform=train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == device),
    )
    test_annotation_path = args.data_root + 'test_dataset.txt'
    with open(test_annotation_path) as f:
        test_lines = f.readlines()
    test_dataset = TrainCLIPData(test_lines,transform=train_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == device),
    )
    net = Hy2018(N=192, M=320)
    #"/xujiayinT/xjy/backdoor_CLIP/clip/ViT-B-32.pt"
    #/xujiayinT/xjy/backdoor_CLIP/clip/ViT-L-14.pt
    clip_model, preprocess = clip.load(args.clip_path, device=device)
    preprocess = transforms.Compose(
            [preprocess.transforms[0], preprocess.transforms[1],preprocess.transforms[-1]])
    #if args.cuda and torch.cuda.device_count() > 1:
    #    net = CustomDataParallel(net)
    #    clip_model = CustomDataParallel(clip_model)
    net = net.to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metrics=args.metrics)
    closs = CLIP_Loss()
    if args.checkpoint != None:
      checkpoint = torch.load(args.checkpoint)
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
      lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.1)
      start_epoch = checkpoint['epoch']
      best_loss = checkpoint['loss']
      current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
      checkpoint = None

    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    logger_train.info(args)
    logger_train.info(config)
    logger_train.info(net)
    logger_train.info(optimizer)
    optimizer.param_groups[0]['lr'] = args.learning_rate
    save_dir = os.path.join('../experiments', args.experiment, 'val_images', '%03d' % (1 + 1))
    
    for epoch in range(start_epoch, args.epochs):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step = train_conditional_f_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            logger_train,
            tb_logger,
            current_step,
            clip_model, 
            preprocess,
            closs,
            args.lambda_clip,
            args.lambda_beta0,
            args.lambda_beta1,
            args.tau,
        )
        loss = test_conditional_epoch(epoch, test_dataloader, net, criterion, save_dir, logger_val, tb_logger,clip_model,preprocess,closs, args.lambda_clip)
        lr_scheduler.step()
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        net.update(force=True)
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                os.path.join('../experiments', args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch + 1))
            )
            if is_best:
                logger_val.info('best checkpoint saved.')

if __name__ == '__main__':
    main()

