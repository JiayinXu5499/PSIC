import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from utils.metrics import compute_metrics, compute_metrics2
from utils.utils import *
import sys
sys.path.append("..")
from loss.prediction import prediction

def test_conditional_epoch(epoch, test_dataloader, model, criterion, save_dir, logger_val, tb_logger, clip_model, preprocess,clip_loss, lambda_clip):
    model.eval()
    clip_model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()
    cliploss = AverageMeter()
    with torch.no_grad():
        for i, (d,targets, lengths, ids, ls) in enumerate(test_dataloader):
            bs = d.size(0)
            betas = torch.ones((bs,))
            d = d.to(device)
            betas = betas.to(device)
            text_inputs = torch.cat([clip.tokenize(target,context_length=77,truncate=True) for target in targets]).to(device)
            clean_output_text = clip_model.module.encode_text(text_inputs).float()
            out_net = model((d, betas))
            out_net['x_hat'] = out_net['x_hat'].clamp(0.0, 1.0) 
            out_criterion = criterion(out_net, d)
            x_output = clip_model.module.encode_image(preprocess(d)).float()
            x_output1 = clip_model.module.encode_image(preprocess(out_net['x_hat'])).float()
            #判断图片文本对是否存在噪声  
            tau=0.1
            n_idx,c_idx,i_label,t_label = prediction(x_output,clean_output_text,lengths, tau,device)
            c_per_output = x_output1[c_idx,:]
            c_output_text = clean_output_text[c_idx,:]
            n_per_output = x_output1[n_idx,:]
            n_output_text = clean_output_text[n_idx,:]
            #closs = (10 * clip_loss(c_per_output,c_output_text,1,0) + clip_loss(n_per_output,clean_output_text,0,n_idx)).mean()
            #closs = (10 * clip_loss(c_per_output,c_output_text,1,0) + 0.5 * (clip_loss(n_per_output,clean_output_text,0,i_label) + clip_loss(n_output_text,x_output1,0,t_label))).mean()
            c_loss = (clip_loss(c_per_output,c_output_text,1,0)).mean()
            n_loss = 0.5* (clip_loss(n_per_output,clean_output_text,0,i_label) + clip_loss(n_output_text,x_output1,0,t_label)).mean()
            w1 = 1 / (c_loss + 1e-8)
            w2 = 1 / (n_loss + 1e-8)
            total_weight = w1+w2
            w1,w2 = w1/total_weight, w2/total_weight
            closs = w1 *c_loss + w2 * n_loss
            aux_loss.update(model.module.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            cliploss.update(closs)
            loss.update(out_criterion["mse_loss"]+(lambda_clip * closs))
            if out_criterion["mse_loss"] is not None:
                mse_loss.update(out_criterion["mse_loss"])
            if out_criterion["ms_ssim_loss"] is not None:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

            rec = torch2img(out_net['x_hat'][0])
            img = torch2img(d[0])
            p, m = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)

    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: cliploss'), cliploss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)

    if out_criterion["mse_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.6f} | "
            f"CLIP loss: {cliploss.avg:.6f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar('{}'.format('[val]: mse_loss'), mse_loss.avg, epoch + 1)
    if out_criterion["ms_ssim_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MS-SSIM loss: {ms_ssim_loss.avg:.6f} | "
            f"CLIP loss: {cliploss.avg:.6f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar('{}'.format('[val]: ms_ssim_loss'), ms_ssim_loss.avg, epoch + 1)
    return loss.avg

def test_conditional_no_epoch(epoch, test_dataloader, model, criterion, save_dir, logger_val, tb_logger, clip_model, preprocess,clip_loss, lambda_clip):
    model.eval()
    clip_model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()
    cliploss = AverageMeter()
    with torch.no_grad():
        for i, (d,targets, lengths, ids, ls) in enumerate(test_dataloader):
            bs = d.size(0)
            betas = torch.ones((bs,))
            d = d.to(device)
            betas = betas.to(device)
            text_inputs = torch.cat([clip.tokenize(target,context_length=77,truncate=True) for target in targets]).to(device)
            clean_output_text = clip_model.encode_text(text_inputs).float()
            out_net = model((d, betas))
            out_criterion = criterion(out_net, d)
            x_output = clip_model.encode_image(preprocess(out_net["x_hat"])).float()
            closs = (10 * clip_loss(x_output,clean_output_text,1,0)).mean()

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            cliploss.update(closs)
            loss.update(out_criterion["mse_loss"]+(lambda_clip * closs))
            if out_criterion["mse_loss"] is not None:
                mse_loss.update(out_criterion["mse_loss"])
            if out_criterion["ms_ssim_loss"] is not None:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

            rec = torch2img(out_net['x_hat'][0])
            img = torch2img(d[0])
            p, m = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)

    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: cliploss'), cliploss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)

    if out_criterion["mse_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.6f} | "
            f"CLIP loss: {cliploss.avg:.6f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar('{}'.format('[val]: mse_loss'), mse_loss.avg, epoch + 1)
    if out_criterion["ms_ssim_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MS-SSIM loss: {ms_ssim_loss.avg:.6f} | "
            f"CLIP loss: {cliploss.avg:.6f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar('{}'.format('[val]: ms_ssim_loss'), ms_ssim_loss.avg, epoch + 1)
    return loss.avg

