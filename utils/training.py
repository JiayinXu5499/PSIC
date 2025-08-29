import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image
import sys
sys.path.append("..")
from clip import clip
from loss.prediction import prediction
def train_conditional_f_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer,epoch, clip_max_norm, logger_train, tb_logger, current_step, clip_model, preprocess,clip_loss,lambda_clip, lambda_beta0, lambda_beta1, tau
):
    model.train()
    device = next(model.parameters()).device
    mse = nn.MSELoss()
    model = model.to(device)
    clip_model = clip_model.to(device)
    for i, (d,targets, lengths, ids, ls) in enumerate(train_dataloader):
        bs = d.size(0)
        betas = torch.randint(0, 2, (bs,))
        text_inputs = torch.cat([clip.tokenize(target,context_length=77,truncate=True) for target in targets]).to(device)
        output_text = clip_model.module.encode_text(text_inputs).float()
        prinst = betas == 0
        image_wise = betas == 1

        prinst_indices = torch.nonzero(prinst).squeeze()
        image_wise_indices = torch.nonzero(image_wise).squeeze()
      
        d = d.to(device)
        betas = betas.to(device)
        if d.size(0) == 0: 
            continue
        extracted_prinst_d = d[prinst_indices, :, :, :].to(device)
        extracted_text0_d = output_text[prinst_indices, :].to(device)
        extracted_image_d = d[image_wise_indices, :, :, :].to(device)
        extracted_text1_d = output_text[image_wise_indices, :].to(device)
        if extracted_prinst_d.size(0) == 0:
            continue
        if extracted_image_d.size(0) == 0:
            continue  

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model((d, betas))
        out_net['x_hat'] = out_net['x_hat'].clamp(0.0, 1.0) 
        out_criterion = criterion(out_net, d)
        
        # beta=0
        extracted_prinst_x_hat = out_net["x_hat"][prinst_indices, :, :, :]
        extracted_prinst_mse = mse(extracted_prinst_x_hat, extracted_prinst_d)
    
        # beta=1
        extracted_image_x_hat = out_net["x_hat"][image_wise_indices, :, :, :]
        extracted_image_mse = mse(extracted_image_x_hat, extracted_image_d)
        #print(extracted_image_d.shape,preprocess(extracted_image_d).shape)
        per_output = clip_model.module.encode_image(preprocess(extracted_image_d)).float()
        per_output1 = clip_model.module.encode_image(preprocess(extracted_image_x_hat)).float()

        #判断图片文本对是否存在噪声  
        n_idx,c_idx,i_label,t_label = prediction(per_output, extracted_text1_d,lengths, tau,device)
        c_per_output = per_output1[c_idx,:]
        c_output_text = extracted_text1_d[c_idx,:]
        n_per_output = per_output1[n_idx,:]
        n_output_text = extracted_text1_d[n_idx,:]

        #closs = (10*clip_loss(c_per_output,c_output_text,1,0)+ clip_loss(n_per_output,output_text,0,n_idx)).mean()
        closs = (10*clip_loss(c_per_output,c_output_text,1,0)+ 0.5 * (clip_loss(n_per_output,output_text,0,i_label) + clip_loss(n_output_text,per_output1,0,t_label))).mean()
        total_loss = out_criterion["bpp_loss"]+ 0.5 * (lambda_beta0 * 255 ** 2 * extracted_prinst_mse + lambda_beta1 * 255 ** 2 * extracted_image_mse)+ lambda_clip * closs
        total_loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        
        aux_loss = model.module.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        
        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), total_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: closs'), closs.item(), current_step)
            #tb_logger.add_scalar('{}'.format('[train]: closs01'),closs01.item(), current_step)
            #tb_logger.add_scalar('{}'.format('[train]: closs02'), closs02.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)

        if i % 100 == 0:
            noise_image = (d)[0].permute(1, 2, 0).detach().cpu().numpy()  # 调整通道顺序为BGR 
            noise_image = (noise_image * 255).astype('uint8')
            r_image = Image.fromarray(noise_image)
            r_image.save('/root/autodl-fs/backdoor_CLIPv1/temp/f/'+str(i)+'_image.jpg')   
            out_image = out_net['x_hat'][0].permute(1, 2, 0).detach().cpu().numpy()  # 调整通道顺序为BGR 
            out_image = (out_image * 255).astype('uint8')
            r_image = Image.fromarray(out_image)
            r_image.save('/root/autodl-fs/backdoor_CLIPv1/temp/f/'+str(i)+'_out_image.jpg') 
            if out_criterion["ms_ssim_loss"] is None:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {total_loss.item():.4f} | '
                    f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f'Clip loss: {closs.item():.4f} | '
                    #f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {total_loss.item():.4f} | '
                    f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f'Clip loss: {closs.item():.2f} | '
                    #f"Aux loss: {aux_loss.item():.2f}"
                )

    return current_step
def train_conditional_s_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer,epoch, clip_max_norm, logger_train, tb_logger, current_step, clip_model, preprocess,clip_loss, lambda_clip, lambda_beta0, lambda_beta1, tau
):
    model.train()
    device = next(model.parameters()).device
    mse = nn.MSELoss()
    model = model.to(device)
    clip_model = clip_model.to(device)
    for i, (d,targets, lengths, ids, ls) in enumerate(train_dataloader):
        bs = d.size(0)
        betas = torch.randint(0, 2, (bs,))
        text_inputs = torch.cat([clip.tokenize(target,context_length=77,truncate=True) for target in targets]).to(device)
        output_text = clip_model.module.encode_text(text_inputs).float()
        prinst = betas == 0
        image_wise = betas == 1

        prinst_indices = torch.nonzero(prinst).squeeze()
        image_wise_indices = torch.nonzero(image_wise).squeeze()
      
        d = d.to(device)
        betas = betas.to(device)
        if d.size(0) == 0: 
            continue
        extracted_prinst_d = d[prinst_indices, :, :, :].to(device)
        extracted_text0_d = output_text[prinst_indices, :].to(device)
        extracted_image_d = d[image_wise_indices, :, :, :].to(device)
        extracted_text1_d = output_text[image_wise_indices, :].to(device)
        if extracted_prinst_d.size(0) == 0:
            continue
        if extracted_image_d.size(0) == 0:
            continue  

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model((d, betas))
        out_net['x_hat'] = out_net['x_hat'].clamp(0.0, 1.0) 
        out_criterion = criterion(out_net, d)
        
        # beta=0
        extracted_prinst_x_hat = out_net["x_hat"][prinst_indices, :, :, :]
        extracted_prinst_mse = mse(extracted_prinst_x_hat, extracted_prinst_d)
    
        # beta=1
        extracted_image_x_hat = out_net["x_hat"][image_wise_indices, :, :, :]
        extracted_image_mse = mse(extracted_image_x_hat, extracted_image_d)
        #print(extracted_image_d.shape,preprocess(extracted_image_d).shape)
        per_output = clip_model.module.encode_image(preprocess(extracted_image_d)).float()
        per_output1 = clip_model.module.encode_image(preprocess(extracted_image_x_hat)).float()

        #判断图片文本对是否存在噪声  
        n_idx,c_idx,i_label,t_label = prediction(per_output, extracted_text1_d,lengths, tau,device)
        c_per_output = per_output1[c_idx,:]
        c_output_text = extracted_text1_d[c_idx,:]
        n_per_output = per_output1[n_idx,:]
        n_output_text = extracted_text1_d[n_idx,:]
        c_loss = (clip_loss(c_per_output,c_output_text,1,0)).mean()
        n_loss = 0.5* (clip_loss(n_per_output,output_text,0,i_label) + clip_loss(n_output_text,per_output1,0,t_label)).mean()
        #n_loss = 0.5* (clip_loss(n_per_output,output_text,0,n_idx) + clip_loss(n_output_text,per_output1,0,n_idx)).mean()
        w1 = 1 / (c_loss + 1e-8)
        w2 = 1 / (n_loss + 1e-8)
        total_weight = w1+w2
        w1,w2 = w1/total_weight, w2/total_weight
        closs = w1 *c_loss + w2 * n_loss
        total_loss = out_criterion["bpp_loss"]+ 0.5 * (lambda_beta0 * 255 ** 2 * extracted_prinst_mse + lambda_beta1 * 255 ** 2 * extracted_image_mse)+ lambda_clip * closs
        total_loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        
        #aux_loss = model.module.aux_loss()
        #aux_loss.backward()
        #aux_optimizer.step()
        
        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), total_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: closs'), closs.item(), current_step)
            #tb_logger.add_scalar('{}'.format('[train]: closs01'),closs01.item(), current_step)
            #tb_logger.add_scalar('{}'.format('[train]: closs02'), closs02.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)

        if i % 100 == 0:
            noise_image = (d)[0].permute(1, 2, 0).detach().cpu().numpy()  # 调整通道顺序为BGR 
            noise_image = (noise_image * 255).astype('uint8')
            r_image = Image.fromarray(noise_image)
            r_image.save('/root/autodl-fs/backdoor_CLIPv1/temp/s/'+str(i)+'_image.jpg')   
            out_image = extracted_image_x_hat[0].permute(1, 2, 0).detach().cpu().numpy()  # 调整通道顺序为BGR 
            out_image = (out_image * 255).astype('uint8')
            r_image = Image.fromarray(out_image)
            r_image.save('/root/autodl-fs/backdoor_CLIPv1/temp/s/'+str(i)+'_out_image.jpg') 
            if out_criterion["ms_ssim_loss"] is None:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {total_loss.item():.4f} | '
                    f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f'Clip loss: {closs.item():.4f} | '
                    #f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {total_loss.item():.4f} | '
                    f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f'Clip loss: {closs.item():.2f} | '
                    #f"Aux loss: {aux_loss.item():.2f}"
                )

    return current_step

def train_conditional_es_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer,epoch, clip_max_norm, logger_train, tb_logger, current_step, clip_model, preprocess,clip_loss, lambda_clip, lambda_beta0, lambda_beta1, tau
):
    model.train()
    device = next(model.parameters()).device
    mse = nn.MSELoss()
    model = model.to(device)
    clip_model = clip_model.to(device)
    for i, (d,targets, lengths, ids, ls) in enumerate(train_dataloader):
        bs = d.size(0)
        betas = torch.randint(0, 2, (bs,))
        text_inputs = torch.cat([clip.tokenize(target,context_length=77,truncate=True) for target in targets]).to(device)
        output_text = clip_model.module.encode_text(text_inputs).float()
        prinst = betas == 0
        image_wise = betas == 1

        prinst_indices = torch.nonzero(prinst).squeeze()
        image_wise_indices = torch.nonzero(image_wise).squeeze()
      
        d = d.to(device)
        betas = betas.to(device)
        if d.size(0) == 0: 
            continue
        extracted_prinst_d = d[prinst_indices, :, :, :].to(device)
        extracted_text0_d = output_text[prinst_indices, :].to(device)
        extracted_image_d = d[image_wise_indices, :, :, :].to(device)
        extracted_text1_d = output_text[image_wise_indices, :].to(device)
        if extracted_prinst_d.size(0) == 0:
            continue
        if extracted_image_d.size(0) == 0:
            continue  

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model((d, betas))
        out_net['x_hat'] = out_net['x_hat'].clamp(0.0, 1.0) 
        out_criterion = criterion(out_net, d)
        
        # beta=0
        extracted_prinst_x_hat = out_net["x_hat"][prinst_indices, :, :, :]
        extracted_prinst_mse = mse(extracted_prinst_x_hat, extracted_prinst_d)
    
        # beta=1
        extracted_image_x_hat = out_net["x_hat"][image_wise_indices, :, :, :]
        extracted_image_mse = mse(extracted_image_x_hat, extracted_image_d)
        #print(extracted_image_d.shape,preprocess(extracted_image_d).shape)
        per_output = clip_model.module.encode_image(preprocess(extracted_image_d)).float()
        per_output1 = clip_model.module.encode_image(preprocess(extracted_image_x_hat)).float()

        #判断图片文本对是否存在噪声  
        n_idx,c_idx,i_label,t_label = prediction(per_output, extracted_text1_d,lengths, tau,device)
        c_per_output = per_output1[c_idx,:]
        c_output_text = extracted_text1_d[c_idx,:]
        n_per_output = per_output1[n_idx,:]
        n_output_text = extracted_text1_d[n_idx,:]
        c_loss = (clip_loss(c_per_output,c_output_text,1,0)).mean()
        n_loss = 0.5* (clip_loss(n_per_output,output_text,0,i_label) + clip_loss(n_output_text,per_output1,0,t_label)).mean()
        closs = (10*clip_loss(c_per_output,c_output_text,1,0)+ 0.5 * (clip_loss(n_per_output,output_text,0,i_label) + clip_loss(n_output_text,per_output1,0,t_label))).mean()
        #n_loss = 0.5* (clip_loss(n_per_output,output_text,0,n_idx) + clip_loss(n_output_text,per_output1,0,n_idx)).mean()
        #w1 = 1 / (c_loss + 1e-8)
        #w2 = 1 / (n_loss + 1e-8)
        #total_weight = w1+w2
        #w1,w2 = w1/total_weight, w2/total_weight
        #closs = w1 *c_loss + w2 * n_loss
        total_loss = out_criterion["bpp_loss"]+ 0.5 * (lambda_beta0 * 255 ** 2 * extracted_prinst_mse + lambda_beta1 * 255 ** 2 * extracted_image_mse)+ lambda_clip * closs
        total_loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        
        #aux_loss = model.module.aux_loss()
        #aux_loss.backward()
        #aux_optimizer.step()
        
        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), total_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: closs'), closs.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
            tb_logger.add_scalar('{}'.format('[train]: prinst_mse_loss'), extracted_prinst_mse.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: image_loss'), extracted_image_mse.item(), current_step)

        if i % 100 == 0:
            noise_image = (extracted_prinst_d)[0].permute(1, 2, 0).detach().cpu().numpy()  # 调整通道顺序为BGR 
            noise_image = (noise_image * 255).astype('uint8')
            r_image = Image.fromarray(noise_image)
            r_image.save('/root/autodl-fs/backdoor_CLIPv1/temp/es/'+str(i)+'_image0.jpg')
            noise_image = (extracted_image_d)[0].permute(1, 2, 0).detach().cpu().numpy()  # 调整通道顺序为BGR 
            noise_image = (noise_image * 255).astype('uint8')
            r_image = Image.fromarray(noise_image)
            r_image.save('/root/autodl-fs/backdoor_CLIPv1/temp/es/'+str(i)+'_image1.jpg')
            out_image = extracted_prinst_x_hat[0].permute(1, 2, 0).detach().cpu().numpy()  # 调整通道顺序为BGR 
            out_image = (out_image * 255).astype('uint8')
            r_image = Image.fromarray(out_image)
            r_image.save('/root/autodl-fs/backdoor_CLIPv1/temp/es/'+str(i)+'_out0_image.jpg') 
            out_image = extracted_image_x_hat[0].permute(1, 2, 0).detach().cpu().numpy()  # 调整通道顺序为BGR 
            out_image = (out_image * 255).astype('uint8')
            r_image = Image.fromarray(out_image)
            r_image.save('/root/autodl-fs/backdoor_CLIPv1/temp/es/'+str(i)+'_out1_image.jpg') 
            if out_criterion["ms_ssim_loss"] is None:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {total_loss.item():.4f} | '
                    f'MSE loss: {extracted_prinst_mse.item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f'Backdoor MSE loss: {extracted_image_mse.item():.4f} | '
                    f'Clip loss: {closs.item():.2f} | '
                    # f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {total_loss.item():.4f} | '
                    f'MS-SSIM loss: {extracted_prinst_mse.item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f'Backdoor MS-SSIM loss: {extracted_prinst_mse.item():.4f} | '
                    f'Clip loss: {closs.item():.2f} | '
                    # f"Aux loss: {aux_loss.item():.2f}"
                )

    return current_step
