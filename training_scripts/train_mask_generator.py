import os
import math
import sys 
sys.path.append('..')
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess
from models.positional_encoder import PositionalEncoder
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple
import torchvision.transforms as T
import json
from PIL import Image


import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from models.hack_unet2d import Hack_UNet2DConditionModel as UNet2DConditionModel

from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPProcessor

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from dataset.dataset import  collate_fn, InvPaintingDataset
from models.positional_encoder import get_embedder

from utils.util import save_videos_grid, zero_rank_print
from utils.text_wrapper import *



import pdb
import numpy as np





def init_dist(launcher="slurm", backend='nccl', port=28888, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank


def get_parameters_without_gradients(model):
    """
    Returns a list of names of the model parameters that have no gradients.

    Args:
    model (torch.nn.Module): The model to check.
    
    Returns:
    List[str]: A list of parameter names without gradients.
    """
    no_grad_params = []
    for name, param in model.named_parameters():
        print(f"{name} : {param.grad}")
        if param.grad is None:
            no_grad_params.append(name)
    return no_grad_params


def main(
    image_finetune: bool,
    
    name: str,
    use_wandb: bool,
    launcher: str,
    additional_input: str,
    binary_output: bool,
    binary_threshold: float,
    
    output_dir: str,
    pretrained_model_path: str,
    description: str,
        
    train_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    feature_type: str = None,

    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 8,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,
    PE_type: str = None,

    global_seed: int = 42,
    is_debug: bool = False,
    win_size: int = None,

    PE_time_max: int = None,
    PE_time_interval: int = None,
    use_PE: bool = False,
):
    check_min_version("0.21.4")

    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher, port=28888)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    # num_processes   = 0
    is_main_process = global_rank == 0


    seed = global_seed + global_rank

    # Set the random seed
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    sample_num = train_data['sample_num']
    pad_mode = train_data['pad_mode']

    prefix = feature_type
    
    if binary_output:
        prefix = feature_type + f"+binary_{binary_threshold}"

    prefix += f"+{additional_input}"

    
    folder_name = f'{prefix}-{pad_mode}_snum{sample_num}_{folder_name}'


    if use_PE:
        folder_name = f"PE_{PE_type}_max{PE_time_max}_interval{PE_time_interval}-" + folder_name
    


    if enable_xformers_memory_efficient_attention:
        folder_name = "xf-" + folder_name



    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="mask generation", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
        
        # print(description)


    import lpips 
    lpips_fn_alex = lpips.LPIPS(net='alex', spatial=True) # best forward scores

    
    from unet_2d.unet_2d_condition import UNet2DConditionModel
    pretrained_model_path = "../base_ckpt/realisticVisionV51_v51VAE"
    config = UNet2DConditionModel.load_config(pretrained_model_path + '/unet')

    config["out_channels"] = 1
    config["in_channels"] = 4 + 4




    if 'lpips_diff' in additional_input:
        config["in_channels"] += 1


    

    if 'noise' in additional_input:
        config["in_channels"] += 1

    if not 'text' in additional_input:
        config["cross_attention_dim"] = None

    from transformers import AutoTokenizer, PretrainedConfig


    tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path,
            subfolder="tokenizer",
            use_fast=False,
        )
    
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_path, revision=None)

    text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_path, subfolder="text_encoder"
    )


        
    RP = UNet2DConditionModel.from_config(config)

  
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")




    
    if use_PE:
        if PE_type == "abs":
            PE = PositionalEncoder(21)
        elif PE_type == "rel":
            PE = PositionalEncoder(42)


    # Freeze vae and text_encoder
    RP.requires_grad_(True)
    vae.requires_grad_(False)
    
    text_encoder.requires_grad_(False)

    if use_PE:
        PE.requires_grad_(True)

    

    trainable_params = list(filter(lambda p: p.requires_grad, RP.parameters()))
    
        
    if use_PE:
        trainable_params += list(filter(lambda p: p.requires_grad, PE.parameters()))
        
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")


    if enable_xformers_memory_efficient_attention:
            # if is_xformers_available():
                RP.enable_xformers_memory_efficient_attention()
                
            # else:
            #     raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Move models to GPU

    lpips_fn_alex.to(local_rank)


    RP.to(local_rank)
    vae.to(local_rank)
    text_encoder.to(local_rank)
  
    if use_PE:
        PE.to(local_rank)
    
    # Get the training dataset
    train_dataset = InvPaintingDataset(**train_data, is_image=image_finetune, PE_type=PE_type, is_train=True, PE_time_interval=PE_time_interval, PE_time_max=PE_time_max, win_size=win_size)

    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    # Get the validation dataset


    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)


    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )



    RP = DDP(RP, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if use_PE:
        PE = DDP(PE, device_ids=[local_rank], output_device=local_rank)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0
    best_loss = 1000000
    best_epoch = -1

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        RP.train()

        if use_PE:
            PE.train()

        for step, batch in enumerate(train_dataloader):
         
            ### >>>> Training >>>> ###
            
            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank)
            pixel_values_cur = batch["pixel_values_cur"].to(local_rank)
            clip_ref_image = batch["clip_ref_image"].to(local_rank)
            pixel_values_ref_img = batch["pixel_values_ref_img"].to(local_rank)
            drop_image_embeds = batch["drop_image_embeds"].to(local_rank) # torch.Size([bs])
            drop_time_step = batch["drop_time_step"].to(local_rank) # torch.Size([bs])
            cur_img_pos = batch["cur_img_pos"].to(local_rank) # torch.Size([bs, 2])
            next_img_pos = batch["next_img_pos"].to(local_rank) # torch.Size([bs, 2])

            cur_img_paths = batch["cur_img_path"]
            next_img_paths = batch["gt_img_path"]
            ref_img_paths = batch["ref_img_path"]
            
            
 
            video_length = pixel_values.shape[1]



            # Sample noise that we'll add to the latents
            bsz = pixel_values_ref_img.shape[0]
            
            # Get the text embedding for conditioning
            with torch.no_grad():



                assert bsz == 1


                gt_next_img = Image.open(next_img_paths[0])
                ref_img = Image.open(ref_img_paths[0])

                if cur_img_paths[0] == 'white':
                    # white image PIL same shape as ref_img
                    cur_img = Image.new('RGB', ref_img.size, (255, 255, 255))
                else:
                    cur_img = Image.open(cur_img_paths[0])
                    



                cur_img = T.ToTensor()(cur_img).unsqueeze(0).to(local_rank)
                gt_next_img = T.ToTensor()(gt_next_img).unsqueeze(0).to(local_rank)
                ref_img = T.ToTensor()(ref_img).unsqueeze(0).to(local_rank)



                # pad the border of them to make it multiple of 16, shape is [1, 3, H, W]
                pad_size = [16 - cur_img.shape[2] % 16, 16 - cur_img.shape[3] % 16]

                # if pad_size[0] != 16 or pad_size[1] != 16:
                if pad_size[0] == 16:
                    pad_size[0] = 0
                if pad_size[1] == 16:
                    pad_size[1] = 0
                
                cur_img = F.pad(cur_img, (0, pad_size[1], 0, pad_size[0]), value=0)
                gt_next_img = F.pad(gt_next_img, (0, pad_size[1], 0, pad_size[0]), value=0)
                ref_img = F.pad(ref_img, (0, pad_size[1], 0, pad_size[0]), value=0)


                cur_img_norm = (cur_img - 0.5) / 0.5
                gt_next_img_norm = (gt_next_img - 0.5) / 0.5
                ref_img_norm = (ref_img - 0.5) / 0.5
                
                lpips_cur_gt = lpips_fn_alex(cur_img_norm, gt_next_img_norm)
                lpips_cur_ref = lpips_fn_alex(cur_img_norm, ref_img_norm)

                # clip to 0, 1
                lpips_cur_gt = torch.clamp(lpips_cur_gt, 0, 1)
                lpips_cur_ref = torch.clamp(lpips_cur_ref, 0, 1)

        
            latents_ref_img = vae.encode(pixel_values_ref_img).latent_dist
            latents_ref_img = latents_ref_img.sample()
            latents_ref_img = latents_ref_img * 0.18215


        

            latents_cur = vae.encode(pixel_values_cur).latent_dist
            latents_cur = latents_cur.sample()
            latents_cur = latents_cur * 0.18215


            if binary_output:
                lpips_cur_ref = (lpips_cur_ref > binary_threshold).float()
                lpips_cur_ref = F.interpolate(lpips_cur_ref, size=latents_ref_img.shape[2:], mode='nearest')
            else:
                lpips_cur_ref = F.interpolate(lpips_cur_ref, size=latents_ref_img.shape[2:], mode='bilinear', align_corners=False)



            encoder_hidden_states = None 
            if 'text' in additional_input:
                text_folder = 'text'

                ref_text_path = ref_img_paths[0].replace('/rgb/', f'/{text_folder}/').replace('.jpg', '.json')
                ref_text_name = ref_img_paths[0].split('/')[-1].replace('.jpg', '')
                next_text_name = next_img_paths[0].split('/')[-1].replace('.jpg', '')

                if cur_img_paths[0] == 'white':
                    cur_text_path = ref_text_path.replace(ref_text_name, f'white_{ref_text_name}')
                else:
                    cur_text_path = cur_img_paths[0].replace('/rgb/', f'/{text_folder}/').replace('.jpg', '.json')

                cur_info = json.load(open(cur_text_path))


                next_prompt = cur_info['next_text']
            

                text_inputs = tokenize_prompt(
                    tokenizer, next_prompt, tokenizer_max_length=None
                )
                input_ids = text_inputs.input_ids
                attention_mask = text_inputs.attention_mask

                encoder_hidden_states = encode_prompt(
                    text_encoder,
                    input_ids,
                    attention_mask,
                )



            if use_PE:
                if PE_type == 'abs':
                    img_pos = cur_img_pos
                elif PE_type == 'rel':
                    img_pos = torch.cat([cur_img_pos, next_img_pos], dim=1)

                if PE_type == 'abs':
                    img_pos_emb = PE(img_pos)


                if encoder_hidden_states is not None:
                    encoder_hidden_states = torch.cat([img_pos_emb, encoder_hidden_states], dim=1)
                else:
                    encoder_hidden_states = img_pos_emb
    

            feat_input = torch.cat([latents_cur, latents_ref_img], dim=1)

            if "lpips_diff" in additional_input:
                feat_input = torch.cat([feat_input, lpips_cur_ref], dim=1)




            if 'noise' in additional_input:
                noise = torch.randn(bsz, 1, latents_ref_img.shape[2], latents_ref_img.shape[3]).to(local_rank)
                feat_input = torch.cat([feat_input, noise], dim=1)
                

                
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                lpips_out = RP(feat_input, encoder_hidden_states=encoder_hidden_states).sample

                if not binary_output:
                    lpips_out = F.sigmoid(lpips_out)
                    lpips_cur_gt = F.interpolate(lpips_cur_gt, size=lpips_out.shape[2:], mode='bilinear', align_corners=False)
                    
                    # mse loss
                    loss = F.mse_loss(lpips_out, lpips_cur_gt, reduction="mean")
                else:
                    lpips_cur_gt = (lpips_cur_gt > binary_threshold).float()
                    lpips_cur_gt = F.interpolate(lpips_cur_gt, size=lpips_out.shape[2:], mode='nearest')

                    # binary cross entropy loss with logits
                    loss = F.binary_cross_entropy_with_logits(lpips_out, lpips_cur_gt, reduction="mean")


                
                
            optimizer.zero_grad()

        

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()      

                
                """ >>> gradient clipping >>> """
                # torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            
  
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            # if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
            if is_main_process and global_step % checkpointing_steps == 0 :
                save_path = os.path.join(output_dir, f"checkpoints")
                if image_finetune:
                    state_dict = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "RP_state_dict": RP.module.state_dict(),
                    }

                    if use_PE:
                        state_dict["PE_state_dict"] = PE.module.state_dict()
              
                    # print(state_dict.keys())
                else:
                    state_dict = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "RP_state_dict": RP.module.state_dict(),
                    }

                    if use_PE:
                        state_dict["PE_state_dict"] = PE.module.state_dict()
                
                if step == len(train_dataloader) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-global_step-{global_step}.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
            # # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                RP.eval() 

                losses = []


                for step, batch in enumerate(val_dataloader):

                    if step > 30:
                        break
                     
                    ### >>>> Validation >>>> ###
                        
                    # Convert videos to latent space            
                    pixel_values = batch["pixel_values"].to(local_rank)
                    pixel_values_cur = batch["pixel_values_cur"].to(local_rank)
                    clip_ref_image = batch["clip_ref_image"].to(local_rank)
                    pixel_values_ref_img = batch["pixel_values_ref_img"].to(local_rank)
                    drop_image_embeds = batch["drop_image_embeds"].to(local_rank)

                    cur_img_pos = batch["cur_img_pos"].to(local_rank)
                    next_img_pos = batch["next_img_pos"].to(local_rank)

                    cur_img_paths = batch["cur_img_path"]
                    next_img_paths = batch["gt_img_path"]
                    ref_img_paths = batch["ref_img_path"]


                    # Sample noise that we'll add to the latents
                    bsz = pixel_values_ref_img.shape[0]
                    
                    # Get the text embedding for conditioning
                    with torch.no_grad():
                        img_pos = torch.cat([cur_img_pos, next_img_pos], dim=1)


                        assert bsz == 1


                        gt_next_img = Image.open(next_img_paths[0])
                        ref_img = Image.open(ref_img_paths[0])

                        if cur_img_paths[0] == 'white':
                            # white image PIL same shape as ref_img
                            cur_img = Image.new('RGB', ref_img.size, (255, 255, 255))
                        else:
                            cur_img = Image.open(cur_img_paths[0])
                            


                        cur_img = T.ToTensor()(cur_img).unsqueeze(0).to(local_rank)
                        gt_next_img = T.ToTensor()(gt_next_img).unsqueeze(0).to(local_rank)
                        ref_img = T.ToTensor()(ref_img).unsqueeze(0).to(local_rank)


                        # pad the border of them to make it multiple of 16, shape is [1, 3, H, W]
                        pad_size = [16 - cur_img.shape[2] % 16, 16 - cur_img.shape[3] % 16]

                        # if pad_size[0] != 16 or pad_size[1] != 16:
                        if pad_size[0] == 16:
                            pad_size[0] = 0
                        if pad_size[1] == 16:
                            pad_size[1] = 0
                        
                        cur_img = F.pad(cur_img, (0, pad_size[1], 0, pad_size[0]), value=1)
                        gt_next_img = F.pad(gt_next_img, (0, pad_size[1], 0, pad_size[0]), value=1)
                        ref_img = F.pad(ref_img, (0, pad_size[1], 0, pad_size[0]), value=1)

                        cur_img_norm = (cur_img - 0.5) / 0.5
                        gt_next_img_norm = (gt_next_img - 0.5) / 0.5
                        ref_img_norm = (ref_img - 0.5) / 0.5
                        
                        lpips_cur_gt = lpips_fn_alex(cur_img_norm, gt_next_img_norm)
                        lpips_cur_ref = lpips_fn_alex(cur_img_norm, ref_img_norm)

                        # clip to 0, 1
                        lpips_cur_gt = torch.clamp(lpips_cur_gt, 0, 1)
                        lpips_cur_ref = torch.clamp(lpips_cur_ref, 0, 1)



                        latents_ref_img = vae.encode(pixel_values_ref_img).latent_dist
                        latents_ref_img = latents_ref_img.sample()
                        latents_ref_img = latents_ref_img * 0.18215


                    
                        latents_cur = vae.encode(pixel_values_cur).latent_dist
                        latents_cur = latents_cur.sample()
                        latents_cur = latents_cur * 0.18215


                        if binary_output:
                            lpips_cur_ref = (lpips_cur_ref > binary_threshold).float()
                            lpips_cur_ref = F.interpolate(lpips_cur_ref, size=latents_ref_img.shape[2:], mode='nearest')
                        else:
                            lpips_cur_ref = F.interpolate(lpips_cur_ref, size=latents_ref_img.shape[2:], mode='bilinear', align_corners=False)



                        encoder_hidden_states = None 
                        if 'text' in additional_input:
                            ref_text_path = ref_img_paths[0].replace('rgb', f'/{text_folder}/').replace('.jpg', '.json')
                            ref_text_name = ref_img_paths[0].split('/')[-1].replace('.jpg', '')
                            next_text_name = next_img_paths[0].split('/')[-1].replace('.jpg', '')

                            if cur_img_paths[0] == 'white':
                                # print("cur_text_path:",cur_img_paths[0])
                                cur_text_path = ref_text_path.replace(ref_text_name, f'white_{ref_text_name}')
                                # print("cur_text_path:",cur_text_path)
                            else:
                                cur_text_path = cur_img_paths[0].replace('rgb', f'/{text_folder}/').replace('.jpg', '.json')

                            cur_info = json.load(open(cur_text_path))

                       
                            next_prompt = cur_info['next_text']
                            # print("next_text_corrected: ",next_prompt)

                            text_inputs = tokenize_prompt(
                                tokenizer, next_prompt, tokenizer_max_length=None
                            )
                            input_ids = text_inputs.input_ids
                            attention_mask = text_inputs.attention_mask

                            encoder_hidden_states = encode_prompt(
                                text_encoder,
                                input_ids,
                                attention_mask,
                            )

                            # print("encoder_hidden_states:",encoder_hidden_states.shape)





                            if use_PE:
                                if PE_type == 'abs':
                                    img_pos = cur_img_pos
                                elif PE_type == 'rel':
                                    img_pos = torch.cat([cur_img_pos, next_img_pos], dim=1)

                                if PE_type == 'abs':
                                    img_pos_emb = PE(img_pos)

    


                                encoder_hidden_states = torch.cat([img_pos_emb, encoder_hidden_states], dim=1)



                        feat_input = torch.cat([latents_cur, latents_ref_img], dim=1)

                        if "lpips_diff" in additional_input:
                            feat_input = torch.cat([feat_input, lpips_cur_ref], dim=1)






                        lpips_out = RP(feat_input, encoder_hidden_states=encoder_hidden_states).sample

                        lpips_out = F.sigmoid(lpips_out)

              
                        if not binary_output:
                            lpips_cur_gt = F.interpolate(lpips_cur_gt, size=lpips_out.shape[2:], mode='bilinear', align_corners=False)
                            
                            # mse loss
                            loss = F.mse_loss(lpips_out, lpips_cur_gt, reduction="mean")

                            lpips_out = F.interpolate(lpips_out, size=gt_next_img.shape[2:], mode='bilinear', align_corners=False)
                            lpips_cur_ref = F.interpolate(lpips_cur_ref, size=gt_next_img.shape[2:], mode='bilinear', align_corners=False)
                            lpips_cur_gt = F.interpolate(lpips_cur_gt, size=gt_next_img.shape[2:], mode='bilinear', align_corners=False)
                            
                        else:
                            lpips_cur_gt = (lpips_cur_gt > binary_threshold).float()
                            lpips_cur_gt = F.interpolate(lpips_cur_gt, size=lpips_out.shape[2:], mode='nearest')

                            # binary cross entropy loss
                            # print(lpips_out.shape, lpips_cur_gt.shape)
                            loss = F.binary_cross_entropy(lpips_out, lpips_cur_gt, reduction="mean")

                            lpips_out = (lpips_out > 0.5).float()
                            lpips_out = F.interpolate(lpips_out, size=gt_next_img.shape[2:], mode='nearest')
                            lpips_cur_ref = F.interpolate(lpips_cur_ref, size=gt_next_img.shape[2:], mode='nearest')
                            lpips_cur_gt = F.interpolate(lpips_cur_gt, size=gt_next_img.shape[2:], mode='nearest')

                        # visualization 
                        lpips_cur_ref_vis = torch.cat([lpips_cur_ref, lpips_cur_ref, lpips_cur_ref], dim=1)

                        vis_input = torch.cat([cur_img, ref_img, lpips_cur_ref_vis], dim=3)



                        if 'noise' in additional_input:
                            noise = torch.cat([noise, noise, noise], dim=1)
                            
                            noise = (noise - noise.min()) / (noise.max() - noise.min())
                            noise = F.interpolate(noise, size=gt_next_img.shape[2:], mode='bilinear', align_corners=False)

                            vis_input = torch.cat([vis_input, noise], dim=3)

                        lpips_out_vis = torch.cat([lpips_out, lpips_out, lpips_out], dim=1)
                        lpips_cur_gt_vis = torch.cat([lpips_cur_gt, lpips_cur_gt, lpips_cur_gt], dim=1)


   

                        vis_output = torch.cat([ lpips_out_vis, lpips_cur_gt_vis, cur_img, gt_next_img, ], dim=3)

                        vis_input = vis_input.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
                        vis_output = vis_output.permute(0, 2, 3, 1)[0].detach().cpu().numpy()



                        
                        assert vis_input.max() <= 1 and vis_input.min() >= 0
                        assert vis_output.max() <= 1 and vis_output.min() >= 0


                        # save using Image
                        vis_input = Image.fromarray((vis_input * 255).astype(np.uint8()))
                        vis_output = Image.fromarray((vis_output * 255).astype(np.uint8()))
                        if 'text' in additional_input:
                            # write text on the vis_output using PIL 
                            draw = ImageDraw.Draw(vis_output)
                            font = ImageFont.truetype("../utils/arial.ttf", 40)
                            draw.text((10, 10), next_prompt, (255, 0, 0), font=font)

                            

                        os.makedirs(f"{output_dir}/samples/{global_step}", exist_ok=True)

                        vis_input.save(f"{output_dir}/samples/{global_step}/{step}_input.jpg")
                        vis_output.save(f"{output_dir}/samples/{global_step}/{step}_output.jpg")


                        
                        losses.append(loss.item())

                avg_loss = sum(losses) / len(losses)
                RP.train()


                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_epoch = global_step

                    print('find best loss!!!')
                print(f"Validation loss: {avg_loss}, Best loss: {best_loss}, Best step: {best_epoch}")

      
                    

                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)


    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
    