import os
import sys 
sys.path.append('../')
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple
from utils.text_wrapper import *
import json
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from models.hack_unet2d import Hack_UNet2DConditionModel as UNet2DConditionModel

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from dataset.dataset import  collate_fn, InvPaintingDataset
from models.positional_encoder import PositionalEncoder

from utils.util import  zero_rank_print
from models.ReferenceEncoder import ReferenceEncoder
from models.hack_cur_image_guider import Hack_CurImageGuider as CurImageGuider

from models.ReferenceNet import ReferenceNet
from models.ReferenceNet_attention import ReferenceNetAttention
import torchvision.transforms as T

import pdb
import numpy as np
from PIL import Image



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
    launcher: str,    
    output_dir: str,
    pretrained_model_path: str,
    clip_model_path:str,
    description: str,
    fusion_blocks: str,
    cur_image_guider_checkpoint_path: str,
    referencenet_checkpoint_path: str,
    train_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    use_RP = False,
    RP_fusion_type =None, 
    feature_type: str = None,
    is_train_TP: bool = False,
    is_train_text_encoder: bool = False,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,

    learning_rate: float = 3e-5,
    use_PE: bool = False,
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
    use_TP: bool = False,
    TP_ckpt_path: str = "",
    PE_type: str = None,
    use_mask_for_loss=False, 
    use_binary_RP = False,
    RP_threshold = 0.5,
    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,
    no_refnet: bool = True,
    use_diff_clip: bool =  True,
    global_seed: int = 42,
    is_debug: bool = False,
    win_size: int = None,
    PE_time_max: int = None,
    PE_time_interval: int = None,
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


    text_folder_name = 'text'
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    pad_mode = train_data['pad_mode']
    
    folder_name = f'{pad_mode}_{folder_name}'


    if use_PE:
        folder_name = f"PE_{PE_type}_max{PE_time_max}_interval{PE_time_interval}-" + folder_name
    


    if no_refnet:
        folder_name = f"no_refnet-" + folder_name

    if use_diff_clip:
        folder_name = f"diff_clip-" + folder_name


    if use_RP:
        RP_name = f'RP_{RP_fusion_type}'
        if not use_mask_for_loss:
            RP_name = f'{RP_name}_no_mask'

        if use_binary_RP:
            
            RP_name = f'{RP_name}_binary{RP_threshold}'

        folder_name = f"{RP_name}-" + folder_name

        

    if use_TP:
        if not is_train_TP:
            folder_name = f"TP_{feature_type}-" + folder_name

        else:
            folder_name = f"TP_train_{feature_type}-" + folder_name




        if feature_type == 'text' and is_train_text_encoder:
            folder_name = f"train_txt_enc+" + folder_name



    
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



    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
        
        print(description)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    clip_image_encoder = ReferenceEncoder(model_path=clip_model_path)

    in_channels = 3 



    if use_RP and RP_fusion_type == "spatial":
        in_channels += 1


    cur_image_guider = CurImageGuider(in_channels=in_channels, noise_latent_channels=320)
    if not no_refnet:
        referencenet = ReferenceNet.from_pretrained(pretrained_model_path, subfolder="unet")

    if not image_finetune:
        cur_image_guider_state_dict = torch.load(cur_image_guider_checkpoint_path, map_location="cpu")
        cur_image_guider.load_state_dict(cur_image_guider_state_dict, strict=False)
        if not no_refnet:
            referencenet_state_dict = torch.load(referencenet_checkpoint_path, map_location="cpu")

            referencenet.load_state_dict(referencenet_state_dict, strict=False)

    
    if use_PE:
        if PE_type == "abs":
            PE = PositionalEncoder(21)
        elif PE_type == "rel":
            PE = PositionalEncoder(42)




    if use_diff_clip:
        from models.clip_adapter import NextImageFeaturePredictor
        image_adpater = NextImageFeaturePredictor()



    if use_TP:

        tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_path,
                subfolder="tokenizer",
                use_fast=False,
            )
        
        text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_path, revision=None)

        text_encoder = text_encoder_cls.from_pretrained(
            pretrained_model_path, subfolder="text_encoder"
        )

            


        
    


    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    if not no_refnet:
        reference_control_writer = ReferenceNetAttention(referencenet, do_classifier_free_guidance=False, mode='write', fusion_blocks=fusion_blocks, batch_size=train_batch_size ,is_image=image_finetune)
        reference_control_reader = ReferenceNetAttention(unet, do_classifier_free_guidance=False, mode='read', fusion_blocks=fusion_blocks, batch_size=train_batch_size ,is_image=image_finetune)
        
    
    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path: zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path

        m, u = unet.load_state_dict(state_dict, strict=True)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        del state_dict
        assert len(u) == 0
        
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    # text_encoder.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)
    
    # Set unet trainable parameters
    unet.requires_grad_(False)
    # unet.requires_grad_(True)
    for name, param in unet.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                # print(trainable_module_name)
                param.requires_grad = True
                break
    
    if use_diff_clip:
        image_adpater.requires_grad_(True)
        
    if image_finetune:
        cur_image_guider.requires_grad_(True)
        if not no_refnet:
            referencenet.requires_grad_(True)
        if use_PE:
            PE.requires_grad_(True)



        if use_TP:
            
            if feature_type == "text":
                if is_train_text_encoder:
                    text_encoder.requires_grad_(True)
                else:
                    text_encoder.requires_grad_(False)
            
            
    else:
        cur_image_guider.requires_grad_(False)
        if not no_refnet:
            referencenet.requires_grad_(False)    
                   
    
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if image_finetune:
        trainable_params += list(filter(lambda p: p.requires_grad, cur_image_guider.parameters())) 

        if not no_refnet:
            trainable_params += list(filter(lambda p: p.requires_grad, referencenet.parameters()))


        if use_diff_clip:
            trainable_params += list(filter(lambda p: p.requires_grad, image_adpater.parameters()))
                   
                
        if use_PE:
            trainable_params += list(filter(lambda p: p.requires_grad, PE.parameters()))


        if use_TP:
           
            if feature_type == "text":
                if is_train_text_encoder:
                    trainable_params += list(filter(lambda p: p.requires_grad, text_encoder.parameters()))


    
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

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            if not no_refnet:
                referencenet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if not no_refnet:
            referencenet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    # text_encoder.to(local_rank)
    clip_image_encoder.to(local_rank)
    cur_image_guider.to(local_rank)
    

    if not no_refnet:
        referencenet.to(local_rank)

    if use_diff_clip:
        image_adpater.to(local_rank)
    
    if use_PE:
        PE.to(local_rank)
    

    if use_TP:
            if feature_type == "text":
                text_encoder.to(local_rank)

    # Get the training dataset
    train_dataset = InvPaintingDataset(**train_data, is_image=image_finetune, PE_type=PE_type, PE_time_interval=PE_time_interval, PE_time_max=PE_time_max, win_size=win_size)

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

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )



    # DDP warpper
    unet.to(local_rank)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)
    
    if image_finetune:
        cur_image_guider = DDP(cur_image_guider, device_ids=[local_rank], output_device=local_rank)
        if not no_refnet:
            referencenet = DDP(referencenet, device_ids=[local_rank], output_device=local_rank)

        if use_diff_clip:
            image_adpater = DDP(image_adpater, device_ids=[local_rank], output_device=local_rank)

        

        if use_PE:
            PE = DDP(PE, device_ids=[local_rank], output_device=local_rank)


        if use_TP:
         
                if feature_type == "text" and is_train_text_encoder:
                    text_encoder = DDP(text_encoder, device_ids=[local_rank], output_device=local_rank)

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

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        cur_image_guider.train()
        if not no_refnet:
            referencenet.train()
        if use_diff_clip:
            image_adpater.train()
        if use_PE:
            PE.train()


        if use_TP:
   
                if feature_type == "text":
                    if is_train_text_encoder:
                        text_encoder.train()

            
        

        for step, batch in enumerate(train_dataloader): 
 
            
            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank)
            pixel_values_cur = batch["pixel_values_cur"].to(local_rank)
            clip_ref_image = batch["clip_ref_image"].to(local_rank)
            clip_cur_image = batch["clip_cur_image"].to(local_rank)
            pixel_values_ref_img = batch["pixel_values_ref_img"].to(local_rank)
            drop_image_embeds = batch["drop_image_embeds"].to(local_rank) # torch.Size([bs])
            drop_time_step = batch["drop_time_step"].to(local_rank) # torch.Size([bs])
            drop_cur_cond = batch["drop_cur_cond"].to(local_rank) # torch.Size([bs])
            drop_RP = batch["drop_RP"].to(local_rank) # torch.Size([bs])
            cur_img_pos = batch["cur_img_pos"].to(local_rank) # torch.Size([bs, 2])
            next_img_pos = batch["next_img_pos"].to(local_rank) # torch.Size([bs, 2])

            cur_img_paths = batch["cur_img_path"]
            next_img_paths = batch["gt_img_path"]
            ref_img_paths = batch["ref_img_path"]
                
            cur_img_name = cur_img_paths[0].split('/')[-1]
            next_img_name = next_img_paths[0].split('/')[-1]



            if PE_type == 'abs':
                img_pos = cur_img_pos
            elif PE_type == 'rel':
                img_pos = torch.cat([cur_img_pos, next_img_pos], dim=1)


            if use_TP:

                drop_feature = batch["drop_feature"].to(local_rank) # torch.Size([bs])
        
                assert len(cur_img_paths) == 1

                gt_next_img = Image.open(next_img_paths[0])
                ref_img = Image.open(ref_img_paths[0])

                if cur_img_paths[0] == 'white':
                    # white image PIL same shape as ref_img
                    cur_img = Image.new('RGB', ref_img.size, (255, 255, 255))

                else:
                    cur_img = Image.open(cur_img_paths[0])


            video_length = pixel_values.shape[1]

            
            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                latents = latents * 0.18215
                
                latents_ref_img = vae.encode(pixel_values_ref_img).latent_dist
                latents_ref_img = latents_ref_img.sample()
                latents_ref_img = latents_ref_img * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            


            mask = drop_cur_cond > 0
            pixel_values_cur[mask] = 0


            cur_region_img = None 
            if use_RP:
                if RP_fusion_type == "spatial":
                    ref_region_path = ref_img_paths[0].replace('/rgb/', f'/lpips/')
                    ref_region_name = ref_img_paths[0].split('/')[-1].replace('.jpg', '')
                    next_region_name = next_img_paths[0].split('/')[-1].replace('.jpg', '')

                    if cur_img_paths[0] == 'white':

                        cur_region_path = ref_region_path.replace(ref_region_name, f'white_{ref_region_name}')
                    else:
                        cur_region_path = cur_img_paths[0].replace('/rgb/', f'/lpips/')

                    
                    # read  gray
                    cur_region_img = Image.open(cur_region_path).convert('L')

                    # to tensor
                    cur_region_img = T.ToTensor()(cur_region_img)[None].to(local_rank)

                    pad_size = [16 - cur_region_img.shape[2] % 16, 16 - cur_region_img.shape[3] % 16]

                    # if pad_size[0] != 16 or pad_size[1] != 16:
                    if pad_size[0] == 16:
                        pad_size[0] = 0
                    if pad_size[1] == 16:
                        pad_size[1] = 0
                    

                    cur_region_img = F.pad(cur_region_img, (0, pad_size[1], 0, pad_size[0]), value=0)

                    cur_region_img = torch.clamp(cur_region_img, 0, 1)

                    if use_binary_RP:
                        cur_region_img = (cur_region_img > RP_threshold).float()

                    
                    if drop_RP:
                        cur_region_img = torch.zeros_like(cur_region_img)
                
                    pixel_values_cur = torch.cat([pixel_values_cur, cur_region_img], dim=1)



        
            latent_cur = cur_image_guider(pixel_values_cur)


            # Get the text embedding for conditioning
            with torch.no_grad():
                encoder_hidden_states_ref = clip_image_encoder(clip_ref_image).unsqueeze(1) # [bs,1,768]

                if use_diff_clip:
                    encoder_hidden_states_cur = clip_image_encoder(clip_cur_image).unsqueeze(1) # [bs,1,768]


                    encoder_hidden_states_next = image_adpater(encoder_hidden_states_cur, encoder_hidden_states_ref)

                else:
                    encoder_hidden_states_next = encoder_hidden_states_ref

            
            # support cfg train
            mask = drop_image_embeds > 0
            mask = mask.unsqueeze(1).unsqueeze(2).expand_as(encoder_hidden_states_next)
            encoder_hidden_states_next[mask] = 0


            if use_PE:
                if PE_type == 'abs':
                    img_pos_emb = PE(img_pos)

                if drop_time_step :
                    img_pos_emb = torch.zeros_like(img_pos_emb)
                encoder_hidden_states_next = torch.cat([encoder_hidden_states_next, img_pos_emb], dim=1)

            
            if use_TP:
                if feature_type =='text':
                    # Get the text embedding for conditioning
                    with torch.no_grad():

            
                        if feature_type == 'text':
                            ref_text_path = ref_img_paths[0].replace('/rgb/', f'/{text_folder_name}/').replace('.jpg', '.json')
                            ref_text_name = ref_img_paths[0].split('/')[-1].replace('.jpg', '')
                            next_text_name = next_img_paths[0].split('/')[-1].replace('.jpg', '')

                            if cur_img_paths[0] == 'white':
                                cur_text_path = ref_text_path.replace(ref_text_name, f'white_{ref_text_name}')
                            else:
                                cur_text_path = cur_img_paths[0].replace('/rgb/', f'/{text_folder_name}/').replace('.jpg', '.json')

                            cur_info = json.load(open(cur_text_path))


                    
                            next_prompt = cur_info['next_text']

                            # print(cur_text_path)

                            # print(next_text_name,cur_info['next_image_name'], ref_text_name, cur_info['ref_img_name'])
                            assert next_text_name == cur_info['next_image_name']
                            assert ref_text_name == cur_info['ref_img_name']


                            if drop_feature: 
                                next_prompt = ''
                            text_inputs = tokenize_prompt(
                                tokenizer, next_prompt, tokenizer_max_length=None
                            )
                            input_ids = text_inputs.input_ids
                            attention_mask = text_inputs.attention_mask

                            pred_next_img_feat = encode_prompt(
                                text_encoder,
                                input_ids,
                                attention_mask,
                            )

                        
                    encoder_hidden_states_next = torch.cat([encoder_hidden_states_next, pred_next_img_feat], dim=1)


            # pdb.set_trace()
            
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                ref_timesteps = torch.zeros_like(timesteps)
                
                if not no_refnet:
                    referencenet(latents_ref_img, ref_timesteps, encoder_hidden_states_ref)
                    reference_control_reader.update(reference_control_writer)
                    
                model_pred = unet(sample=noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states_next, latent_cur=latent_cur).sample
                
                if cur_region_img is None:

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:

                    if use_mask_for_loss: 
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        mask = F.interpolate(cur_region_img, size=(cur_region_img.shape[2] // 8, cur_region_img.shape[3] // 8), mode='nearest')
                        loss = loss * mask 

                        loss = loss.mean()

                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            




                if use_TP and is_train_TP:
                    print("TP_loss:",TP_loss, "loss:",loss)

                    loss = loss +  TP_loss
                    
                
                
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
            
            if not no_refnet:
                reference_control_reader.clear()
                reference_control_writer.clear()
            global_step += 1
            
            ### <<<< Training <<<< ###
            

            # Save checkpoint
            if is_main_process and global_step % checkpointing_steps == 0 :
                save_path = os.path.join(output_dir, f"checkpoints")
                if image_finetune:
                    state_dict = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "unet_state_dict": unet.module.state_dict(),
                        "cur_image_guider_state_dict": cur_image_guider.module.state_dict(),
                    
                    }
                    if use_PE:
                        state_dict["PE_state_dict"] = PE.module.state_dict()

                    if not no_refnet:
                        state_dict["referencenet_state_dict"] =  referencenet.module.state_dict()

                    if use_diff_clip:
                        state_dict["image_adpater_state_dict"] = image_adpater.module.state_dict()

     

                    if use_TP:
                        if feature_type == "text":
                            if is_train_text_encoder:
                                print("text_encoder_state_dict")
                                state_dict["text_encoder_state_dict"] = text_encoder.module.state_dict()

                else:
                    state_dict = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "unet_state_dict": unet.module.state_dict(),
                    }
                
      
                torch.save(state_dict, os.path.join(save_path, f"checkpoint-global_step-{global_step}.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
        
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)


    main(name=name, launcher=args.launcher, **config)
    
