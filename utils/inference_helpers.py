
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, DDIMScheduler
from pathlib import Path
import torch.nn.functional as F
from utils.text_wrapper import *
import torch.nn as nn
import cv2
import pdb
import glob
import os
import random
import numpy as np
from utils.text_wrapper import *

from PIL import Image
from omegaconf import OmegaConf
import torchvision.transforms as T

import torch
import json
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor
from models.hack_cur_image_guider import Hack_CurImageGuider as CurImageGuider
from models.ReferenceEncoder import ReferenceEncoder
from models.ReferenceNet import ReferenceNet
from pipelines.pipeline_stage_1 import InvPaintingPipeline
from models.hack_unet2d import Hack_UNet2DConditionModel as UNet2DConditionModel
import matplotlib.pyplot as plt
from models.positional_encoder import PositionalEncoder
        
    
def prepare_results_dir(config, ckpt_path, root_dst_dir):

    root_dst_folder = ckpt_path.split('/')[-1]
    root_dst_dir = f'{root_dst_dir}/{root_dst_folder}'
    os.makedirs(root_dst_dir, exist_ok=True)

    return root_dst_dir





def get_dataset_info(test_dir):

    source_images = glob.glob(f'{test_dir}/*.jpg', recursive=True) + glob.glob(f'{test_dir}/*.jpeg', recursive=True) + glob.glob(f'{test_dir}/*.png', recursive=True)
    source_images = sorted(source_images)
    images_info = {}
    for src_image in source_images:
        images_info[src_image] = []

    return images_info


    

def load_pipeline(config, pretrained_model_path, pretrained_clip_path, full_state_dict, dtype, device):

    ### >>> create pipeline >>> ###
    inference_config = OmegaConf.load('./configs/inference/inference.yaml')

    tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path,
            subfolder="tokenizer",
            use_fast=False,
        )
    
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_path, revision=None)

    text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_path, subfolder="text_encoder"
    )
    
    unet_config = UNet2DConditionModel.load_config(pretrained_model_path, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config)
    unet.load_state_dict(full_state_dict['unet_state_dict'], strict=True)

    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")

    clip_image_processor = CLIPProcessor.from_pretrained(pretrained_clip_path,local_files_only=True)
    clip_image_encoder = ReferenceEncoder(model_path=pretrained_clip_path)

    in_channels = 4
    cur_image_guider = CurImageGuider.from_pretrained(pretrained_model_path=full_state_dict['cur_image_guider_state_dict'], in_channels=in_channels)
    cur_image_guider.eval()

    from models.clip_adapter import NextImageFeaturePredictor
    image_adpater = NextImageFeaturePredictor()
    image_adpater.load_state_dict(full_state_dict['image_adpater_state_dict'])
    image_adpater.eval()
    image_adpater.to(dtype).to(device)


    
    referencenet = ReferenceNet.load_referencenet(pretrained_model_path=full_state_dict['referencenet_state_dict'])

    vae.to(dtype)
    unet.to(dtype)
    text_encoder.to(dtype)
    referencenet.to(dtype).to(device)
    cur_image_guider.to(dtype).to(device)
    clip_image_encoder.to(dtype).to(device)
    
    kwargs = {}
    pipeline = InvPaintingPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, 
        scheduler = EulerAncestralDiscreteScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),

        **kwargs
    )

    pipeline.to(device, dtype)


    kwargs = {
        'clip_image_processor': clip_image_processor,
        'clip_image_encoder': clip_image_encoder,
        'cur_image_guider': cur_image_guider,
        'referencenet': referencenet,
        'image_adpater': image_adpater,
    }

    return pipeline,  kwargs



class PE_wrapper():
    def __init__(self, config,full_state_dict, device, dtype):
        PE_type = config['PE_type']
        if PE_type == 'rel':
            channel = 42 
        elif PE_type == 'abs':
            channel = 21

        self.PE_type = PE_type
        self.config = config
        self.dtype = dtype
        self.device = device

        self.PE_time_interval = config['PE_time_interval']
        self.PE_time_max = config['PE_time_max']


        model = PositionalEncoder(channel)
        model.load_state_dict(full_state_dict['PE_state_dict'])
        model.to(dtype).to(device)
        model = model.eval()
        self.model = model




    def norm(self, pos):            
        pos = torch.tensor([pos], dtype=self.dtype, device=self.device)[None]
        pos = pos * 2 - 1
        return pos



    def embed(self, cur_pos):
        if self.PE_type == 'rel':
            assert False
            cur_pos = self.norm(cur_pos)
            next_pos = self.norm(next_pos)

            x = torch.cat([cur_pos, next_pos], dim=1)

        elif self.PE_type == 'abs':
            
            if  self.PE_time_interval > 1:
                # make the PE_sec to be cloestest multiple of PE_time_interval, +1 is to avoid 0 
                cur_pos = round(cur_pos / self.PE_time_interval) * self.PE_time_interval    
            cur_pos = cur_pos / self.PE_time_max

            # clamp to 0 to 1 
            cur_pos = max(0, min(1, cur_pos))
            cur_pos = self.norm(cur_pos)

            x = cur_pos
        with torch.no_grad():
            x = self.model(x)

        return x


def pad_to_16(source_image):
    if isinstance(source_image, np.ndarray):
        # resize the shortest edge to 512 but keep aspect ratio
        if source_image.shape[0] < source_image.shape[1]:
            source_image = cv2.resize(source_image, (int(512 * source_image.shape[1] / source_image.shape[0]), 512))
        else:
            source_image = cv2.resize(source_image, (512, int(512 * source_image.shape[0] / source_image.shape[1])))


        # pad four border to 16 multiple
        pad_size = [16 - source_image.shape[0] % 16, 16 - source_image.shape[1] % 16]

        # if pad_size[0] != 16 or pad_size[1] != 16:
        if pad_size[0] == 16:
            pad_size[0] = 0
        if pad_size[1] == 16:
            pad_size[1] = 0
        source_image = np.pad(source_image, ((0, pad_size[0]), (0, pad_size[1]), (0, 0)), mode='constant', constant_values=0)
    elif isinstance(source_image, torch.Tensor):
        # resize the shortest edge to 512 but keep aspect ratio,   N, 3, H, W
        # print("source_image.shape:",source_image.shape)
        if source_image.shape[3] < source_image.shape[2]:
            source_image = F.interpolate(source_image, (int(512 * source_image.shape[2] / source_image.shape[3]), 512), mode='bilinear')
        else:
            source_image = F.interpolate(source_image, (512, int(512 * source_image.shape[3] / source_image.shape[2])), mode='bilinear')

        # pad four border to 16 multiple
        pad_size = [16 - source_image.shape[2] % 16, 16 - source_image.shape[3] % 16]

        # if pad_size[0] != 16 or pad_size[1] != 16:
        if pad_size[0] == 16:
            pad_size[0] = 0
        if pad_size[1] == 16:
            pad_size[1] = 0

        source_image = F.pad(source_image, (0, pad_size[1], 0, pad_size[0]), value=0)

        
    return source_image



class TP_wrapper(nn.Module):
    def __init__(self, config, full_state_dict, device, dtype):
        super(TP_wrapper, self).__init__()

        model = TP_text_wrapper(config, full_state_dict, device, dtype)


        self.model = model 
        print('TP loaded')


    def forward(self, cur_img_path, ref_img_path, cache_path=None):
        return self.model.forward(cur_img_path, ref_img_path, cache_path)

    def get_negative_embeddings(self):
        return self.model.get_negative_embeddings()

    def get_negative_embeddings_exclude(self, next_text):
        return self.model.get_negative_embeddings_exclude(next_text)
    
    def encode_text_prompt(self, next_prompt):
        return self.model.encode_text_prompt(next_prompt)

class TP_text_wrapper(nn.Module):
    def __init__(self, config,full_state_dict, device, dtype):
        super(TP_text_wrapper, self).__init__()
        pretrained_model_path = config['pretrained_model_path']
        tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_path,
                subfolder="tokenizer",
                use_fast=False,
            )

            
        if config['llava_path'] is not None: 
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            from utils.llava_utils import Predictor
            prompt = "There are two images side by side. The left image is an intermediate stage in a painting process of the right image. Please tell me what content should be painted next? The answer should be less than 2 words."
            model_path = config['llava_path']
            args = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": prompt,
                "conv_mode": None,
                "image_file": None,
                "sep": ",",
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512,
                # 'num_return_sequences': 10, 
            })()

            self.predictor = Predictor(args)
            self.args = args


        if config['is_train_text_encoder']:
            text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_path, revision=None)
            text_encoder = text_encoder_cls.from_pretrained(
                pretrained_model_path, subfolder="text_encoder"
            ).to(device).to(dtype)
            
            text_encoder.load_state_dict(full_state_dict['text_encoder_state_dict'], strict=True)

        else:
            text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_path, revision=None)
            text_encoder = text_encoder_cls.from_pretrained(
                pretrained_model_path, subfolder="text_encoder"
            ).to(device).to(dtype)


        self.text_encoder = text_encoder
        self.tokenizer = tokenizer


    def forward(self, cur_img_path, ref_img_path, cache_path=None):

        cur_img = Image.open(cur_img_path)
        ref_img = Image.open(ref_img_path)
        canvas = Image.new('RGB', (ref_img.width*2, ref_img.height))
        canvas.paste(cur_img, (0, 0))
        canvas.paste(ref_img, (ref_img.width, 0))
        canvas.save(cache_path)


        self.args.image_file = cache_path 
        self.predictor.set_args(self.args)
        next_prompt = self.predictor.eval_model()
        print("next_prompt:",next_prompt)


        text_inputs = tokenize_prompt(
            self.tokenizer, next_prompt, tokenizer_max_length=None
        )
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        text_embeddings = encode_prompt(
            self.text_encoder,
            input_ids,
            attention_mask,
        )

        return text_embeddings, next_prompt


    def encode_text_prompt(self, next_prompt):


        text_inputs = tokenize_prompt(
            self.tokenizer, next_prompt, tokenizer_max_length=None
        )
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        text_embeddings = encode_prompt(
            self.text_encoder,
            input_ids,
            attention_mask,
        )

        return text_embeddings




    def get_negative_embeddings(self):


        text_inputs = tokenize_prompt(
            self.tokenizer, '', tokenizer_max_length=None
        )
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        text_embeddings = encode_prompt(
            self.text_encoder,
            input_ids,
            attention_mask,
        )


        return text_embeddings






class RP_wrapper(nn.Module):
    def __init__(self, config, full_state_dict, device, dtype):
        super(RP_wrapper, self).__init__()

        self.config = config
        dtype = torch.float16
        self.dtype = dtype
        self.device = device

        self.use_PE = config['use_PE']

        RP_path = config['RP_path']
        
        import lpips 
        lpips_fn_alex = lpips.LPIPS(net='alex', spatial=True) # best forward scores
    
        from unet_2d.unet_2d_condition import UNet2DConditionModel
        pretrained_model_path = "./base_ckpt/realisticVisionV51_v51VAE"
        config = UNet2DConditionModel.load_config(pretrained_model_path + '/unet')

        
        config["in_channels"] = 4 + 4 + 1 

        if 'noise' in RP_path:
            config["in_channels"] += 1

        if not self.config['use_TP']:
            config["cross_attention_dim"] = None


        config["out_channels"] = 1


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

        unet = UNet2DConditionModel.from_config(config)

        # load 
        state_dict = torch.load(RP_path, map_location='cpu')
        unet.load_state_dict(state_dict['RP_state_dict'], strict=True)
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")


        if self.use_PE:
            PE_model = PE_wrapper(self.config, state_dict, device, dtype)
            self.PE_model = PE_model

        # to cuda 
        self.unet = unet.to(dtype).to(device)
        self.vae = vae.to(dtype).to(device)
        self.text_encoder = text_encoder.to(dtype).to(device)
        self.lpips_fn_alex = lpips_fn_alex.to(device).to(torch.float32)
        self.tokenizer = tokenizer

        
        self.unet.eval()
        self.vae.eval()

        print('Trained RP loaded')


           
        self.binary_threshold = self.config['binary_threshold']



    def read_RP_mask(self, mask_path):
        mask = Image.open(mask_path).convert('L')
        mask = T.ToTensor()(mask)[None]
        mask = pad_to_16(mask)
        mask = (mask > 0.5).to(self.dtype).to(self.device)
        return mask


    def forward(self, cur_img_path, ref_img_path, next_prompt=None, next_RP_embeddings_prev=None, PE_sec=None, generator=None, threshold=None ):
        
            # print(cur_img_path, ref_img_path)
            ref_img = Image.open(ref_img_path).convert('RGB')

            if cur_img_path == 'white':
                cur_img = Image.new('RGB', (ref_img.width, ref_img.height), (255, 255, 255))
            else:
                cur_img = Image.open(cur_img_path).convert('RGB')

            cur_img = T.ToTensor()(cur_img)[None]
            ref_img = T.ToTensor()(ref_img)[None]

            cur_img = pad_to_16(cur_img).to(self.dtype).to(self.device)
            ref_img = pad_to_16(ref_img).to(self.dtype).to(self.device)


            cur_img = cur_img * 2 - 1
            ref_img = ref_img * 2 - 1


            

            with torch.no_grad():

                latents_cur_img = self.vae.encode(cur_img).latent_dist
                latents_cur_img = latents_cur_img.sample()
                latents_cur_img = latents_cur_img * 0.18215
                latents_cur_img = latents_cur_img.to(self.dtype).to(self.device)

                latents_ref_img = self.vae.encode(ref_img).latent_dist
                latents_ref_img = latents_ref_img.sample()
                latents_ref_img = latents_ref_img * 0.18215
                latents_ref_img = latents_ref_img.to(self.dtype).to(self.device)

                lpips_cur_ref = self.lpips_fn_alex(cur_img.to(torch.float32), ref_img.to(torch.float32)).to(self.dtype).to(self.device)
                

                lpips_cur_ref =  (lpips_cur_ref > self.binary_threshold).to(self.dtype).to(self.device)

                lpips_cur_ref = F.interpolate(lpips_cur_ref, size=(latents_ref_img.shape[2], latents_ref_img.shape[3]), mode='nearest')

                if next_RP_embeddings_prev is not None:
                    next_RP_embeddings_prev = F.interpolate(next_RP_embeddings_prev, size=(latents_ref_img.shape[2], latents_ref_img.shape[3]), mode='nearest').to(self.dtype).to(self.device)
                
                    lpips_cur_ref = lpips_cur_ref * (1-next_RP_embeddings_prev)
              
                

                lpips_cur_ref = lpips_cur_ref.to(self.dtype).to(self.device)
        

                feat_input = torch.cat([latents_cur_img, latents_ref_img, lpips_cur_ref], dim=1)

                if 'noise' in self.config['RP_path']:
                    noise = torch.randn(feat_input.shape[0], 1, feat_input.shape[2], feat_input.shape[3], device=self.device, dtype=self.dtype, generator=generator)
                    feat_input = torch.cat([feat_input, noise * 5], dim=1)
                    print('noise added', noise.mean())

                if self.config['use_TP']:
                    text_inputs = tokenize_prompt(
                        self.tokenizer, next_prompt, tokenizer_max_length=None
                    )
                    input_ids = text_inputs.input_ids
                    attention_mask = text_inputs.attention_mask

                    encoder_hidden_states = encode_prompt(
                        self.text_encoder,
                        input_ids,
                        attention_mask,
                    )
                    encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(self.device)

                else:
                    encoder_hidden_states = None

                if self.use_PE:
                    pos_embed = self.PE_model.embed(PE_sec)
                    pos_embed = pos_embed.to(self.dtype).to(self.device)

                    if encoder_hidden_states is not None:

                        encoder_hidden_states = torch.cat([pos_embed, encoder_hidden_states], dim=1)
                    else:
                        encoder_hidden_states = pos_embed

                    


                lpips_out = self.unet(feat_input, encoder_hidden_states=encoder_hidden_states).sample

                lpips_out = F.sigmoid(lpips_out)
    

                lpips_out = (lpips_out > threshold).to(self.dtype).to(self.device)
                lpips_out = F.interpolate(lpips_out, size=ref_img.shape[2:], mode='nearest')
             

                return lpips_out, F.interpolate(lpips_cur_ref, size=ref_img.shape[2:], mode='nearest')



