import argparse
import datetime
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.distributed as dist
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F
from utils.dist_tools import distributed_init
from utils.inference_helpers import *
import lpips

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with specified configuration.")
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/renderer/ckpt/checkpoint-global_step-200000.ckpt', help="Path to renderer checkpoint.")
    parser.add_argument("--RP_path", type=str, default='./checkpoints/RP/checkpoint-global_step-80000.ckpt', help="Path to LLaVA model checkpoint.")
    parser.add_argument("--output_dir", type=str, default='./results', help="Path to the output directory.")
    parser.add_argument("--llava_path", type=str, default='checkpoints/TP_llava', help="Path to LLaVA model checkpoint.")
    parser.add_argument("--test_dir", type=str, default='./data/demo', help="Path to the directory containing test images.")
    parser.add_argument("--random_seeds", type=int, nargs='+', default=[1], help="List of random seeds for inference.")
    parser.add_argument("--num_actual_inference_steps", type=int, default=50, help="Number of actual inference steps.")
    parser.add_argument("--steps", type=int, default=25, help="Number of steps.")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="Guidance scale for inference.")
    parser.add_argument("--TP_guidance_scale", type=float, default=5.0, help="TP guidance scale.")
    parser.add_argument("--cur_guidance_scale", type=float, default=1.0, help="Current image guidance scale.")
    parser.add_argument("--RP_guidance_scale", type=float, default=5.0, help="RP guidance scale.")
    parser.add_argument("--PE_guidance_scale", type=float, default=5.0, help="PE guidance scale.")
    parser.add_argument("--dilate_RP", type=bool, default=True, help="Dilate RP or not.")
    parser.add_argument("--PE_sec", type=int, default=20, help="PE section.")
    parser.add_argument("--total_step", type=int, default=50, help="Total steps.")
    parser.add_argument("--binary_threshold", type=float, default=0.2, help="Binary threshold.")
    parser.add_argument("--combine_init", type=bool, default=True, help="Combine initial image.")
    parser.add_argument("--combine_init_ratio", type=float, default=0.2, help="Ratio to combine initial image.")
    parser.add_argument("--split", type=str, default='test', help="Data split.")
    parser.add_argument("--cur_alpha", type=float, default=0.0, help="Current alpha value.")
    parser.add_argument("--pretrained_model_path", type=str, default="base_ckpt/realisticVisionV51_v51VAE", help="Path to pretrained model.")
    parser.add_argument("--pretrained_clip_path", type=str, default="./base_ckpt/clip-vit-base-patch32", help="Path to pretrained CLIP model.")
    parser.add_argument("--tmp_cur_img_folder", type=str, default='cache_cur_img', help="Temporary image folder.")
    parser.add_argument("--dist", action="store_true", required=False, help="Enable distributed mode.")
    parser.add_argument("--rank", type=int, default=0, required=False, help="Rank for distributed mode.")
    parser.add_argument("--world_size", type=int, default=1, required=False, help="World size for distributed mode.")
    return parser.parse_args()

def main(args):



    # Load configurations and initialize device
    device = torch.device(f"cuda:{args.rank}")
    dist_kwargs = {"rank": args.rank, "world_size": args.world_size, "dist": args.dist}
    dtype = torch.float16
    config_path = os.path.join(os.path.dirname(args.ckpt_path), '..', 'config.yaml')
    config = OmegaConf.load(config_path)

    # Update config with arguments
    config.update({
        'pretrained_model_path': args.pretrained_model_path,
        'split': args.split,
        'llava_path': args.llava_path,
        'binary': args.binary_threshold > 0,
        'binary_threshold': args.binary_threshold,
        'PE_sec': args.PE_sec,
        'RP_path': args.RP_path,
    })


    # Set up output directory and data paths
    root_dst_dir = prepare_results_dir(config, args.ckpt_path, args.output_dir)
    images_info = get_dataset_info(args.test_dir)
    full_state_dict = torch.load(args.ckpt_path, map_location='cpu')
    
    # get time 
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    rand_num = random.randint(0, 100000)


    total_step, guidance_scale, cur_guidance_scale, cur_alpha, PE_guidance_scale = args.total_step, args.guidance_scale, args.cur_guidance_scale, args.cur_alpha, args.PE_guidance_scale
    TP_guidance_scale, RP_guidance_scale, dilate_RP, combine_init, combine_init_ratio = args.TP_guidance_scale, args.RP_guidance_scale, args.dilate_RP, args.combine_init, args.combine_init_ratio
    steps, num_actual_inference_steps = args.steps, args.num_actual_inference_steps
    
    # this is used for saving images for text generator
    tmp_cur_img_path = f'{args.tmp_cur_img_folder}/{time_str}_{rand_num}.png'
    os.makedirs(args.tmp_cur_img_folder, exist_ok=True)
    next_RP_embeddings = None
    next_prompt  = None

    # prepare text generator and mask generator
    TP = TP_wrapper(config, full_state_dict, device, dtype)
    RP = RP_wrapper(config, full_state_dict, device, dtype)
        
    # prepare time embeddings
    PE_sec = config['PE_sec']
    PE = PE_wrapper(config, full_state_dict, device, dtype)
    with torch.no_grad():
        PE_embeddings = PE.embed(PE_sec)
            
    # prepare negative text embeddings
    negative_next_TP_embeddings = TP.get_negative_embeddings()

    # Load inference pipeline and LPIPS for similarity calculations
    pipeline, pipeline_kwargs = load_pipeline(config, args.pretrained_model_path, args.pretrained_clip_path, full_state_dict, dtype, device)
    lpips_fn_alex = lpips.LPIPS(net='alex', spatial=False).to(device)

    print('Start inference')
    for random_seed in args.random_seeds:

        for ref_img_path in images_info:
            seed = 1
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            
            gt_next_img_paths = images_info[ref_img_path]
            

            dst_dir = f'{root_dst_dir}/' + ref_img_path.split('/')[-2] + '/' + ref_img_path.split('/')[-1].split('.')[0] + f'/seed_{random_seed}_total_step_{total_step}'
            guidance_name = f'gs+clip{guidance_scale}+cur_alpha{cur_alpha}'

            guidance_name += f'+PE{PE_guidance_scale}'

            guidance_name += f'+TP{TP_guidance_scale}'

            guidance_name += f'+RP{RP_guidance_scale}'
            if combine_init:
                guidance_name += f'_combine{combine_init_ratio}'
                if config['binary']:
                    guidance_name += f'_binary'
                if dilate_RP:
                    guidance_name += f'_dilate'


            dst_dir += f'_{guidance_name}'
            
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
            else:
                print(f'{dst_dir} exists, skip')
                continue

            # target image
            ref_img = np.array(Image.open(ref_img_path).convert('RGB'))
            ori_h, ori_w, c = ref_img.shape
            
            plt.imsave(f"{dst_dir}/ori_img.jpg", ref_img)
            ref_img = pad_to_16(ref_img)

            # current image: starting from white canvas
            cur_img = np.ones((ori_h, ori_w, 3)) * 255
            cur_img = pad_to_16(cur_img)
            cur_img = cur_img.astype(np.uint8)
            plt.imsave(f"{dst_dir}/sample_0.jpg", cur_img[:ori_h, :ori_w])

            generator = torch.Generator(device=torch.device("cuda:0"))
            generator.manual_seed(random_seed)
            state = generator.get_state()

            next_RP_embeddings_prev = None

            cur_next_diffs = []
            next_ref_diffs = []        
            for idx in tqdm(range( total_step)):
                generator.set_state(state)
        
                generator_RP = torch.Generator(device=torch.device("cuda:0"))
                generator_RP.manual_seed(random_seed + idx)


                H, W, C = ref_img.shape

                # determine whether to stop, based on the last two differences
                if len(cur_next_diffs) > 3 and cur_next_diffs[-2] < 1e-3 and cur_next_diffs[-1] < 1e-3:
                    break

                # determine whether to stop, based on difference between next and reference
                if len(next_ref_diffs) > 0 and next_ref_diffs[-1] < 1e-1:
                    break

                kwargs = {}
                # copy dist_kwargs
                kwargs.update(dist_kwargs)  
                kwargs.update(pipeline_kwargs)              
                
                kwargs['use_PE'] = config['use_PE']
                kwargs['PE_guidance_scale'] = PE_guidance_scale
                kwargs['PE_embeddings'] = PE_embeddings
                kwargs['negative_PE_embeddings'] = torch.zeros_like(PE_embeddings)
            
                        
                cur_img_path = tmp_cur_img_path 
                plt.imsave(cur_img_path, cur_img)
                
                cache_path = cur_img_path.replace('.png', '_.png')
                next_text_embeddings, next_prompt = TP(cur_img_path, ref_img_path, cache_path=cache_path)
            
                
                kwargs['TP_feature'] = next_text_embeddings
                kwargs['use_TP'] = config['use_TP']
                kwargs['TP_guidance_scale'] = TP_guidance_scale
                kwargs['negative_TP_feature'] = negative_next_TP_embeddings


                if idx == 0:
                    cur_img_path = 'white'

                cur_img_path = tmp_cur_img_path 
                plt.imsave(cur_img_path, cur_img)


                ##### for mask generation #####

                # the predicted mask in the previous step 
                if next_RP_embeddings is not None:
                    next_RP_embeddings_prev = next_RP_embeddings.clone().to(torch.float32)

                
                if dilate_RP:
                    # This is a trick to make the RP more robust, ensuring the generated mask is not too small. 
                    # We don't use 0.5 as the threshold, but try different thresholds
                
                    threshold_list = [ 0.5, 0.4, 0.3, 0.2, 0.1]
                    for threshold in threshold_list:
                        next_RP_embeddings, input_RP_embeddings_diff = RP(cur_img_path, ref_img_path, next_prompt=next_prompt, next_RP_embeddings_prev=None, PE_sec=PE_sec, generator=generator_RP, threshold=threshold)

                        if idx == 0:
                            break

                        next_RP_embeddings_sum = next_RP_embeddings.sum()

                        if next_RP_embeddings_sum < int(H * W * 0.05):
                            print(f'Warning: next_RP_embeddings is too small: {next_RP_embeddings_sum}, change to {threshold}')
                            continue 
                        
                        # compute iou 
                        iou = (next_RP_embeddings * next_RP_embeddings_prev).sum() / ((next_RP_embeddings + next_RP_embeddings_prev) > 0).sum()
                        if iou < 0.8:
                            break
                        else:
                            sum_diff = next_RP_embeddings.float().sum() - next_RP_embeddings_prev.float().sum()
                            print(f'Warning: iou {iou} is too high, sum_diff {sum_diff}, change to {threshold}')

                else:
                    next_RP_embeddings, input_RP_embeddings_diff = RP(cur_img_path, ref_img_path, next_prompt=next_prompt, next_RP_embeddings_prev=None, PE_sec=PE_sec, generator=generator_RP, threshold=0.5)
                    

                kwargs['RP_guidance_scale'] = RP_guidance_scale
                kwargs['RP_embeddings'] = next_RP_embeddings.to(dtype)
                kwargs['negative_RP_embeddings'] = torch.zeros_like(next_RP_embeddings)

                if combine_init:
                    if idx > 0:
                        kwargs['combine_init'] = combine_init
                        kwargs['combine_init_ratio'] = combine_init_ratio
                        kwargs['img_init_latents'] = pred_next_latents


    
      
                generator = generator.set_state(state)
                outputs = pipeline(
                    num_inference_steps     = steps,
                    guidance_scale          = guidance_scale,
                    cur_guidance_scale      = cur_guidance_scale, 
                    width                   = W,
                    height                  = H,
                    generator               = generator,
                    num_actual_inference_steps = num_actual_inference_steps,
                    source_image            = ref_img,
                    cur_condition           = cur_img,
                    cur_alpha               = cur_alpha,
                    **kwargs,
                )


                pred_next_img = outputs.images
                pred_next_latents = outputs.latents


                
                # save sample torch tensor (1, H, W, 3)
                pred_next_img = pred_next_img[0]
                pred_next_img = pred_next_img.cpu().numpy()
                pred_next_img = np.clip(pred_next_img * 255, 0, 255).astype(np.uint8)

        
                cur_img_tensor = torch.tensor(cur_img).permute(2, 0, 1).unsqueeze(0).to(dtype).to(device)[:ori_h, :ori_w, :]
                pred_next_img_tensor = torch.tensor(pred_next_img).permute(2, 0, 1).unsqueeze(0).to(dtype).to(device)[:ori_h, :ori_w, :]
                ref_img_tensor = torch.tensor(ref_img).permute(2, 0, 1).unsqueeze(0).to(dtype).to(device)[:ori_h, :ori_w, :]

                cur_img_tensor = (cur_img_tensor / 255.)  * 2 - 1
                pred_next_img_tensor = (pred_next_img_tensor / 255.)  * 2 - 1
                ref_img_tensor = (ref_img_tensor / 255.)  * 2 - 1

                # difference between current and next, next and reference
                cur_next_diff = lpips_fn_alex(cur_img_tensor.cuda(), pred_next_img_tensor.cuda()).item()
                next_ref_diff = lpips_fn_alex(ref_img_tensor.cuda(), pred_next_img_tensor.cuda()).item()

                cur_next_diffs.append(cur_next_diff)
                next_ref_diffs.append(next_ref_diff)

                # Visualization
                next_RP_embeddings_vis = next_RP_embeddings.cpu().detach().numpy()
                next_RP_embeddings_vis = np.clip(next_RP_embeddings_vis * 255, 0, 255).astype(np.uint8)
                next_RP_embeddings_vis = next_RP_embeddings_vis[0,0]
                next_RP_embeddings_vis = next_RP_embeddings_vis[..., None]
                next_RP_embeddings_vis = np.concatenate([next_RP_embeddings_vis, next_RP_embeddings_vis, next_RP_embeddings_vis], axis=2)
                next_RP_embeddings_vis = next_RP_embeddings_vis[:ori_h, :ori_w, :]


                next_RP_embeddings_vis = Image.fromarray(next_RP_embeddings_vis)
                draw = ImageDraw.Draw(next_RP_embeddings_vis)
                font = ImageFont.truetype("utils/arial.ttf", 40)
                draw.text((10, 10), next_prompt, (255, 0, 0), font=font)
                next_RP_embeddings_vis = np.array(next_RP_embeddings_vis)

                next_RP_embeddings_vis = np.concatenate([pred_next_img[:ori_h, :ori_w], next_RP_embeddings_vis], axis=1)
                plt.imsave(f"{dst_dir}/vis_sample_{idx+1}.jpg", next_RP_embeddings_vis)
                plt.imsave(f"{dst_dir}/sample_{idx+1}.jpg", pred_next_img[:ori_h, :ori_w, :])



                cur_img = pred_next_img

                # post processing for cur_img, pad to multiple of 16
                cur_img = cur_img[:ori_h, :ori_w, :]
                cur_img = pad_to_16(cur_img)
                cur_img = cur_img.astype(np.uint8)

                assert cur_img.shape[0] == ref_img.shape[0] and cur_img.shape[1] == ref_img.shape[1]



if __name__ == "__main__":

    args = parse_args()
    main(args)
