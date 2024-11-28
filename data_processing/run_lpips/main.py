import glob
import os
import json
import numpy as np
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
import lpips
import torch
import cv2
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Compute LPIPS and visualize.")
parser.add_argument("--datadata_folder", type=str, default="../../data/sample_data",
                    help="Path to the data folder containing videos.")
parser.add_argument("--split", type=str, default="train",
                    help="Dataset split to process (e.g., train, val, test).")
parser.add_argument("--sample_num", type=int, default=1000,
                    help="Number of frames to sample.")
parser.add_argument("--save_vis", action="store_true", help="Whether to save visualization outputs.")


args = parser.parse_args()

# Initialize LPIPS
lpips_fn_alex = lpips.LPIPS(net='alex', spatial=True).cuda()

datadata_folder = args.datadata_folder
split = args.split
sample_num = args.sample_num
save_vis = args.save_vis

cache_dir = f'cache'
os.makedirs(cache_dir, exist_ok=True)

video_dirs = glob.glob(f'{datadata_folder}/{split}/rgb/*')
video_dirs = sorted(video_dirs)
print(video_dirs)

dst_dir = f'{datadata_folder}/{split}/lpips'
dst_vis_dir = f'{datadata_folder}/{split}/lpips_vis'

for video_dir in tqdm.tqdm(video_dirs):
    last_aligned_frame_inv_path = f'{video_dir}/last_aligned_frame_inv.json'
    video_name = video_dir.split('/')[-1]

    # Load JSON
    with open(last_aligned_frame_inv_path) as f:
        last_aligned_frame_inv_path_dict = json.load(f)

    for ref_img_name in list(last_aligned_frame_inv_path_dict.keys()):
        canvas_candidate_list_full = last_aligned_frame_inv_path_dict[ref_img_name]
        ref_image = Image.open(f'{video_dir}/{ref_img_name}.jpg')
        canvas_candidate_list_full = canvas_candidate_list_full[1:]

        canvas_candidate_list = ['white']
        if len(canvas_candidate_list_full) >= (sample_num - 1):
            sample_inds = np.round(np.linspace(0, len(canvas_candidate_list_full) - 1, sample_num - 1)).astype(int)
            canvas_candidate_list = canvas_candidate_list + [canvas_candidate_list_full[i] for i in sample_inds]
            canvas_candidate_list.append(ref_img_name)
            assert len(canvas_candidate_list) == (sample_num + 1)
        else:
            canvas_candidate_list = canvas_candidate_list + canvas_candidate_list_full
            canvas_candidate_list.append(ref_img_name)

        for i in range(len(canvas_candidate_list[:-1])):
            cur_image_name = canvas_candidate_list[i]
            next_image_name = canvas_candidate_list[i + 1]

            if cur_image_name == 'white':
                cur_image = Image.new('RGB', (ref_image.width, ref_image.height), (255, 255, 255))
                cur_image_name = f'white_{ref_img_name}'
            else:
                cur_image_path = f'{video_dir}/{cur_image_name}.jpg'
                cur_image = Image.open(cur_image_path)

            next_image_path = f'{video_dir}/{next_image_name}.jpg'
            next_image = Image.open(next_image_path)

            # Convert to torch tensors
            cur_image = torch.tensor(np.array(cur_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            next_image = torch.tensor(np.array(next_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # Normalize to [-1, 1]
            cur_image = cur_image * 2 - 1
            next_image = next_image * 2 - 1

            # Move to GPU
            cur_image = cur_image.cuda()
            next_image = next_image.cuda()

            # Compute LPIPS
            lpips_out = lpips_fn_alex(cur_image, next_image)
            lpips_out = torch.clamp(lpips_out, 0, 1)
            lpips_mask = lpips_out.cpu().detach().numpy()

            # Save LPIPS mask
            dst_path = f'{dst_dir}/{video_name}'
            os.makedirs(dst_path, exist_ok=True)
            cv2.imwrite(f'{dst_path}/{cur_image_name}.jpg', (lpips_mask[0] * 255).astype(np.uint8).transpose(1, 2, 0))

            if save_vis:
                # Visualize LPIPS mask
                cur_image = (cur_image + 1) / 2
                next_image = (next_image + 1) / 2

                lpips_mask = lpips_mask[0]
                lpips_mask = (lpips_mask * 255).astype(np.uint8).transpose(1, 2, 0)
                next_image = next_image[0].cpu().detach().numpy()
                next_image = (next_image * 255).astype(np.uint8).transpose(1, 2, 0)
                cur_image = cur_image[0].cpu().detach().numpy()
                cur_image = (cur_image * 255).astype(np.uint8).transpose(1, 2, 0)

                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(cur_image)
                ax1.axis('off')
                ax2.imshow(next_image)
                ax2.imshow(lpips_mask[:, :, 0], cmap='hot', alpha=0.5, interpolation='bilinear')
                ax2.axis('off')

                os.makedirs(f'{dst_vis_dir}/{video_name}', exist_ok=True)
                plt.savefig(f'{dst_vis_dir}/{video_name}/{cur_image_name}.jpg', bbox_inches='tight')
                plt.close()
