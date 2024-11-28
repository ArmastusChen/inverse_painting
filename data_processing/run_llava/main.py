import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from utils import Predictor

import cv2
import glob
import os
import json
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import tqdm
import matplotlib.pyplot as plt


def main(args):
    datadata_folder = args.data_folder
    split = args.split
    model_path = args.model_path
    sample_num = args.sample_num
    save_vis = args.save_vis
    prompt = args.prompt

    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)

    video_dirs = glob.glob(f'{datadata_folder}/{split}/rgb/*')
    video_dirs = sorted(video_dirs)

    dst_dir = f'{datadata_folder}/{split}/text'
    dst_vis_dir = f'{datadata_folder}/{split}/text_vis'

    args_obj = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": None,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    predictor = Predictor(args_obj)

    cache_path = f'{cache_dir}/cache.png'

    for video_dir in tqdm.tqdm(video_dirs[:]):
        last_aligned_frame_inv_path = f'{video_dir}/last_aligned_frame_inv.json'
        video_name = os.path.basename(video_dir)

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
                canvas_candidate_list += [canvas_candidate_list_full[i] for i in sample_inds]
                canvas_candidate_list.append(ref_img_name)
                assert len(canvas_candidate_list) == (sample_num + 1)
            else:
                canvas_candidate_list += canvas_candidate_list_full
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

                # Horizontal concat
                canvas = Image.new('RGB', (ref_image.width * 2, ref_image.height))
                canvas.paste(cur_image, (0, 0))
                canvas.paste(next_image, (ref_image.width, 0))

                canvas.save(cache_path)

                args_obj.image_file = cache_path
                cur_prompt = prompt
                args_obj.query = cur_prompt

                predictor.set_args(args_obj)
                out_text = predictor.eval_model()

                os.makedirs(f"{dst_dir}/{video_name}", exist_ok=True)

                # Save JSON
                save_dict = {
                    'ref_img_name': ref_img_name,
                    'cur_image_name': cur_image_name,
                    'next_image_name': next_image_name,
                    'prompt': cur_prompt,
                    'next_text': out_text,
                }

                with open(f"{dst_dir}/{video_name}/{cur_image_name}.json", 'w') as f:
                    json.dump(save_dict, f)

                if save_vis:
                    canvas = np.array(canvas)
                    plt.imshow(canvas)
                    plt.axis('off')
                    plt.text(0, 0, out_text, fontsize=12, color='red', fontweight='bold')
                    os.makedirs(f"{dst_vis_dir}/{video_name}", exist_ok=True)
                    plt.savefig(f"{dst_vis_dir}/{video_name}/{cur_image_name}.jpg", bbox_inches='tight', pad_inches=0)
                    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process painting steps using a pretrained model.")
    parser.add_argument("--data_folder", type=str, default='../../data/sample_data', help="Path to the data folder.")
    parser.add_argument("--split", type=str, default='train', choices=["train", "val", "test"], help="Data split to process.")
    parser.add_argument("--model_path", type=str, default='../../checkpoints/TP_llava_annotator', help="Path to the pretrained model.")
    parser.add_argument("--sample_num", type=int, default=100, help="Number of samples to process.")
    parser.add_argument("--save_vis", action="store_true", help="Whether to save visualization outputs.")
    parser.add_argument("--prompt", type=str, default="There are two images side by side. The right image is the next step of the left image in the painting process of a painting. Please tell me what is added to right image? The answer should be less than 2 words.", help="Prompt for the model.")

    args = parser.parse_args()
    main(args)
