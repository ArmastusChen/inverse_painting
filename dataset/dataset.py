import os, io, csv, math, random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from transformers import CLIPProcessor
import glob
import json


# adapt from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/data/dataset.py

import torch.distributed as dist
def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)

def load_im_as_tensor(im_paths):

    if isinstance(im_paths, list):
        im = [Image.open(im_path) for im_path in im_paths]
        im = [im.convert('RGB') for im in im]
        im = [np.array(im) for im in im]
        im = [torch.from_numpy(im).permute(2, 0, 1).contiguous().float()[None] for im in im]
        im = [im / 255.0 for im in im]
        im = torch.cat(im, dim=0)
    else:
        im = Image.open(im_paths)
        im = im.convert('RGB')
        im = np.array(im)
        im = torch.from_numpy(im).permute(2, 0, 1).contiguous().float()[None]
        im = im / 255.0
    return im
        
    

class InvPaintingDataset(Dataset):
    def __init__(
            self,
            data_folder,
            sample_size=768, sample_stride=4, sample_n_frames=24, sample_num=20, 
            is_image=False, clip_model_path="openai/clip-vit-base-patch32",
            is_train=True,
            pad_mode=False, 
            PE_type=None, 
            PE_time_interval=None, 
            PE_time_max=None,
            win_size=None,
        ):
        zero_rank_print(f"loading annotations from {data_folder} ...")


        self.is_train = is_train
        self.spilt = 'train' if self.is_train else 'val'

        self.sample_size = sample_size
        self.pad_mode= pad_mode
        self.PE_type = PE_type
        self.PE_time_max= PE_time_max
        self.PE_time_interval = PE_time_interval
        self.win_size = win_size
    

        self.video_dirs = glob.glob(f'{data_folder}/{self.spilt}/rgb/*')
        
        self.length = len(self.video_dirs)
        zero_rank_print(f"video nums: {self.length}")
        print(f"video nums: {self.length}")

        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        self.sample_num = sample_num
        
        self.clip_image_processor = CLIPProcessor.from_pretrained(clip_model_path,local_files_only=True)
 
        self.pixel_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def __len__(self):
        return self.length
    
    def get_batch(self,idx):
        video_dir = self.video_dirs[idx]
        video_name = video_dir.split('/')[-1]

        last_aligned_frame_inv_path = f'{video_dir}/last_aligned_frame_inv.json'

        # load json 
        with open(last_aligned_frame_inv_path) as f:
            last_aligned_frame_inv_path_dict = json.load(f)
            
        # first sample the ref image  (final canvas) randomly from  key of the last_aligned_frame_inv_path_dict
        ref_img_name = random.choice(list(last_aligned_frame_inv_path_dict.keys()))


        canvas_candidate_list_full = last_aligned_frame_inv_path_dict[ref_img_name]
        

        if len(canvas_candidate_list_full) == 1:
            return None, None, None, None


        # compute time difference 
        canvas_candidate_list_full_time = canvas_candidate_list_full + [ref_img_name]
        canvas_candidate_list_full_time = [int(name.split('_')[-1].split(':')[0]) * 60 + int(name.split('_')[-1].split(':')[1]) for name in canvas_candidate_list_full_time]
        

        # make the first time to be 5
        if canvas_candidate_list_full_time[0] > 30:
            canvas_candidate_list_full_time[0] = 30



        canvas_candidate_list_full_time = [canvas_candidate_list_full_time[i+1] - canvas_candidate_list_full_time[i] for i in range(len(canvas_candidate_list_full_time) - 1)]

        
        # first frame as white image, last frame as final canvas
        canvas_candidate_list = ['white']
        # remove first frame (likely white image)
        canvas_candidate_list_full = canvas_candidate_list_full[1:]

        

        
        if len(canvas_candidate_list_full) >= (self.sample_num - 1):
            # uniformly sample 10 frames from canvas_candidate_list
            sample_inds = np.round(np.linspace(0, len(canvas_candidate_list_full) - 1, self.sample_num - 1)).astype(int)
            canvas_candidate_list = canvas_candidate_list + [canvas_candidate_list_full[i] for i in sample_inds]

            # add last frame
            canvas_candidate_list.append(ref_img_name)
        

            assert len(canvas_candidate_list) == (self.sample_num + 1)
        else:
            # add last frame
            canvas_candidate_list = canvas_candidate_list + canvas_candidate_list_full
            canvas_candidate_list.append(ref_img_name)




        ref_img_path = f'{video_dir}/{ref_img_name}.jpg'
        
        pixel_values_ref_img = load_im_as_tensor(ref_img_path)

        ref_img_pil = Image.open(ref_img_path)
        clip_ref_image = self.clip_image_processor(images=ref_img_pil, return_tensors="pt").pixel_values



        # then sample current image 
        cur_img_ind = np.random.choice(len(canvas_candidate_list[:-1]))
        cur_img_name = canvas_candidate_list[cur_img_ind]

        if self.PE_type == 'rel':
            assert False
            cur_img_pos = cur_img_ind / (len(canvas_candidate_list) - 1)
        elif self.PE_type == 'abs':
            cur_img_pos = canvas_candidate_list_full_time[cur_img_ind]

            if  self.PE_time_interval > 1:
                # make the cur_img_pos to be cloestest multiple of PE_time_interval, +1 is to avoid 0 
                cur_img_pos = round(cur_img_pos / self.PE_time_interval) * self.PE_time_interval 


            # print(f'cur_img_pos: {cur_img_pos}')
            cur_img_pos = cur_img_pos / self.PE_time_max

            # clamp to 0 to 1 
            cur_img_pos = max(0, min(1, cur_img_pos))

            
        #to [-1, 1]
        cur_img_pos = cur_img_pos * 2 - 1
        

        if cur_img_name == 'white':
            pixel_values_cur = torch.ones_like(pixel_values_ref_img)
            cur_img_path = 'white'
            cur_img_pil = Image.new('RGB', (224, 224), (255, 255, 255))

        else:
            cur_img_path = f'{video_dir}/{cur_img_name}.jpg'
            pixel_values_cur = load_im_as_tensor(cur_img_path)
            cur_img_pil = Image.open(cur_img_path)

   




        clip_cur_image = self.clip_image_processor(images=cur_img_pil, return_tensors="pt").pixel_values

        # get next canvas
        gt_img_name = canvas_candidate_list[cur_img_ind + 1]
        gt_img_path = f'{video_dir}/{gt_img_name}.jpg'
        pixel_values = load_im_as_tensor(gt_img_path)

        # # to [-1, 1]
        next_img_pos = cur_img_pos


        # if not  pixel_values pixel_values_cur pixel_values_ref_img should have same shape, return 
        if not (pixel_values.shape == pixel_values_cur.shape == pixel_values_ref_img.shape):
            return None, None, None, None
        
        assert pixel_values.shape == pixel_values_cur.shape == pixel_values_ref_img.shape



        if self.pad_mode == 'pad_to_square+resize':
            # Pad to square image
            max_size = max(pixel_values.shape[2], pixel_values.shape[3])
            pixel_values = torch.nn.functional.pad(pixel_values, (0, max_size - pixel_values.shape[3], 0, max_size - pixel_values.shape[2]))
            pixel_values_cur = torch.nn.functional.pad(pixel_values_cur, (0, max_size - pixel_values_cur.shape[3], 0, max_size - pixel_values_cur.shape[2]))
            pixel_values_ref_img = torch.nn.functional.pad(pixel_values_ref_img, (0, max_size - pixel_values_ref_img.shape[3], 0, max_size - pixel_values_ref_img.shape[2]))

            # resize to sample_size
            pixel_values = torch.nn.functional.interpolate(pixel_values, size=(self.sample_size, self.sample_size), mode='bilinear')
            pixel_values_cur = torch.nn.functional.interpolate(pixel_values_cur, size=(self.sample_size, self.sample_size), mode='bilinear')
            pixel_values_ref_img = torch.nn.functional.interpolate(pixel_values_ref_img, size=(self.sample_size, self.sample_size), mode='bilinear')


        if self.pad_mode == 'pad_to_16':

            # pad the border of them to make it multiple of 16, shape is [1, 3, H, W]
            pad_size = [16 - pixel_values.shape[2] % 16, 16 - pixel_values.shape[3] % 16]

            # if pad_size[0] != 16 or pad_size[1] != 16:
            if pad_size[0] == 16:
                pad_size[0] = 0
            if pad_size[1] == 16:
                pad_size[1] = 0
            # pad the border of them to make it multiple of 16, shape is [1, 3, H, W]
            pixel_values = torch.nn.functional.pad(pixel_values, (0, pad_size[1], 0, pad_size[0]))
            pixel_values_cur = torch.nn.functional.pad(pixel_values_cur, (0, pad_size[1], 0, pad_size[0]))
            pixel_values_ref_img = torch.nn.functional.pad(pixel_values_ref_img, (0, pad_size[1], 0, pad_size[0]))

            
 
        if self.is_image:
            pixel_values = pixel_values[0]
            pixel_values_cur = pixel_values_cur[0]
        
        pixel_values_ref_img = pixel_values_ref_img[0]
        

        return pixel_values, pixel_values_cur, clip_ref_image, clip_cur_image, pixel_values_ref_img, cur_img_pos, next_img_pos, cur_img_path, gt_img_path, ref_img_path
    
    def __getitem__(self, idx):
        while True:
            idx = random.randint(0, self.length-1) 
            try:
                pixel_values, pixel_values_cur, clip_ref_image, clip_cur_image, pixel_values_ref_img, cur_img_pos, next_img_pos, cur_img_path, gt_img_path, ref_img_path = self.get_batch(idx)
                if pixel_values is not None:
                    break
            except Exception as e:
                print('exception!!!!!!')
                pass
                
      
            

        pixel_values = self.pixel_transforms(pixel_values)
        pixel_values_cur = self.pixel_transforms(pixel_values_cur)
        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
        pixel_values_ref_img = self.pixel_transforms(pixel_values_ref_img)
        pixel_values_ref_img = pixel_values_ref_img.squeeze(0)
        cur_img_pos, next_img_pos = torch.tensor(cur_img_pos).unsqueeze(0), torch.tensor(next_img_pos).unsqueeze(0)

        
        
        # clip_ref_image = clip_ref_image.unsqueeze(1) # [bs,1,768]
        drop_image_embeds = 1 if random.random() < 0.1 else 0
        drop_time_step = 1 if random.random() < 0.1 else 0
        drop_feature = 1 if random.random() < 0.1 else 0
        drop_cur_cond = 1 if random.random() < 0.1 else 0
        drop_RP = 1 if random.random() < 0.1 else 0
        sample = dict(
            pixel_values=pixel_values, 
            pixel_values_cur=pixel_values_cur,
            clip_ref_image=clip_ref_image,
            clip_cur_image=clip_cur_image,
            pixel_values_ref_img=pixel_values_ref_img,
            drop_image_embeds=drop_image_embeds,
            drop_time_step = drop_time_step,
            drop_feature = drop_feature,
            drop_cur_cond=drop_cur_cond,
            drop_RP=drop_RP,
            cur_img_pos=cur_img_pos,
            next_img_pos=next_img_pos,
            cur_img_path=cur_img_path,
            gt_img_path=gt_img_path,
            ref_img_path=ref_img_path

            
            )
        
        return sample




# https://github.com/tencent-ailab/IP-Adapter/blob/main/tutorial_train.py#L341

def collate_fn(data):

    
    pixel_values = torch.stack([example["pixel_values"] for example in data])
    pixel_values_cur = torch.stack([example["pixel_values_cur"] for example in data])
    clip_ref_image = torch.cat([example["clip_ref_image"] for example in data])
    clip_cur_image = torch.cat([example["clip_cur_image"] for example in data])
    pixel_values_ref_img = torch.stack([example["pixel_values_ref_img"] for example in data])
    cur_img_pos = torch.stack([example["cur_img_pos"] for example in data])
    next_img_pos = torch.stack([example["next_img_pos"] for example in data])

    drop_image_embeds = [example["drop_image_embeds"] for example in data]
    drop_image_embeds = torch.Tensor(drop_image_embeds)

    drop_time_step = [example["drop_time_step"] for example in data]
    drop_time_step = torch.Tensor(drop_time_step)

    drop_feature = [example["drop_feature"] for example in data]
    drop_feature = torch.Tensor(drop_feature)

    drop_cur_cond = [example["drop_cur_cond"] for example in data]
    drop_cur_cond = torch.Tensor(drop_cur_cond)

    drop_RP = [example["drop_RP"] for example in data]
    drop_RP = torch.Tensor(drop_RP)


    
    return {
        "pixel_values": pixel_values,
        "pixel_values_cur": pixel_values_cur,
        "clip_ref_image": clip_ref_image,
        "clip_cur_image": clip_cur_image,
        "pixel_values_ref_img": pixel_values_ref_img,
        "drop_image_embeds": drop_image_embeds,
        "drop_time_step": drop_time_step,
        "drop_feature": drop_feature,
        "drop_cur_cond": drop_cur_cond,
        "drop_RP": drop_RP,
        "cur_img_pos": cur_img_pos,
        "next_img_pos": next_img_pos,
        "cur_img_path": [example["cur_img_path"] for example in data],
        "gt_img_path": [example["gt_img_path"] for example in data],
        "ref_img_path": [example["ref_img_path"] for example in data]
        
    }

