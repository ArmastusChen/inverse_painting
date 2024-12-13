<h1 align='Center'>Inverse Painting: Reconstructing The Painting Process</h1>

<div align='Center'>
            <a href="https://homes.cs.washington.edu/~boweiche/">Bowei Chen</a>&emsp;
            <a href="https://scholar.google.com/citations?user=R3sUe_EAAAAJ&hl=en">Yifan Wang</a>&emsp;
            <a href="https://homes.cs.washington.edu/~curless/">Brian Curless</a>&emsp;
            <a href="https://www.irakemelmacher.com">Ira Kemelmacher-Shlizerman</a>&emsp;
            <a href="https://www.smseitz.com">Steven M. Seitz</a>&emsp;
</div>
<div align='Center'>
    University of Washington
</div>
<div align='Center'>
<i><strong><a href='https://asia.siggraph.org/2024/' target='_blank'>SIGGRAPH Asia 2024</a></strong></i>
</div>

<div align='Center'>
    <a href='https://inversepainting.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://arxiv.org/abs/2409.20556'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://youtu.be/T89auOvTm0o'><img src='https://badges.aleen42.com/src/youtube.svg'></a>
</div>



# Installation

The code can be run under environment with Python 3.10, pytorch 2.1.2 and cuda 11.8.  (It should run with other versions, but we have not tested it).  

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to set up an environment:

    conda create --name inverse_painting python=3.10

    conda activate inverse_painting

Install the required packages:

    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

    pip install -r requirements.txt

Install LLaVA

    git clone https://github.com/haotian-liu/LLaVA.git
    cd LLaVA
    pip install -e .
    cd ..

Install xformers

    pip3 install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

# Inference
We provide demo code to run our pretrained models on any target landscape painting.


## Download Pretrained Models
Download pretrained models either from [Huggingface](https://huggingface.co/boweiche/inverse_painting) or [Google Drive](https://drive.google.com/drive/folders/1exu6Ws-NIZO-3qNO5s50b71fSQALkdvK?usp=drive_link), and then put them into the root folder.  We recommend using the following commands for downloading from Huggingface:

    git lfs install
    git clone https://huggingface.co/boweiche/inverse_painting


After downloading, the pretrained models should be organized as follows:
```text
./checkpoints/
|-- renderer
|-- RP
|-- TP_llava
|-- TP_llava_annotator    # optional, only required for training. 

./base_ckpt/
|-- clip-vit-base-patch32
|-- realisticVisionV51_v51VAE
```

## Run Demo

For demo, we provide several target paintings in `./data/demo`. You can run the demo code using the following command
```shell
python demo.py
```

The generated results will be saved in `results`. 

# Training 

The text generator, mask generator, and renderer are trained separately. You can train these models simutineously because GT text and mask instructions, instead of predicted ones, will be used to train mask generator and renderer. 


## Dataset Pre-Processing

We provide an example of sample data in `data/sample_data`. Belows are the data structure before running the data pre-processing. 

```text
./data/sample_data/train
|-- rgb/                                   # folders of training samples 
   |-- example/                            # name of the sample
      |-- {ind}_{time}.jpg                 # name of each frame, {ind} is the frame index and {time} indicates the timestamp of the frame within the video.
      |-- last_aligned_frame_inv.json      # a json file to define the frame as target painting. 
```


The following steps help you to pre-process this sample data for training, including the preparation of the GT text and mask instructions.  
You can refer to the code in `data_processing`. We also provide our processed data in `data/sample_data_processed` for your reference. 


### Prepare Text Instruction
This step prepares the ground truth (GT) text instructions by feeding both the GT current canvas and the GT next canvas into the LLaVA model.

In this codebase, rather than using the pretrained LLaVA model online, we utilize our fine-tuned version of LLaVA for more accurate GT text generation. If you have not downloaded `TP_llava_annotator` in the previous step, you can download it from [Google Drive](https://drive.google.com/drive/folders/1Lj4pSlHJTXvJdyXBWbOT6u-ZhiOyyWQG?usp=drive_link) and put it into the folder `checkpoints`. This model has been fine-tuned using the GT current image, GT next image, and GT text from our dataset, with any inaccurate GT text manually corrected.

You can now run the following commands to prepare GT text instruction. 
```shell
cd data_processing/run_llava      
python main.py   --save_vis   --model_path  ../../checkpoints/TP_llava_annotator    # you can remove --save_vis if you don't want the visualization
python make_list.py      # prepare the data format for the training of text generator
cd ../../
```

The generated text will be saved in `data/sample_data/train/text` and `data/sample_data/train/text_vis` (if you use --save_vis). 
The training data in the format of LLaVA is in `data/sample_data/train/llava_image` and `data/sample_data/train/llava_json.json`

### Prepare Mask Instruction
This step prepares the GT text instructions by computing the LPIPS difference between GT current and next canvas. 

You can now run the following commands to prepare mask text instruction. 
```shell
cd data_processing/run_lpips      
python main.py   --save_vis     # you can remove --save_vis if you don't want the visualization
cd ../../
```
The generated mask will be saved in `data/sample_data/train/lpips` and `data/sample_data/train/lpips_vis` (if you use --save_vis). 



## Training Two-Stage Pipeline
The training code is in `training_scripts`. The following three models can be trained in any order because they are not dependent on each other. 


### Train Text Generator 
```shell
cd training_scripts
bash train_text_generator.sh     # This trains a lora of LLaVA, saved in `./checkpoints/llava-v1.5-7b-task-lora`. It will complete very fast because the sample dataset is very small
bash merge_ckpt.sh               # After training, merge the lora with the base model, saved in `./checkpoints/llava-v1.5-7b-task-lora_final`
cd ..
```

### Train Mask Generator 
```shell
cd training_scripts
torchrun --nnodes=1 --nproc_per_node=1  --master_port=25678 train_mask_generator.py    --config  ../configs/train/train_mask_gen.yaml   
cd ..
```
This trains a Unet with cross-attention layers, saved in `./outputs/mask_gen`


### Train Next Frame Renderer
```shell
cd training_scripts
torchrun --nnodes=1 --nproc_per_node=1  --master_port=12678  train_renderer.py    --config   ../configs/train/train_renderer.yaml   
cd ..
```
The output will be saved in `./outputs/renderer`




## Acknowledgement

This codebase is adpated from [diffusers](https://github.com/huggingface/diffusers), [Open-AnimateAnyone
](https://github.com/guoqincode/Open-AnimateAnyone), and [LLaVA](https://github.com/haotian-liu/LLaVA).



# Disclaimer

We tested this codebase on a single NVIDIA A40 GPU. The result produced by this code might be slightly different when running on a different machine. 



# Citation

If you find our work useful for your research, please consider citing the paper:

```
@inproceedings{chen2024inverse,
  title={Inverse Painting: Reconstructing The Painting Process},
  author={Chen, Bowei and Wang, Yifan and Curless, Brian and Kemelmacher-Shlizerman, Ira and Seitz, Steven M},
  booktitle={SIGGRAPH Asia 2024 Conference Papers},
  year={2024}
}
```