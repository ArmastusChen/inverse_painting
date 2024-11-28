# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/pipelines/pipeline_animation.py

import inspect, math
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.image_processor import VaeImageProcessor
import PIL
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange
from models.positional_encoder import PositionalEncoder
import torch.nn.functional as F
from PIL import Image

from diffusers.utils.torch_utils import randn_tensor

from .context import (
    get_context_scheduler,
    get_total_steps
)

from models.ReferenceNet_attention import ReferenceNetAttention
from diffusers.models import UNet2DConditionModel
import torchvision.transforms as transforms

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class InvPaintingPipelineOutput(BaseOutput):
    images: Union[torch.Tensor, np.ndarray]
    latents: Optional[torch.Tensor] = None



def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")




class InvPaintingPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        PE: PositionalEncoder = None,
    ):
        super().__init__()
        '''
        referencenet:ReferenceNet,
        referenceencoder:ReferenceEncoder,
        '''

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            # deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            # deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        



    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device


    def decode_latents(self, latents, rank, decoder_consistency=None):
        # deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs


    
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents


    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)
            # print(image.min(), image.max())

            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs
    

    @torch.no_grad()
    def images2latents(self, images, dtype):
        """
        Convert RGB image to VAE latents
        """
        device = self._execution_device
        images = torch.from_numpy(images).float().to(dtype) / 127.5 - 1
        images = rearrange(images, "f h w c -> f c h w").to(device)
        latents = []
        for frame_idx in range(images.shape[0]):
            latents.append(self.vae.encode(images[frame_idx:frame_idx+1]).latent_dist.sample() * 0.18215)
        latents = torch.cat(latents)
        return latents

    
    

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start
    
    @torch.no_grad()
    def __call__(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_batch_size: int = 1, 
        context_schedule: str = "uniform",
        init_latents: Optional[torch.FloatTensor] = None,
        num_actual_inference_steps: Optional[int] = None,
        referencenet = None,
        cur_image_guider = None,
        clip_image_processor = None,
        clip_image_encoder = None,
        cur_condition = None,
        PE = None, 
        RP = None,
        TP = None,
        PE_guidance_scale = None,
        PE_embeddings=None,
        TP_guidance_scale = None,
        cur_guidance_scale = None, 
        reference_control_writer = None,
        reference_control_reader = None,
        source_image: str = None,
        decoder_consistency = None, 
        TP_feature = None,
        negative_TP_feature=None, 
        cur_alpha = 0.0, 
        RP_embeddings = None, 
        negative_RP_embeddings = None,
        RP_guidance_scale = None,
        combine_init = False, 
        combine_init_ratio = 0.0,
        image_adpater= None, 
        negative_PE_embeddings = None,
        img_init_latents = None, 
        **kwargs,
    ):

        source_image_PIL = Image.fromarray(source_image).convert('RGB')
        cur_condition_PIL = Image.fromarray(cur_condition).convert('RGB')

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        dtype = torch.float16


        # Define call parameters
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]


        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0 or cur_guidance_scale > 1.0 
        do_classifier_free_guidance =  do_classifier_free_guidance or PE_guidance_scale > 1.0
        do_classifier_free_guidance =  do_classifier_free_guidance or RP_guidance_scale > 1.0
        do_classifier_free_guidance =  do_classifier_free_guidance or TP_guidance_scale > 1.0


        from models.positional_encoder import get_embedder
        embedder, _  = get_embedder(10)

        if referencenet is not None:
            reference_control_writer = ReferenceNetAttention(referencenet, do_classifier_free_guidance=do_classifier_free_guidance, mode='write', fusion_blocks="full", batch_size=context_batch_size, is_image=True,dtype=dtype)
            reference_control_reader = ReferenceNetAttention(self.unet, do_classifier_free_guidance=do_classifier_free_guidance, mode='read', fusion_blocks="full", batch_size=context_batch_size, is_image=True,dtype=dtype)

        self.unet = self.unet.to(dtype)
        is_dist_initialized = kwargs.get("dist", False)
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)


        assert batch_size == 1         

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        strength = 1.0
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * 1)        

        # Prepare latent variables
        if init_latents is not None:
            latents = rearrange(init_latents, "(b f) c h w -> b c f h w", f=1)
        else:
            num_channels_latents = self.unet.in_channels
            if img_init_latents is None:
                return_image_latents = True
            else:
                return_image_latents = False

            latents_outputs = self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                dtype,
                device,
                generator,
                latents,
                image=self.image_processor.preprocess(cur_condition_PIL),
                timestep=latent_timestep,
                is_strength_max=True,
                return_noise=True,
                return_image_latents=return_image_latents,
            )

            if return_image_latents:
                latents, noise, img_init_latents = latents_outputs
            else:
                latents, noise = latents_outputs

            img_latents = self.scheduler.add_noise(img_init_latents, noise, latent_timestep)

            latents = img_latents * cur_alpha + latents * (1 - cur_alpha)


        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)



        # For img2img setting
        if num_actual_inference_steps is None:
            num_actual_inference_steps = num_inference_steps
        
        if isinstance(source_image, str):
            ref_image_latents = self.images2latents(np.array(Image.open(source_image).resize((width, height)))[None, :], dtype).cuda()
            clip_ref_image = clip_image_processor(images=Image.open(source_image).convert('RGB'), return_tensors="pt").pixel_values
        
        elif isinstance(source_image, np.ndarray):
            ref_image_latents = self.images2latents(source_image[None, :], dtype).cuda()
            clip_ref_image = clip_image_processor(images=Image.fromarray(source_image).convert('RGB'), return_tensors="pt").pixel_values

            if image_adpater is not None:
                clip_cur_image = clip_image_processor(images=cur_condition_PIL, return_tensors="pt").pixel_values

        # prepare clip image embedding
        # adapt from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L115
        clip_ref_image = clip_ref_image.to(device=latents.device)
        image_embeddings_ref = clip_image_encoder(clip_ref_image).unsqueeze(1).to(device=latents.device,dtype=latents.dtype)
        bs_embed, seq_len, _ = image_embeddings_ref.shape
        image_embeddings_ref = image_embeddings_ref.repeat(1, 1, 1)
        image_embeddings_ref = image_embeddings_ref.view(bs_embed , seq_len, -1)

        if image_adpater is not None:
            clip_cur_image = clip_cur_image.to(device=latents.device)
            image_embeddings_cur = clip_image_encoder(clip_cur_image).unsqueeze(1).to(device=latents.device,dtype=latents.dtype)
            bs_embed, seq_len, _ = image_embeddings_cur.shape
            image_embeddings_cur = image_embeddings_cur.repeat(1, 1, 1)
            image_embeddings_cur = image_embeddings_cur.view(bs_embed, seq_len, -1)
            image_embeddings = image_adpater(image_embeddings_cur, image_embeddings_ref)

        else:
            image_embeddings = image_embeddings_ref



        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)



        if not do_classifier_free_guidance:
            assert False
            encoder_hidden_states = image_embeddings
            encoder_hidden_states = torch.cat([encoder_hidden_states, img_pos_emb], dim=1)
            encoder_hidden_states = torch.cat([encoder_hidden_states, TP_feature], dim=1)

        else:
            encoder_hidden_states0 = torch.cat([negative_image_embeddings, negative_PE_embeddings, negative_TP_feature], dim=1)
            encoder_hidden_states1 = torch.cat([image_embeddings, negative_PE_embeddings, negative_TP_feature], dim=1)
            encoder_hidden_states2 = torch.cat([image_embeddings, PE_embeddings, negative_TP_feature], dim=1)
            encoder_hidden_states3 = torch.cat([image_embeddings, PE_embeddings, TP_feature], dim=1)

            encoder_hidden_states = torch.cat([encoder_hidden_states0, encoder_hidden_states1, encoder_hidden_states1, encoder_hidden_states2, encoder_hidden_states2, encoder_hidden_states3], dim=0)
            num_repeat_needed = 6



        encoder_hidden_states_ref = image_embeddings_ref.repeat(num_repeat_needed if do_classifier_free_guidance else 1, 1, 1)
        context_scheduler = get_context_scheduler(context_schedule)
        
        
        pixel_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
        cur_condition = torch.from_numpy(cur_condition.copy())[None,].to(device=device, dtype=latents.dtype).permute(0, 3, 1, 2) / 255.0
        cur_condition = pixel_transforms(cur_condition)
    
        if do_classifier_free_guidance:
            cur_condition_negative = torch.zeros_like(cur_condition).to(latents.device)
            cur_condition0 = torch.cat([cur_condition_negative, negative_RP_embeddings], dim=1)
            cur_condition1 = torch.cat([cur_condition, negative_RP_embeddings], dim=1)
            cur_condition2 = torch.cat([cur_condition, RP_embeddings], dim=1)
            cur_condition_input = torch.cat([cur_condition0, cur_condition0, cur_condition1, cur_condition1, cur_condition2, cur_condition2], dim=0)


        latents_cur = cur_image_guider(cur_condition_input)

           
        
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=(rank!=0)):

            if i == 0:
                referencenet(
                    ref_image_latents.repeat(context_batch_size * (num_repeat_needed if do_classifier_free_guidance else 1), 1, 1, 1),
                    torch.zeros_like(t),
                    encoder_hidden_states=encoder_hidden_states_ref,
                    return_dict=False,
                )
                reference_control_reader.update(reference_control_writer)

            latent_model_input = torch.cat([latents] * num_repeat_needed) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) 

    
            noise_pred = self.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=encoder_hidden_states,
                latent_cur=latents_cur,
                return_dict=False,
            )[0]

            
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_clip, noise_pred_cur, noise_pred_PE, noise_pred_RP, noise_pred_TP = noise_pred.chunk(6)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_clip - noise_pred_uncond) + cur_guidance_scale * (noise_pred_cur - noise_pred_clip) + PE_guidance_scale * (noise_pred_PE - noise_pred_cur) + RP_guidance_scale * (noise_pred_RP - noise_pred_PE) + TP_guidance_scale * (noise_pred_TP - noise_pred_RP)


            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        

            if combine_init:
                if i < int(num_inference_steps * combine_init_ratio):

                    # downsample 
                    RP_embeddings_mask = F.interpolate(RP_embeddings, size=(latents.shape[2], latents.shape[3]), mode='nearest')
                    RP_embeddings_mask = RP_embeddings_mask > 0.1
                    RP_embeddings_mask = RP_embeddings_mask.to(device=latents.device, dtype=latents.dtype)

                    img_init_latents_cur = img_init_latents
                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        img_init_latents_cur = self.scheduler.add_noise(img_init_latents, noise, torch.tensor([noise_timestep]))
                    latents = RP_embeddings_mask * latents + (1 - RP_embeddings_mask) * img_init_latents_cur
        

        images = self.decode_latents(latents, rank, decoder_consistency=decoder_consistency)

        if RP is not None:
            pixel_values_cur_mask_vis = pixel_values_cur_mask.cpu().permute(0, 2, 3, 1).float().numpy()
            images = np.concatenate([images, pixel_values_cur_mask_vis], axis=-1)


        if is_dist_initialized:
            dist.barrier()

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if not return_dict:
            return images
        
        return InvPaintingPipelineOutput(images=images, latents=latents)
