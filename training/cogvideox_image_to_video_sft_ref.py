# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import math
import os
import shutil
import sys
import random
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Union, Optional
import cv2
import copy
import diffusers
import torch
import transformers
import wandb
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel
from decord import VideoReader, cpu
from einops import rearrange
import numpy as np
import decord  # isort:skip
decord.bridge.set_bridge("torch")
from PIL import Image
from args import get_args  # isort:skip
from dataset_ref import VideoDatasetWithResizingTrackingWorldModel, BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop, VideoDatasetWithResizingTracking  # isort:skip
from text_encoder import compute_prompt_embeddings  # isort:skip
from utils import get_gradient_norm, get_optimizer, prepare_rotary_positional_embeddings, print_memory, reset_memory, read_video, concatenate_videos, pil_list_to_tensor # isort:skip

from diffusers.utils import load_image
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from models.cogvideox_tracking_ref import CogVideoXImageToVideoPipelineTracking
from models.cogvideox_tracking import CogVideoXTransformer3DModelTracking

from models.cogvideox_tracking_ref import CogVideoXTransformer3DModelTrackingRef

from diffusers.utils import export_to_video, load_image, load_video
from diffusers.image_processor import VaeImageProcessor



logger = get_logger(__name__)

def save_model_card(
    repo_id: str,
    videos=None,
    base_model: str = None,
    validation_prompt=None,
    repo_folder=None,
    fps=8,
):
    widget_dict = []
    if videos is not None:
        for i, video in enumerate(videos):
            export_to_video(video, os.path.join(repo_folder, f"final_video_{i}.mp4", fps=fps))
            widget_dict.append(
                {
                    "text": validation_prompt if validation_prompt else " ",
                    "output": {"url": f"video_{i}.mp4"},
                }
            )

    model_description = f"""
# CogVideoX Full Finetune

<Gallery />

## Model description

This is a full finetune of the CogVideoX model `{base_model}`.

The model was trained using [CogVideoX Factory](https://github.com/a-r-r-o-w/cogvideox-factory) - a repository containing memory-optimized training scripts for the CogVideoX family of models using [TorchAO](https://github.com/pytorch/ao) and [DeepSpeed](https://github.com/microsoft/DeepSpeed). The scripts were adopted from [CogVideoX Diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/cogvideo/train_cogvideox_lora.py).

## Download model

[Download LoRA]({repo_id}/tree/main) in the Files & Versions tab.

## Usage

Requires the [🧨 Diffusers library](https://github.com/huggingface/diffusers) installed.

```py
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained("{repo_id}", torch_dtype=torch.bfloat16).to("cuda")

video = pipe("{validation_prompt}", guidance_scale=6, use_dynamic_cfg=True).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

For more details, checkout the [documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox) for CogVideoX.

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE) and [here](https://huggingface.co/THUDM/CogVideoX-2b/blob/main/LICENSE).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=validation_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-video",
        "diffusers-training",
        "diffusers",
        "cogvideox",
        "cogvideox-diffusers",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))





def log_validation(
    accelerator: Accelerator,
    pipe: Union[CogVideoXPipeline, CogVideoXImageToVideoPipelineTracking],
    vae: Union[AutoencoderKLCogVideoX, None],
    dataset: Union[VideoDatasetWithResizingTracking, None],
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    epoch,
    is_final_validation: bool = False,
    random_flip: Optional[float] = None,
):


    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    negative_prompt = pipeline_args.pop("negative_prompt", None)
    guidance_scale = pipeline_args.pop("guidance_scale", None)
    max_sequence_length = pipeline_args.pop("max_sequence_length", None)
    use_dynamic_cfg = pipeline_args.pop("use_dynamic_cfg", None)

    validation_dir = pipeline_args.pop("validation_dir", None)
    height = pipeline_args.pop("height", None)
    width = pipeline_args.pop("width", None)

    results = []
    num_val = len(validation_dir)
    cond_frames = args.cond_frames
    device = accelerator.device

    pipe = pipe.to(accelerator.device)
    logger.info(
        f"Running validation... \n Generating {num_val} videos."
    )

    for i in range(num_val):
        validation_dir_test = validation_dir[i]
        val_name = validation_dir_test.split("/")[1]
        
        with open(f"{validation_dir_test}/prompt.txt", "r", encoding="utf-8") as file:
            prompt = file.read().strip()        
        video_mt = load_video(f"{validation_dir_test}/Vid_masktarget.mp4")
        video_target = load_video(f"{validation_dir_test}/Vid_target.mp4")
        video_source = load_video(f"{validation_dir_test}/Vid_source.mp4")
        num_frames = len(video_mt)
        video_mt_pil = copy.deepcopy(video_mt)  
        
        video_mt_raw = copy.deepcopy(video_mt) 
        needgen_frames = num_frames - cond_frames
        video_target_pil = video_source[-cond_frames:] + video_target[1:needgen_frames+1]
        video_mt_pil = video_source[-cond_frames:] + video_mt_raw[1:needgen_frames+1]

        video_mt = pil_list_to_tensor(video_mt_pil, target_size=(height, width))
        video_target = pil_list_to_tensor(video_target_pil, target_size=(height, width))


        ### preprocess ###
        image_tensor = video_target[0].clone()
        tracking_tensor = video_mt.clone()
        video = video_source[-cond_frames:]   ##add

        # Process tracking tensor
        tracking_maps = video_mt.float() # [T, C, H, W]
        tracking_maps = tracking_maps.to(device=device, dtype=torch.bfloat16)
        tracking_cond_frames = tracking_maps[:cond_frames]  # Get first frame as [f, C, H, W]

        # Convert tensor to PIL Image
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        height, width = image.height, image.width

        # Process tracking tensor
        tracking_maps = video_mt.float() # [T, C, H, W]
        tracking_maps = tracking_maps.to(device=device, dtype=torch.bfloat16)
        tracking_cond_frames = tracking_maps[:cond_frames]  # Get first frame as [f, C, H, W]
        height, width = tracking_cond_frames.shape[2], tracking_cond_frames.shape[3]

        #### process reference condition ####
        video_source_tensor = read_video(f"{validation_dir_test}/Vid_source.mp4")
        reference_frames = video_source_tensor.to(torch.bfloat16).to(device)

        with torch.no_grad():
            print("Encoding tracking maps")
            tracking_maps = tracking_maps.unsqueeze(0) # [B, T, C, H, W]
            tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            tracking_latent_dist = pipe.vae.encode(tracking_maps).latent_dist
            tracking_maps = tracking_latent_dist.sample() * pipe.vae.config.scaling_factor
            tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

        
            #  Generate the video frames based on the prompt.
            video_generate = pipe(
                prompt=prompt,
                negative_prompt="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
                video=video,                          ## * ##
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=49,
                use_dynamic_cfg=use_dynamic_cfg,
                guidance_scale=guidance_scale,
                generator=generator,
                tracking_maps=tracking_maps,          ## * ##
                tracking_video=tracking_cond_frames,  ## * ##
                height=height,
                width=width,
                reference=reference_frames[:, :, -10:, :, :],                   ## * ##

                ).frames[0]

        video_viz = concatenate_videos(video_target_pil, video_mt_pil, video_generate, (480, 832))
        results.append(video_viz)
    
    video_filenames = []
    for i, video_viz in enumerate(results):
        name = validation_dir[i].split("/")[1]
        
        filename = os.path.join(args.output_dir, f"{name}__ep{epoch}_{i}th.mp4")
        print("filename:", filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        export_to_video(video_viz, filename, fps=20)
        video_filenames.append(filename)


    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {val_name}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )
    torch.cuda.empty_cache()
    return results



def process_reference(vae,
                      image_processor,
                      reference,
                      device,
                      dtype,
                      height,
                      width,
                      batch_size,
                      do_classifier_free_guidance=False
                      ):
    
    ref_length = reference.shape[2]
    ref_video = image_processor.preprocess(
        rearrange(reference, "b c f h w -> (b f) c h w"), height=height, width=width
    )
    ref_video = rearrange(ref_video, "(b f) c h w -> b c f h w", f=ref_length)
    bs = 1
    ref_video = ref_video.to(device=device, dtype=vae.dtype)
    new_ref_video = []
    for i in range(0, ref_video.shape[0], bs):
        video_bs = ref_video[i : i + bs]
        video_bs = vae.encode(video_bs)[0]
        video_bs = video_bs.sample()
        new_ref_video.append(video_bs)
    new_ref_video = torch.cat(new_ref_video, dim=0)
    new_ref_video = new_ref_video * vae.config.scaling_factor
            
    ref_latents = new_ref_video.repeat(
        batch_size // new_ref_video.shape[0], 1, 1, 1, 1
    )
    
    ref_latents = ref_latents.to(device=device, dtype=dtype)
    ref_latents = rearrange(ref_latents, "b c f h w -> b f c h w")
    ref_input = (
        torch.cat([ref_latents] * 2) if do_classifier_free_guidance else ref_latents
    )
    return ref_input







class CollateFunction:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        return {
            "videos": videos,
            "prompts": prompts,
        }

class CollateFunctionTracking:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        tracking_maps = [x["tracking_map"] for x in data[0]]
        tracking_maps = torch.stack(tracking_maps).to(dtype=self.weight_dtype, non_blocking=True)

        return {
            "videos": videos,
            "prompts": prompts,
            "tracking_maps": tracking_maps,
        }

class CollateFunctionImageTracking:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        images = [x["image"] for x in data[0]]
        images = torch.stack(images).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        tracking_maps = [x["tracking_map"] for x in data[0]]
        tracking_maps = torch.stack(tracking_maps).to(dtype=self.weight_dtype, non_blocking=True)

        return {
            "images": images,
            "videos": videos,
            "prompts": prompts,
            "tracking_maps": tracking_maps,
        }


class CollateFunctionWorldModelTracking:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        
        prompts = [x["prompt"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)


        reference = [x["reference"] for x in data[0]]
        reference = torch.stack(reference).to(dtype=self.weight_dtype, non_blocking=True)

        images = [x["image"] for x in data[0]]
        images = torch.stack(images).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        tracking_maps = [x["tracking_map"] for x in data[0]]
        tracking_maps = torch.stack(tracking_maps).to(dtype=self.weight_dtype, non_blocking=True)

        return {
            "reference": reference,
            "images": images,
            "videos": videos,
            "prompts": prompts,
            "tracking_maps": tracking_maps,
        }



def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    if not args.tracking_column:
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=load_dtype,
            revision=args.revision,
            variant=args.variant,
        )
    else:
        transformer = CogVideoXTransformer3DModelTrackingRef.from_pretrained_2d(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            # torch_dtype=load_dtype,
            # revision=args.revision,
            # variant=args.variant,
            # num_tracking_blocks=args.num_tracking_blocks,
            # is_train_cross=args.is_train_cross, 
            # cross_attn_dim_head=128,
            # cross_attn_in_channels=16,
            # cross_attn_interval=args.cross_attn_interval,
            # cross_attn_num_heads=16
        ).to(dtype=load_dtype) 

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    # transformer.requires_grad_(True)
    # print("===================trainable_module_name====================")
    # transformer.requires_grad_(False)
    # for name, param in transformer.named_parameters():
        
    #     for trainable_module_name in args.trainable_modules:
    #         if trainable_module_name in name:
    #             print("===> train params:", name)
    #             param.requires_grad = True
    #             break    
    
    
    VAE_SCALING_FACTOR = vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model: CogVideoXTransformer3DModelTrackingRef
                    model = unwrap_model(model)
                    model.save_pretrained(
                        os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB"
                    )
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        transformer_ = None
        init_under_meta = False
        print("load_model_hook")

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(model).__class__}")
            print("load_model_hook 11111")

        else:

            with init_empty_weights():
                transformer_ = CogVideoXTransformer3DModelTracking.from_config(
                    args.pretrained_model_name_or_path, subfolder="transformer"
                )
                init_under_meta = True

        load_model = CogVideoXTransformer3DModelTracking.from_pretrained(os.path.join(input_dir, "transformer"))
        transformer_.register_to_config(**load_model.config)
        transformer_.load_state_dict(load_model.state_dict(), assign=init_under_meta)
        del load_model

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    def load_model_hook_tracking(models, input_dir):
        transformer_ = None
        init_under_meta = False
        print("load_model_hook_tracking")

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(model).__class__}")
            print("load_model_hook_tracking 11111")

        else:

            with init_empty_weights():
                transformer_ = CogVideoXTransformer3DModelTrackingRef.from_pretrained_2d(
                    args.pretrained_model_name_or_path, subfolder="transformer", 
                )
                # transformer_ = CogVideoXTransformer3DModelTrackingRef.from_config(
                #     args.pretrained_model_name_or_path, subfolder="transformer", 
                #     num_tracking_blocks=args.num_tracking_blocks,
                #     is_train_cross=args.is_train_cross, 
                #     cross_attn_dim_head=128,
                #     cross_attn_in_channels=16,
                #     cross_attn_interval=args.cross_attn_interval,
                #     cross_attn_num_heads=16
                # )
                init_under_meta = True

        load_model = CogVideoXTransformer3DModelTrackingRef.from_pretrained_2d(os.path.join(input_dir, "transformer"))
        transformer_.register_to_config(**load_model.config)
        transformer_.load_state_dict(load_model.state_dict(), assign=init_under_meta)
        del load_model

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook if args.resume_from_checkpoint is None else load_model_hook_tracking)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)

    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(
        params_to_optimize=params_to_optimize,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        prodigy_decouple=args.prodigy_decouple,
        prodigy_use_bias_correction=args.prodigy_use_bias_correction,
        prodigy_safeguard_warmup=args.prodigy_safeguard_warmup,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        use_torchao=args.use_torchao,
        use_deepspeed=use_deepspeed_optimizer,
        use_cpu_offload_optimizer=args.use_cpu_offload_optimizer,
        offload_gradients=args.offload_gradients,
    )

    # Dataset and DataLoader
    if args.video_reshape_mode is None:
        if args.tracking_column is not None:
            dataset_init_kwargs = {
                "data_root": args.data_root,
                "dataset_file": args.dataset_file,
                "caption_column": args.caption_column,
                "tracking_column": args.tracking_column,
                "video_column": args.video_column,
                "max_num_frames": args.max_num_frames,
                "id_token": args.id_token,
                "height_buckets": args.height_buckets,
                "width_buckets": args.width_buckets,
                "frame_buckets": args.frame_buckets,
                "load_tensors": args.load_tensors,
                "random_flip": args.random_flip,
                "cond_frames": args.cond_frames,
                "ref_frames": args.ref_frames,

                "image_to_video": True,
            }   
            train_dataset = VideoDatasetWithResizingTrackingWorldModel(**dataset_init_kwargs) ## * ##

            # train_dataset = VideoDatasetWithResizingTracking(**dataset_init_kwargs) ## * ##
        else:
            dataset_init_kwargs = {
                "data_root": args.data_root,
                "dataset_file": args.dataset_file,
                "caption_column": args.caption_column,
                "video_column": args.video_column,
                "max_num_frames": args.max_num_frames,
                "id_token": args.id_token,
                "height_buckets": args.height_buckets,
                "width_buckets": args.width_buckets,
                "frame_buckets": args.frame_buckets,
                "load_tensors": args.load_tensors,
                "random_flip": args.random_flip,
                "cond_frames": args.cond_frames,
                "image_to_video": True,
            } 
            train_dataset = VideoDatasetWithResizing(**dataset_init_kwargs)
    else:
        train_dataset = VideoDatasetWithResizeAndRectangleCrop(
            video_reshape_mode=args.video_reshape_mode, **dataset_init_kwargs
        )

    collate_fn = CollateFunction(weight_dtype, args.load_tensors)
    collate_fn_tracking = CollateFunctionTracking(weight_dtype, args.load_tensors)
    collate_fn_image_tracking = CollateFunctionImageTracking(weight_dtype, args.load_tensors)
    collate_fn_worldmodel_tracking = CollateFunctionWorldModelTracking(weight_dtype, args.load_tensors)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=BucketSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True),
        collate_fn=collate_fn if args.tracking_column is None else collate_fn_worldmodel_tracking,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.use_cpu_offload_optimizer:
        lr_scheduler = None
        accelerator.print(
            "CPU Offload Optimizer cannot be used with DeepSpeed or builtin PyTorch LR Schedulers. If "
            "you are training with those settings, they will be ignored."
        )
    else:
        if use_deepspeed_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=args.max_train_steps * accelerator.num_processes,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            )
        else:
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
                num_cycles=args.lr_num_cycles,
                power=args.lr_power,
            )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-sft"
        accelerator.init_trackers(tracker_name, config=vars(args))

        accelerator.print("===== Memory before training =====")
        reset_memory(accelerator.device)
        print_memory(accelerator.device)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num trainable parameters = {num_trainable_parameters}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num epochs = {args.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            print("path get:", path)
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            
            print(f"Resuming from checkpoint {os.path.join(args.output_dir, path)}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    transformer.train()
    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device, dtype=torch.float32)
    
    image_processor = VaeImageProcessor(vae_scale_factor= 2 ** (len(vae.config.block_out_channels) - 1))
    
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            accelerator.wait_for_everyone()

            models_to_accumulate = [transformer]
            gradient_norm_before_clip = None
            gradient_norm_after_clip = None

            
            with accelerator.accumulate(models_to_accumulate):

                reference = batch["reference"].to(accelerator.device, non_blocking=True) # [B, F, C, H, W] REFERENCE
                reference = rearrange(reference, "b f c h w -> b c f h w")
                reference = (reference+1)/2
                videos = batch["videos"].to(accelerator.device, non_blocking=True) # [B, F, C, H, W] VIDEO_TARGET
                images = batch["images"].to(accelerator.device, non_blocking=True) # [B, F, C, H, W] VIDEO_COND
                prompts = batch["prompts"]

                if args.tracking_column is not None:
                    tracking_maps = batch["tracking_maps"].to(accelerator.device, non_blocking=True) # [B, F, C, H, W] VIDEO_MT
                    tracking_image = tracking_maps.clone()  #### **** ####

                # print("reference:", reference.shape, reference.min(), reference.max()) #[2, 49, 3, 480, 720]
                # print("videos:", videos.shape) #[2, 49, 3, 480, 720]
                # print("images:", images.shape) #[2, 5, 3, 480, 720]
                # print("tracking_maps:", tracking_maps.shape) #[2, 49, 3, 480, 720]
                # print("tracking_image:", tracking_image.shape) #[2, 5, 3, 480, 720]
                
                
                
                # Encode videos
                if not args.load_tensors:
                    videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                    latent_dist = vae.encode(videos).latent_dist

                    images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                    image_noise_sigma = torch.normal(
                        mean=-3.0, std=0.5, size=(images.size(0),), device=accelerator.device, dtype=weight_dtype
                    )
                    image_noise_sigma = torch.exp(image_noise_sigma)
                    noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
                    image_latent_dist = vae.encode(noisy_images).latent_dist

                    if args.tracking_column is not None:
                        tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                        tracking_latent_dist = vae.encode(tracking_maps).latent_dist

                        tracking_image = tracking_image.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                        tracking_image_latent_dist = vae.encode(tracking_image).latent_dist
                else:
                    latent_dist = DiagonalGaussianDistribution(videos)
                    image_latent_dist = DiagonalGaussianDistribution(images)

                image_latents = image_latent_dist.sample() * VAE_SCALING_FACTOR
                image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                video_latents = latent_dist.sample() * VAE_SCALING_FACTOR
                video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                
                cond_frames_latent_t = (args.cond_frames - 1) // 4 + 1
                padding_shape = (video_latents.shape[0], video_latents.shape[1] - cond_frames_latent_t, *video_latents.shape[2:])
                # padding_shape = (video_latents.shape[0], video_latents.shape[1] - args.cond_frames, *video_latents.shape[2:])
                

                
                latent_padding = image_latents.new_zeros(padding_shape)
                if image_latents.shape[1] != video_latents.shape[1]:
                    image_latents = torch.cat([image_latents, latent_padding], dim=1)
                    
                tracking_image_latent_dist = tracking_image_latent_dist.sample() * VAE_SCALING_FACTOR
                tracking_image_latent_dist = tracking_image_latent_dist.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                tracking_image_latent_dist = tracking_image_latent_dist.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                tracking_latent_padding = tracking_image_latent_dist.new_zeros(padding_shape)
                
                if tracking_image_latent_dist.shape[1] != video_latents.shape[1]:         
                    tracking_image_latents = torch.cat([tracking_image_latent_dist, tracking_latent_padding], dim=1)
                else:
                    tracking_image_latents = tracking_image_latent_dist.clone()
                    
                    
                if random.random() < args.noised_image_dropout:
                    image_latents = torch.zeros_like(image_latents)
                    tracking_image_latents = torch.zeros_like(tracking_image_latents)

                if args.tracking_column is not None:
                    tracking_maps = tracking_latent_dist.sample() * VAE_SCALING_FACTOR
                    tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                    tracking_maps = tracking_maps.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                # Encode prompts
                if not args.load_tensors:
                    prompt_embeds = compute_prompt_embeddings(
                        tokenizer,
                        text_encoder,
                        prompts,
                        model_config.max_text_seq_length,
                        accelerator.device,
                        weight_dtype,
                        requires_grad=False,
                    )
                else:
                    prompt_embeds = prompts.to(dtype=weight_dtype)

                # Sample noise that will be added to the latents
                noise = torch.randn_like(video_latents)
                batch_size, num_frames, num_channels, height, width = video_latents.shape

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    dtype=torch.int64,
                    device=accelerator.device,
                )

                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=height * VAE_SCALE_FACTOR_SPATIAL,
                        width=width * VAE_SCALE_FACTOR_SPATIAL,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                        patch_size=model_config.patch_size,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
                noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)
                tracking_latents = torch.cat([tracking_maps, tracking_image_latents], dim=2)


                ref_input = process_reference(vae, image_processor,
                                                reference,
                                                accelerator.device,
                                                weight_dtype,
                                                height,
                                                width,
                                                batch_size,
                                                )
        

                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    ref_hidden_states=ref_input, 

                    tracking_maps=tracking_latents,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)

                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = video_latents

                loss = torch.mean(
                    (weights * (model_pred - target) ** 2).reshape(batch_size, -1),
                    dim=1,
                )
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    gradient_norm_before_clip = get_gradient_norm(transformer.parameters())
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    gradient_norm_after_clip = get_gradient_norm(transformer.parameters())

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                if not args.use_cpu_offload_optimizer:
                    lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else args.learning_rate
            logs = {"loss": loss.detach().item(), "lr": last_lr}
            # gradnorm + deepspeed: https://github.com/microsoft/DeepSpeed/issues/4555
            if accelerator.distributed_type != DistributedType.DEEPSPEED:
                logs.update(
                    {
                        "gradient_norm_before_clip": gradient_norm_before_clip,
                        "gradient_norm_after_clip": gradient_norm_after_clip,
                    }
                )
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:   ###***###
                # if (epoch + 1) % args.validation_epochs == 0 or global_step % 10 == 0:
                if global_step % 10 == 0:

                    accelerator.print("===== Memory before validation =====")
                    print_memory(accelerator.device)
                    torch.cuda.synchronize(accelerator.device)
                    transformer.eval()

                    if args.tracking_column is None:
                        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            transformer=unwrap_model(transformer),
                            scheduler=scheduler,
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=weight_dtype,
                        )
                    else:
                        pipe = CogVideoXImageToVideoPipelineTracking.from_pretrained(
                            args.pretrained_model_name_or_path,
                            transformer=unwrap_model(transformer),
                            scheduler=scheduler,
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=weight_dtype,
                        )

                    if args.enable_slicing:
                        pipe.vae.enable_slicing()
                    if args.enable_tiling:
                        pipe.vae.enable_tiling()
                    if args.enable_model_cpu_offload:
                        pipe.enable_model_cpu_offload()

                    
                    pipeline_args = {
                        "validation_dir": args.validation_dir,
                        "negative_prompt": "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
                        "guidance_scale": args.guidance_scale,
                        "use_dynamic_cfg": args.use_dynamic_cfg,
                        "height": args.height,
                        "width": args.width,
                        "max_sequence_length": model_config.max_text_seq_length,
                    }
                    
                    if args.tracking_column is not None:
                        pipeline_args["tracking_map_path"] = args.tracking_map_path

                    log_validation(
                        accelerator=accelerator,
                        pipe=pipe,
                        vae=vae,
                        dataset=train_dataset,
                        args=args,
                        pipeline_args=pipeline_args,
                        epoch=0,
                        is_final_validation=False,
                    )

                    transformer.train()
                    accelerator.print("===== Memory after validation =====")
                    print_memory(accelerator.device)
                    reset_memory(accelerator.device)

                    del pipe
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(accelerator.device)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        dtype = (
            torch.float16
            if args.mixed_precision == "fp16"
            else torch.bfloat16
            if args.mixed_precision == "bf16"
            else torch.float32
        )
        transformer = transformer.to(dtype)

        if args.tracking_column is None:
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=transformer,  # Use trained transformer
                revision=args.revision,
                variant=args.variant,
                torch_dtype=dtype,
            )
        else:
            pipe = CogVideoXImageToVideoPipelineTracking.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=transformer,  # Use trained transformer 
                revision=args.revision,
                variant=args.variant,
                torch_dtype=dtype,
            )

        pipe.save_pretrained(
            args.output_dir,
            safe_serialization=True,
            max_shard_size="5GB",
        )

        # Cleanup trained models to save memory
        if args.load_tensors:
            del pipe
        else:
            del text_encoder, vae, pipe

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

        accelerator.print("===== Memory before testing =====")
        print_memory(accelerator.device)
        reset_memory(accelerator.device)

        # Final test inference
        if args.tracking_column is None:
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=unwrap_model(transformer),
                scheduler=scheduler,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
        else:
            pipe = CogVideoXImageToVideoPipelineTracking.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=unwrap_model(transformer),
                scheduler=scheduler,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()
        if args.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        # Run inference
        validation_outputs = []

        pipeline_args = {
            "validation_dir": args.validation_dir,
            "negative_prompt": "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
            "height": args.height,
            "width": args.width,
            "max_sequence_length": model_config.max_text_seq_length,
        }

        if args.tracking_column is not None:
            pipeline_args["tracking_map_path"] = args.tracking_map_path

        video = log_validation(
                accelerator=accelerator,
                pipe=pipe,
                vae=vae,
                dataset=train_dataset,
                args=args,
                pipeline_args=pipeline_args,
                epoch=0,
                is_final_validation=False,
            )
        validation_outputs.extend(video)

        accelerator.print("===== Memory after testing =====")
        print_memory(accelerator.device)
        reset_memory(accelerator.device)
        torch.cuda.synchronize(accelerator.device)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                videos=validation_outputs,
                base_model=args.pretrained_model_name_or_path,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
                fps=args.fps,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)


