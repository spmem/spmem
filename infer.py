



"""
Generates a video based on the given prompt and saves it to the specified path.

Parameters:
- prompt (str): The description of the video to be generated.
- model_path (str): The path of the pre-trained model to be used.
- tracking_tensor (torch.Tensor): Tracking video tensor [T, C, H, W] in range [0,1]
- image_tensor (torch.Tensor): Input image tensor [C, H, W] in range [0,1]
- output_path (str): The path where the generated video will be saved.
- num_inference_steps (int): Number of steps for the inference process.
- guidance_scale (float): The scale for classifier-free guidance.
- num_videos_per_prompt (int): Number of videos to generate per prompt.
- dtype (torch.dtype): The data type for computation.
- seed (int): The seed for reproducibility.
"""
import os
import sys
import math
import copy
from tqdm import tqdm
from PIL import Image, ImageDraw
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.path.append(os.path.join(project_root, "submodules/MoGe"))
    sys.path.append(os.path.join(project_root, "submodules/vggt"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
except:
    print("Warning: MoGe not found, motion transfer will not be applied")
    
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from diffusers import FluxControlPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video, load_image, load_video
import imageio
import argparse
import time

from models.cogvideox_tracking_ref import CogVideoXImageToVideoPipelineTracking
import torchvision.transforms.functional as TF

from diffusers.utils import export_to_video, load_image,load_video
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler
from models.cogvideox_tracking_ref import CogVideoXTransformer3DModelTrackingRef
import cv2
from decord import VideoReader, cpu
from einops import rearrange


from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def read_video(video_path, height=480, width=832):
    video_reader = VideoReader(video_path)
    length = len(video_reader)
    numbers = [i for i in range(length)]
    frames = video_reader.get_batch(numbers) 
    pixel_values = frames.asnumpy()

    orig_h, orig_w = pixel_values.shape[1], pixel_values.shape[2]
    if orig_h == height and orig_w == width:
        pass
    else:
        resized_frames = []
        for i in range(len(pixel_values)):
            frame = pixel_values[i]
            resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
            resized_frames.append(resized_frame)
        pixel_values = np.array(resized_frames)


    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    pixel_values = pixel_values / 255.
    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
    return pixel_values


def resize_videos(video_generate,target_size):
    target_h, target_w = target_size
    video_new = []
    
    for frame in video_generate:
        frame = frame.resize((target_w, target_h))
        video_new.append(frame)
    
    return video_new

from PIL import Image
import torch

def concatenate_videos(video_target, video_mt, video_generate, target_size):
    assert len(video_target) == len(video_mt) == len(video_generate), "All videos must have the same number of frames"
    
    target_h, target_w = target_size
    video_viz = []
    
    for frame_t, frame_m, frame_g in zip(video_target, video_mt, video_generate):
        frame_t = frame_t.resize((target_w, target_h))
        frame_m = frame_m.resize((target_w, target_h))
        frame_g = frame_g.resize((target_w, target_h))
        
        new_width = target_w * 3
        new_height = target_h
        
        new_image = Image.new('RGB', (new_width, new_height))
        
        new_image.paste(frame_t, (0, 0))
        new_image.paste(frame_m, (target_w, 0))
        new_image.paste(frame_g, (target_w * 2, 0))
        
        video_viz.append(new_image)
    
    return video_viz

def pil_list_to_tensor(video_mt, target_size=(480, 720)):
    if not isinstance(video_mt, list):
        raise TypeError("Input must be a list of PIL images")

    tensor_frames = []
    for pil_img in video_mt:
        if not isinstance(pil_img, Image.Image):
            raise TypeError("List elements must be PIL.Image objects")
        
        img_resized = TF.resize(pil_img, target_size, interpolation=Image.BILINEAR)
        img_tensor = TF.to_tensor(img_resized)  
        
        tensor_frames.append(img_tensor)
    
    result = torch.stack(tensor_frames, dim=0)  # [N, C, H, W]
    if result.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, but got {result.shape[1]} channels")
    
    return result


def load_qwen(model_path="ckpt/Qwen2.5-VL-7B-Instruct"):
    
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    qwen_processor = AutoProcessor.from_pretrained(model_path)

    return qwen_model, qwen_processor

def get_prompt(model, processor, video_path, device):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 0.1,
                },
                {"type": "text", "text": "Describe this video in 50 words."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    prompt = output_text[0]
    return prompt


parser = argparse.ArgumentParser()
parser.add_argument("--val_name", type=str, default="000000000000.4_001")
parser.add_argument("--input_dir", type=str, default=None)
parser.add_argument("--prompt", type=str, default='A person riding a motorcycle in a video game.')
parser.add_argument("--save_dir", type=str, default="outputs_infer_worldmodel")
parser.add_argument("--use_cond", action="store_true")
parser.add_argument("--cond_frames", type=int, default=5)
parser.add_argument("--seed", type=int, default=25)
parser.add_argument("--strength", type=float, default=1.0)

parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=5)


args = parser.parse_args()



val_name_list = []
val_name = args.val_name
prompt = args.prompt
strength=args.strength
cond_frames = args.cond_frames ##(13,25,37) must (cond_frames-1) % 4 ==0
input_dir = args.input_dir


height = 480 
width = 720 
seed = args.seed
device = "cuda"
dtype = torch.bfloat16

model_path = "ckpt/spmem_ckpt"
qwen_model, qwen_processor = load_qwen(model_path="ckpt/Qwen2.5-VL-7B-Instruct")

if input_dir is not None:
    for item in sorted(os.listdir(input_dir)):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path): 
            val_name_list.append(item)  
else:
    val_name_dir = val_name
    val_name = val_name_dir.split("/")[-1]
    input_dir=os.path.dirname(val_name_dir)
    val_name_list.append(val_name)

val_name_list = sorted(val_name_list)  
val_name_list = val_name_list[args.start:args.end]




for val_name in val_name_list:
    outputs_dir = f"{args.save_dir}/{val_name}"

    try:
        if os.path.exists(os.path.join(outputs_dir, "result.mp4")):
            continue
        
        video_mt = load_video(f"{input_dir}/{val_name}/Vid_masktarget.mp4")
        video_target = load_video(f"{input_dir}/{val_name}/Vid_target.mp4")
        video_source = load_video(f"{input_dir}/{val_name}/Vid_source.mp4")
        video_source_tensor = read_video(f"{input_dir}/{val_name}/Vid_source.mp4").to(dtype).to(device)
        video_target_tensor = read_video(f"{input_dir}/{val_name}/Vid_target.mp4")

        video_mt_pil = copy.deepcopy(video_mt)  
        os.makedirs(outputs_dir, exist_ok=True)


        num_frames = len(video_mt)

        if args.use_cond:
            video_mt_raw = copy.deepcopy(video_mt) 
            needgen_frames = num_frames - cond_frames
            video_target_pil = video_source[-cond_frames:] + video_target[1:needgen_frames+1]
            video_mt_pil = video_source[-cond_frames:] + video_mt_raw[1:needgen_frames+1]
                

        video_mt = pil_list_to_tensor(video_mt_pil, target_size=(height, width))
        video_target = pil_list_to_tensor(video_target_pil, target_size=(height, width))

        
        prompt = get_prompt(qwen_model, qwen_processor, f"{input_dir}/{val_name}/Vid_target.mp4", device)        
        print("[INFO] QWEN PROPMT:", prompt)
            
        del qwen_model, qwen_processor
        
            
            
        vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
        text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
        transformer = CogVideoXTransformer3DModelTrackingRef.from_pretrained_2d("ckpt/spmem_ckpt", subfolder="transformer")
        transformer.requires_grad_(False)

    
        # transformer.is_train_cross = False

        scheduler = CogVideoXDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

        pipe = CogVideoXImageToVideoPipelineTracking(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler
        )


        ### preprocess ###
        image_tensor = video_target[0].clone()
        tracking_tensor = video_mt.clone()
        video = video_source[-cond_frames:]   ##add

        # Convert tensor to PIL Image
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        height, width = image.height, image.width

        pipe.transformer.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.vae.requires_grad_(False)

        # Process tracking tensor
        tracking_maps = video_mt.float() # [T, C, H, W]
        tracking_maps = tracking_maps.to(device=device, dtype=torch.bfloat16)
        tracking_cond_frames = tracking_maps[:cond_frames]  # Get first frame as [f, C, H, W]

        height, width = tracking_cond_frames.shape[2], tracking_cond_frames.shape[3]

        # 2. Set Scheduler.
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

        pipe.to(device, dtype=torch.bfloat16)
        # pipe.enable_sequential_cpu_offload()

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.transformer.eval()
        pipe.text_encoder.eval()
        pipe.vae.eval()

        pipe.transformer.gradient_checkpointing = False

        print("Encoding tracking maps")
        tracking_maps = tracking_maps.unsqueeze(0) # [B, T, C, H, W]
        tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        tracking_latent_dist = pipe.vae.encode(tracking_maps).latent_dist
        tracking_maps = tracking_latent_dist.sample() * pipe.vae.config.scaling_factor
        tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]



        #### process reference condition ####
        reference_frames = video_source_tensor.to(dtype).to(device)

        start_time = time.time()

        # 4. Generate the video frames based on the prompt.
        video_generate = pipe(
            prompt=prompt,
            negative_prompt="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
            video=video,                         ## * ##
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            use_dynamic_cfg=True,
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(seed),
            tracking_maps=tracking_maps,          ## * ##
            tracking_video=tracking_cond_frames,  ## * ##
            height=height,
            width=width,
            reference=reference_frames[:, :, -10:, :, :],                   ## * ##
        ).frames[0]

        end_time = time.time()

        total_seconds = end_time - start_time

        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60

        print(f"耗时: {minutes}分 {seconds:.2f}秒")

        video_viz = concatenate_videos(video_target_pil, video_mt_pil, video_generate, (480, 832))


        export_to_video(resize_videos(video_generate,(480, 832)), f"{outputs_dir}/result.mp4", fps=20)
        export_to_video(video_viz, f"{outputs_dir}/viz.mp4", fps=20)
        
    except Exception as e:
        print("run error, skip:", e)

        
