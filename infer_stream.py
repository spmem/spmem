
import os
import sys
import math
import copy
import time
import argparse
import json

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import cv2
import imageio
from tqdm import tqdm
from einops import rearrange
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TSDF_ROOT = os.path.join(PROJECT_ROOT, "tsdf")
DA3_ROOT = os.path.join(PROJECT_ROOT, "Depth-Anything-3")
DA3_STREAM_ROOT = os.path.join(DA3_ROOT, "da3_streaming")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, TSDF_ROOT)
sys.path.insert(0, os.path.join(DA3_ROOT, "src"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── local imports ──────────────────────────────────────────────────────────
from diffusers import CogVideoXDPMScheduler, AutoencoderKLCogVideoX, CogVideoXDDIMScheduler
from diffusers.utils import export_to_video, load_video
from transformers import T5EncoderModel, T5Tokenizer
from models.cogvideox_tracking_ref import (
    CogVideoXImageToVideoPipelineTracking,
    CogVideoXTransformer3DModelTrackingRef,
)

try:
    import open3d as o3d
except ImportError:
    print("Warning: open3d not found.")

# TSDF + DA3 helpers (from existing demo_da3.py)
sys.path.insert(0, os.path.join(TSDF_ROOT))
from demo_da3 import (
    run_tsdf_fusion,
    render_pointcloud,
    render_tracking_maps_from_cache,
    save_ply,
    remove_outliers_iqr,
)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False
    print("Warning: Qwen2.5-VL not available; will use user-supplied prompts only.")


def _sample_points_in_frustum(K, fov_y_deg, aspect, near, far, rng):
    fov_y = np.deg2rad(fov_y_deg)
    tan_half = np.tan(fov_y / 2.0)
    z = rng.uniform(near, far, size=(K,))
    u = rng.uniform(-1.0, 1.0, size=(K,))
    v = rng.uniform(-1.0, 1.0, size=(K,))
    x = u * z * tan_half * aspect
    y = v * z * tan_half
    return np.stack([x, y, z], axis=-1)


def _auto_estimate_scene_scale(positions, percentile=60.0):
    from scipy.spatial.distance import pdist
    N = len(positions)
    if N < 2:
        return 1.0
    max_pairs = min(5000, N * (N - 1) // 2)
    if N <= 100:
        dists = pdist(positions, metric="euclidean")
    else:
        idx1 = np.random.randint(0, N, size=max_pairs)
        idx2 = np.random.randint(0, N, size=max_pairs)
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]
        dists = np.linalg.norm(positions[idx1] - positions[idx2], axis=1)
    if len(dists) == 0:
        return 1.0
    return max(np.percentile(dists, percentile), 1e-6)


def compute_frustum_overlap_scores(
    memory_c2w, query_c2w, fov_deg=90.0, aspect=1.5,
    num_samples=32, batch_size=200,
):

    M = memory_c2w.shape[0]
    T = query_c2w.shape[0]
    if M == 0 or T == 0:
        return np.zeros((M, T), dtype=np.float32)

    all_positions = np.concatenate([memory_c2w[:, :3, 3], query_c2w[:, :3, 3]], axis=0)
    scene_scale = _auto_estimate_scene_scale(all_positions)

    near = max(scene_scale * 0.02, 1e-4)
    far = max(scene_scale * 3.0, near * 2.0)

    rng = np.random.default_rng(1234)
    fov_y = np.deg2rad(fov_deg)
    tan_half = np.tan(fov_y / 2.0)
    eps = 1e-6

    pts_cam = _sample_points_in_frustum(num_samples, fov_deg, aspect, near, far, rng)

    query_world_pts = np.zeros((T, num_samples, 3), dtype=np.float32)
    for t in range(T):
        R_q = query_c2w[t, :3, :3].astype(np.float32)
        t_q = query_c2w[t, :3, 3].astype(np.float32)
        query_world_pts[t] = (R_q @ pts_cam.T).T + t_q

    mem_pos = memory_c2w[:, :3, 3].astype(np.float32)
    mem_R = memory_c2w[:, :3, :3].astype(np.float32)
    R_wc_T = np.transpose(mem_R, (0, 2, 1))

    scores = np.zeros((M, T), dtype=np.float32)

    for start in range(0, M, batch_size):
        end = min(M, start + batch_size)
        batch_R_wc_T = R_wc_T[start:end]
        batch_pos = mem_pos[start:end]

        pts_world_exp = query_world_pts[None, :, :, :]
        batch_pos_exp = batch_pos[:, None, None, :]

        p_rel = pts_world_exp - batch_pos_exp
        p_cam = np.einsum("bij,btkj->btki", batch_R_wc_T, p_rel)

        x, y, z = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
        z_in = (z >= near) & (z <= far) & (z > 0.0)
        x_in = np.abs(x) <= (z * tan_half * aspect + eps)
        y_in = np.abs(y) <= (z * tan_half + eps)

        inside = z_in & x_in & y_in
        scores[start:end, :] = inside.astype(np.float32).mean(axis=2)

    return np.clip(scores, 0.0, 1.0)


def select_keyframe_indices(
    memory_c2w, query_c2w, N=10, fov_deg=90.0, aspect=1.5,
    num_samples=32, overlap_threshold=0.3, force_count=False,
):

    M = memory_c2w.shape[0]
    N = min(N, M)
    if M == 0 or N == 0:
        return None

    scores = compute_frustum_overlap_scores(
        memory_c2w, query_c2w, fov_deg=fov_deg, aspect=aspect,
        num_samples=num_samples,
    )

    T = scores.shape[1]
    current_best = np.zeros(T, dtype=np.float32)
    selected = []

    for step in range(N):
        marginal_gain = np.sum(np.maximum(0.0, scores - current_best), axis=1)
        for idx in selected:
            marginal_gain[idx] = -1.0

        best_i = int(np.argmax(marginal_gain))
        if marginal_gain[best_i] <= 0 and step > 0:
            if not force_count:
                break
            single_score = np.max(scores, axis=1)
            for idx in selected:
                single_score[idx] = -1.0
            best_i = int(np.argmax(single_score))
            if single_score[best_i] < 0:
                break

        selected.append(best_i)
        np.maximum(current_best, scores[best_i], out=current_best)

    if not selected:
        return None

    if not force_count:
        selected_arr = np.array(selected)
        max_overlaps = np.max(scores[selected_arr], axis=1)
        keep_mask = max_overlaps >= overlap_threshold
        filtered = selected_arr[keep_mask].tolist()

        if not filtered:
            print(f"[KeyframeSelect] No frame meets overlap >= {overlap_threshold}, "
                  f"falling back to top-{N} by marginal gain")
            filtered = selected[:N]
    else:
        # force_count: skip threshold filter, keep all N
        filtered = selected

    print(f"[KeyframeSelect] Selected {len(filtered)}/{M} memory frames "
          f"(force={force_count}): {filtered}")
    return filtered


def print_gpu_memory():
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        cached = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[GPU {dev}] Total: {total:.2f} GB | Alloc: {alloc:.2f} GB | Cached: {cached:.2f} GB")


def read_video_to_tensor(video_path, height=480, width=720):
    from decord import VideoReader
    vr = VideoReader(video_path)
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    oh, ow = frames.shape[1], frames.shape[2]
    if oh != height or ow != width:
        resized = [cv2.resize(f, (width, height), interpolation=cv2.INTER_NEAREST) for f in frames]
        frames = np.array(resized)
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).contiguous().float() / 255.0
    t = rearrange(t, "b f c h w -> b c f h w")
    return t


def read_video_to_pil(video_path, height=480, width=720, max_frames=None):
    """Read video to list of PIL images."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        frames.append(Image.fromarray(frame))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def pil_list_to_tensor(pil_list, target_size=(480, 720)):
    tensors = []
    for img in pil_list:
        img_r = TF.resize(img, target_size, interpolation=Image.BILINEAR)
        tensors.append(TF.to_tensor(img_r))
    return torch.stack(tensors, dim=0)


def tensor_to_pil_list(tensor):
    imgs = []
    for i in range(tensor.shape[0]):
        arr = (tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        imgs.append(Image.fromarray(arr))
    return imgs


def resize_pil_list(pil_list, target_size):
    h, w = target_size
    return [img.resize((w, h)) for img in pil_list]


def video_to_frames_dir(video_path, frames_dir, target_size=None):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if target_size is not None:
            frame = cv2.resize(frame, (target_size[1], target_size[0]))
        cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:06d}.png"), frame)
        idx += 1
    cap.release()
    return idx


def video_to_numpy_rgb(video_path, height=None, width=None, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if height is not None and width is not None:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return np.array(frames, dtype=np.uint8)



def make_identity_c2w():
    return np.eye(4, dtype=np.float64)


def apply_translation(c2w, direction, speed):
    R_mat = c2w[:3, :3]
    local_delta = np.zeros(3, dtype=np.float64)
    if direction == "W":
        local_delta[2] = speed
    elif direction == "S":
        local_delta[2] = -speed
    elif direction == "A":
        local_delta[0] = -speed
    elif direction == "D":
        local_delta[0] = speed
    elif direction == "Reverse":
        local_delta[2] = speed
    world_delta = R_mat @ local_delta
    new_c2w = c2w.copy()
    new_c2w[:3, 3] += world_delta
    return new_c2w


def apply_rotation(c2w, direction, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    if direction == "LookUp":
        axis_local = np.array([1, 0, 0], dtype=np.float64)
    elif direction == "LookDown":
        axis_local = np.array([1, 0, 0], dtype=np.float64)
        angle_rad = -angle_rad

    elif direction == "LookLeft":
        axis_local = np.array([0, 1, 0], dtype=np.float64)
        angle_rad = -angle_rad
    elif direction == "LookRight":
        axis_local = np.array([0, 1, 0], dtype=np.float64)
    else:
        return c2w

    R_local = R.from_rotvec(axis_local * angle_rad).as_matrix()
    new_c2w = c2w.copy()
    new_c2w[:3, :3] = c2w[:3, :3] @ R_local
    return new_c2w


TRANSLATION_DIRS = {"W", "A", "S", "D", "Reverse"}
ROTATION_DIRS = {"LookUp", "LookDown", "LookLeft", "LookRight"}

KEY_MAP = {
    'w': "W", 'a': "A", 's': "S", 'd': "D", 'v': "Reverse",
    'i': "LookUp", 'j': "LookLeft", 'k': "LookDown", 'l': "LookRight",
}


def generate_camera_trajectory(base_c2w, actions, speed, angle, num_frames_total):

    poses = [base_c2w.copy()]
    current = base_c2w.copy()
    for direction, n_frames in actions:
        for _ in range(n_frames):
            if direction in TRANSLATION_DIRS:
                current = apply_translation(current, direction, speed)
            elif direction in ROTATION_DIRS:
                current = apply_rotation(current, direction, angle)
            poses.append(current.copy())
    while len(poses) < num_frames_total:
        poses.append(poses[-1].copy())
    poses = poses[:num_frames_total]
    return np.array(poses, dtype=np.float64)


def parse_action_string(action_str, default_frames=44):

    tokens = action_str.strip().lower().split()
    actions = []
    i = 0
    while i < len(tokens):
        key = tokens[i]
        direction = KEY_MAP.get(key)
        if direction is None:
            i += 1
            continue
        n = default_frames
        if i + 1 < len(tokens):
            try:
                n = int(tokens[i + 1])
                i += 2
            except ValueError:
                i += 1
        else:
            i += 1
        actions.append((direction, n))
    return actions



def run_da3_multiview(video_path, output_dir, da3_root=DA3_ROOT,
                       process_res=504, n_frames=49):

    frames_dir = os.path.join(output_dir, "da3_frames")
    if os.path.exists(frames_dir):
        import shutil
        shutil.rmtree(frames_dir)
    n_extracted = video_to_frames_dir(video_path, frames_dir)
    n_use = min(n_frames, n_extracted)
    print(f"[DA3] Extracted {n_extracted} frames, using {n_use}")

    frame_paths = sorted([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.endswith('.png')
    ])[:n_use]

    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=torch.device("cuda"))

    with torch.no_grad():
        prediction = model.inference(
            frame_paths,
            process_res=process_res,
            ref_view_strategy="saddle_balanced",
        )

    images = prediction.processed_images
    depths = np.squeeze(prediction.depth)
    intrinsics = prediction.intrinsics
    extrinsics_w2c = prediction.extrinsics

    N = images.shape[0]
    cam_c2w = np.zeros((N, 4, 4), dtype=np.float64)
    for i in range(N):
        w2c_4x4 = np.eye(4)
        w2c_4x4[:3, :4] = extrinsics_w2c[i]
        cam_c2w[i] = np.linalg.inv(w2c_4x4)

    del model
    torch.cuda.empty_cache()

    print(f"[DA3] Output: images={images.shape}, depths={depths.shape}")
    return images, depths, intrinsics, extrinsics_w2c, cam_c2w


def run_da3_streaming_depth(video_path, output_dir, cam_c2w_known,
                             da3_root=DA3_ROOT, process_res=504):

    frames_dir = os.path.join(output_dir, "da3_gen_frames")
    if os.path.exists(frames_dir):
        import shutil
        shutil.rmtree(frames_dir)
    n_extracted = video_to_frames_dir(video_path, frames_dir)

    frame_paths = sorted([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.endswith('.png')
    ])[:n_extracted]

    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=torch.device("cuda"))

    with torch.no_grad():
        prediction = model.inference(
            frame_paths,
            process_res=process_res,
            ref_view_strategy="saddle_balanced",
        )

    images = prediction.processed_images
    depths = np.squeeze(prediction.depth)
    intrinsics = prediction.intrinsics
    extrinsics_w2c = prediction.extrinsics

    N = images.shape[0]
    da3_cam_c2w = np.zeros((N, 4, 4), dtype=np.float64)
    for i in range(N):
        w2c_4x4 = np.eye(4)
        w2c_4x4[:3, :4] = extrinsics_w2c[i]
        da3_cam_c2w[i] = np.linalg.inv(w2c_4x4)

    del model
    torch.cuda.empty_cache()

    intrinsic = intrinsics[0] if len(intrinsics.shape) > 2 else intrinsics

    print(f"[DA3-Stream] depths range: {depths[depths>0].min():.3f} ~ {depths.max():.3f}")
    print(f"[DA3-Stream] DA3 cam positions span: {np.ptp(da3_cam_c2w[:,:3,3], axis=0)}")
    print(f"[DA3-Stream] Known cam positions span: {np.ptp(cam_c2w_known[:N,:3,3], axis=0)}")

    return depths, intrinsic, da3_cam_c2w, images


def backproject_depth_to_pointcloud(images, depths, intrinsic, cam_c2w,
                                     depth_percentile=98, sample_step=2):

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    all_valid = depths[depths > 0]
    depth_cap = np.percentile(all_valid, depth_percentile) if len(all_valid) > 0 else 1e6

    all_pts, all_clr = [], []
    N, H, W = depths.shape
    for i in range(N):
        d = depths[i].copy()
        d[d > depth_cap] = 0
        mask = d > 0

        v, u = np.where(mask)
        v, u = v[::sample_step], u[::sample_step]
        z = d[v, u]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
        pts_world = (cam_c2w[i] @ pts_cam.T).T[:, :3]

        colors = images[i][v, u].astype(np.float64) / 255.0
        all_pts.append(pts_world)
        all_clr.append(colors)

    return np.concatenate(all_pts, axis=0), np.concatenate(all_clr, axis=0)



class SpatialMemoryCache:


    def __init__(self, device="cuda"):
        self.device = device
        self.points = None
        self.colors = None
        self.intrinsic = None       # DA3-resolution intrinsic (3, 3)
        self.da3_h = None           # DA3 processing height
        self.da3_w = None           # DA3 processing width
        self.voxel_size = 0.02

    def _get_scaled_intrinsic(self, render_H, render_W):
        intr = self.intrinsic.copy()
        if self.da3_h is not None and self.da3_w is not None:
            scale_x = render_W / self.da3_w
            scale_y = render_H / self.da3_h
            intr[0, 0] *= scale_x   # fx
            intr[1, 1] *= scale_y   # fy
            intr[0, 2] *= scale_x   # cx
            intr[1, 2] *= scale_y   # cy
        return intr

    def build_from_da3(self, images, depths, intrinsic, cam_c2w, n_imgs=49,
                       save_dir=None):
        self.intrinsic = intrinsic.copy()
        self.da3_h = images.shape[1]
        self.da3_w = images.shape[2]
        n_imgs = min(n_imgs, images.shape[0])

        raw_pts, raw_clr = backproject_depth_to_pointcloud(
            images[:n_imgs], depths[:n_imgs], intrinsic, cam_c2w[:n_imgs])

        if save_dir is not None:
            save_ply(raw_pts, raw_clr, os.path.join(save_dir, "3d_cache_da3raw.ply"))
            print(f"[SpatialMemory] DA3 raw point cloud: {raw_pts.shape[0]} points -> 3d_cache_da3raw.ply")

        pc, voxel_size = run_tsdf_fusion(
            images, depths, intrinsic, cam_c2w, n_imgs=n_imgs
        )
        self.voxel_size = voxel_size
        self.points = pc[:, :3]
        self.colors = pc[:, 3:6] / 255.0

        if save_dir is not None:
            save_ply(self.points, self.colors,
                     os.path.join(save_dir, "3d_cache.ply"))
            print(f"[SpatialMemory] After TSDF: {self.points.shape[0]} points -> 3d_cache.ply")

        print(f"[SpatialMemory] Initial cache: {self.points.shape[0]} points")

    def rebuild_from_long_video(self, all_images, all_depths, all_cam_c2w,
                                intrinsic=None, save_dir=None, chunk_idx=0):

        if intrinsic is not None:
            self.intrinsic = intrinsic
            self.da3_h = all_images.shape[1]
            self.da3_w = all_images.shape[2]
        intr = self.intrinsic

        N_total = all_images.shape[0]
        print(f"[SpatialMemory] Rebuilding cache from {N_total} total frames...")

        raw_pts, raw_clr = backproject_depth_to_pointcloud(
            all_images, all_depths, intr, all_cam_c2w)

        if save_dir is not None:
            save_ply(raw_pts, raw_clr, os.path.join(save_dir, "3d_cache_da3raw.ply"))
            print(f"[SpatialMemory] DA3 raw point cloud: {raw_pts.shape[0]} points -> 3d_cache_da3raw.ply")

        pc, voxel_size = run_tsdf_fusion(
            all_images, all_depths, intr, all_cam_c2w, n_imgs=N_total
        )
        self.voxel_size = voxel_size
        self.points = pc[:, :3]
        self.colors = pc[:, 3:6] / 255.0

        if save_dir is not None:
            save_ply(self.points, self.colors,
                     os.path.join(save_dir, "3d_cache.ply"))

        print(f"[SpatialMemory] Rebuilt cache: {self.points.shape[0]} points (from {N_total} frames)")

    def _deduplicate(self, voxel_size=0.04):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        self.points = np.asarray(pcd_down.points)
        self.colors = np.asarray(pcd_down.colors)

    def render_tracking_maps(self, cam_c2w, H, W, radius=0.008):
        """Render the 3D cache from given camera trajectory.
        Scales DA3 intrinsics to match the render resolution (H, W).
        """
        render_intrinsic = self._get_scaled_intrinsic(H, W)
        return render_tracking_maps_from_cache(
            self.points, self.colors, render_intrinsic, cam_c2w, H, W, radius
        )

    def save(self, path):
        save_ply(self.points, self.colors, path)


def load_qwen(model_path="ckpt/Qwen2.5-VL-7B-Instruct"):
    if not HAS_QWEN:
        return None, None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def get_prompt_from_video(model, processor, video_path, device="cuda"):
    if model is None:
        return "A video of a scene with camera movement."
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 0.1},
            {"type": "text", "text": "Describe this video in 50 words."},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt", **video_kwargs,
    ).to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]



def build_pipeline(model_path="ckpt/spmem_ckpt",
                   checkpoint_path="ckpt/spmem_checkpoint",
                   device="cuda", dtype=torch.bfloat16):
    vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
    text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
    tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
    transformer = CogVideoXTransformer3DModelTrackingRef.from_pretrained_2d(
        checkpoint_path, subfolder="transformer"
    )
    transformer.requires_grad_(False)
    scheduler = CogVideoXDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

    pipe = CogVideoXImageToVideoPipelineTracking(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        transformer=transformer, scheduler=scheduler,
    )
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(device, dtype=dtype)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    pipe.transformer.gradient_checkpointing = False

    return pipe


def encode_tracking_maps(pipe, tracking_pil_list, height, width, device, dtype):
    tracking_tensor = pil_list_to_tensor(
        [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in tracking_pil_list],
        target_size=(height, width)
    ).float().to(device=device, dtype=dtype)

    tracking_maps = tracking_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
    tracking_latent_dist = pipe.vae.encode(tracking_maps).latent_dist
    tracking_maps = tracking_latent_dist.sample() * pipe.vae.config.scaling_factor
    tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)
    return tracking_maps, tracking_tensor


def generate_chunk(pipe, prompt, cond_frames_pil, tracking_pil_list,
                   reference_tensor, height, width, cond_frames=5,
                   num_frames=49, seed=25, device="cuda", dtype=torch.bfloat16,
                   max_ref_frames=10):

    tracking_maps, tracking_tensor = encode_tracking_maps(
        pipe, tracking_pil_list, height, width, device, dtype
    )
    tracking_cond = tracking_tensor[:cond_frames]
    ref_frames = reference_tensor.to(dtype).to(device)
    n_ref = min(max_ref_frames, ref_frames.shape[2])

    video_generate = pipe(
        prompt=prompt,
        negative_prompt=(
            "The video is not of a high quality, it has a low resolution. "
            "Watermark present in each frame. The background is solid. "
            "Strange body and strange trajectory. Distortion."
        ),
        video=cond_frames_pil,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=6.0,
        generator=torch.Generator().manual_seed(seed),
        tracking_maps=tracking_maps,
        tracking_video=tracking_cond,
        height=height,
        width=width,
        reference=ref_frames[:, :, -n_ref:, :, :],
    ).frames[0]

    return video_generate



def interactive_prompt(speed, angle, chunk_idx, total_frames, cache_points):
    """Display the interactive control panel and read user input."""
    info = (
        "\n"
        "+---------------------------------------------------------+\n"
        "|       Streaming Camera Control         |\n"
        "+---------------------------------------------------------+\n"
        "|  Movement:  W(fwd) A(left) S(back) D(right) V(reverse)  |\n"
        "|  Look:      I(up)  J(left) K(down) L(right)             |\n"
        "|  Combo:     e.g. 'w 20 a 10' (fwd 20fr, left 10fr)     |\n"
        "|  Settings:  T(speed) R(angle) C(info)                   |\n"
        "|  Control:   P(pause/noop) N(quit)                       |\n"
        "+---------------------------------------------------------+\n"
        f"|  Chunk: {chunk_idx:3d} | Frames: {total_frames:5d} | "
        f"Cache: {cache_points:8d} pts  |\n"
        f"|  Speed: {speed:.4f} | Angle: {angle:.2f} deg/frame"
        f"{'':>16s}|\n"
        "+---------------------------------------------------------+\n"
        "Enter action: "
    )
    return input(info).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Streaming interactive world-model video generation v1 "
                    "(video input + DA3 + TSDF + Spatial Memory)"
    )
    parser.add_argument("--input_video", type=str, required=True,
                        help="Input video.mp4 path")
    parser.add_argument("--output_dir", type=str, default="outputs_stream_v1")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (auto-generated via Qwen if not provided)")
    parser.add_argument("--model_path", type=str, default="ckpt/spmem_ckpt")
    parser.add_argument("--checkpoint_path", type=str, default="ckpt/spmem_ckpt")
    parser.add_argument("--qwen_path", type=str, default="ckpt/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--cond_frames", type=int, default=5,
                        help="Number of conditioning frames (must satisfy (cond_frames-1) %% 4 == 0)")
    parser.add_argument("--ref_frames", type=int, default=10,
                        help="Number of reference frames for appearance consistency")
    parser.add_argument("--num_frames", type=int, default=49,
                        help="Total frames per chunk")
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--speed", type=float, default=0.1,
                        help="Translation speed per frame")
    parser.add_argument("--angle", type=float, default=0.5,
                        help="Rotation degrees per frame")
    parser.add_argument("--da3_process_res", type=int, default=504,
                        help="DA3 processing resolution")
    parser.add_argument("--skip_da3", action="store_true",
                        help="Skip DA3; load from --da3_cache_dir instead")
    parser.add_argument("--da3_cache_dir", type=str, default=None,
                        help="Directory with cached DA3 output (da3_cache.npz)")
    parser.add_argument("--vis_dynamic", action="store_true",
                        help="Launch viser viewer for dynamic point clouds after DA3")
    parser.add_argument("--vis_static", action="store_true",
                        help="Launch viser viewer for static cache after TSDF")
    parser.add_argument("--vis_port", type=int, default=8890,
                        help="Port for viser visualization server")
    parser.add_argument("--force_ref_count", action="store_true",
                        help="Force exactly ref_frames reference frames per chunk "
                             "(disable early-stop and overlap threshold filtering)")
    parser.add_argument("--naive_ref", action="store_true",
                        help="Use naive reference selection (last ref_frames frames) "
                             "instead of frustum-overlap keyframe selection")
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    height, width = args.height, args.width
    cond_frames = args.cond_frames
    ref_frames = args.ref_frames
    num_frames = args.num_frames
    seed = args.seed
    speed = args.speed
    angle = args.angle
    new_frames_per_chunk = num_frames - cond_frames  # 44

    os.makedirs(args.output_dir, exist_ok=True)

    all_cam_c2w = []

    print("\n" + "=" * 60)
    print("  STEP 0: Depth & Pose Estimation (Depth-Anything-3)")
    print("=" * 60)

    da3_out_dir = os.path.join(args.output_dir, "da3_chunk0")

    if args.skip_da3 and args.da3_cache_dir:
        print(f"[INFO] Loading cached DA3 output from {args.da3_cache_dir}")
        da3_out_dir = args.da3_cache_dir
        cache_data = np.load(os.path.join(da3_out_dir, "da3_cache.npz"))
        images_da3 = cache_data["images"]
        depths_da3 = cache_data["depths"]
        intrinsics_da3 = cache_data["intrinsics"]
        cam_c2w_da3 = cache_data["cam_c2w"]
    else:
        os.makedirs(da3_out_dir, exist_ok=True)
        images_da3, depths_da3, intrinsics_da3, _, cam_c2w_da3 = run_da3_multiview(
            args.input_video, da3_out_dir,
            process_res=args.da3_process_res, n_frames=num_frames
        )
        np.savez(os.path.join(da3_out_dir, "da3_cache.npz"),
                 images=images_da3, depths=depths_da3,
                 intrinsics=intrinsics_da3, cam_c2w=cam_c2w_da3)

    intrinsic_canonical = intrinsics_da3[0] if len(intrinsics_da3.shape) > 2 else intrinsics_da3

    if args.vis_dynamic:
        print("[Vis] Launching dynamic point cloud viewer for input video...")
        from tsdf.vis_viser_pcd import visualize_dynamic_pcd
        viewer = visualize_dynamic_pcd(
            images_da3, depths_da3, intrinsic_canonical, cam_c2w_da3,
            port=args.vis_port, sample_step=4
        )
        print(f"[Vis] Dynamic PCD viewer running on port {args.vis_port}.")
        print("[Vis] Press Ctrl+C or close the browser to continue...")
        try:
            viewer.run()
        except KeyboardInterrupt:
            pass

    print("\n" + "=" * 60)
    print("  STEP 1: Building Initial 3D Spatial Memory Cache (TSDF)")
    print("=" * 60)

    spatial_cache = SpatialMemoryCache(device=device)
    spatial_cache.build_from_da3(
        images_da3, depths_da3, intrinsic_canonical, cam_c2w_da3,
        n_imgs=num_frames, save_dir=args.output_dir
    )

    last_input_idx = min(num_frames - 1, len(cam_c2w_da3) - 1)
    current_c2w = cam_c2w_da3[last_input_idx].copy()

    for i in range(min(num_frames, len(cam_c2w_da3))):
        all_cam_c2w.append(cam_c2w_da3[i].copy())

    if args.vis_static:
        cache_ply = os.path.join(args.output_dir, "3d_cache.ply")
        if os.path.exists(cache_ply):
            print("[Vis] Launching static cache viewer...")
            from tsdf.vis_viser_pcd import visualize_static_pcd
            viewer = visualize_static_pcd(cache_ply, port=args.vis_port)
            print(f"[Vis] Static cache viewer running on port {args.vis_port}.")
            print("[Vis] Press Ctrl+C or close the browser to continue...")
            try:
                viewer.run()
            except KeyboardInterrupt:
                pass

    print("\n" + "=" * 60)
    print("  STEP 2: Generating Text Prompt")
    print("=" * 60)

    if args.prompt:
        prompt = args.prompt
    else:
        qwen_model, qwen_processor = load_qwen(args.qwen_path)
        prompt = get_prompt_from_video(qwen_model, qwen_processor, args.input_video, device)
        del qwen_model, qwen_processor
        torch.cuda.empty_cache()
    print(f"[PROMPT] {prompt}")

    print("\n" + "=" * 60)
    print("  STEP 3: Loading CogVideoX Pipeline")
    print("=" * 60)

    pipe = build_pipeline(args.model_path, args.checkpoint_path, device, dtype)


    source_video_pil = read_video_to_pil(args.input_video, height, width, max_frames=num_frames)
    source_video_tensor = read_video_to_tensor(args.input_video, height, width).to(dtype).to(device)

    all_generated_frames = list(source_video_pil[:num_frames])
    ref_tensor = source_video_tensor

    chunk_idx = 0

    # ══════════════════════════════════════════════════════════════════════
    #  Interactive Generation Loop
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STREAMING GENERATION READY")
    print(f"  Input video: {args.input_video}")
    print(f"  Initial frames: {len(all_generated_frames)}")
    print(f"  3D cache points: {spatial_cache.points.shape[0]}")
    print("=" * 60)

    while True:
        cache_pts = spatial_cache.points.shape[0] if spatial_cache.points is not None else 0
        inp = interactive_prompt(speed, angle, chunk_idx, len(all_generated_frames), cache_pts).lower()

        if inp == 'n':
            print("[INFO] Stopping generation. Saving final outputs...")
            break

        elif inp == 'p':
            print("  [Paused] Enter any action to continue.")
            continue

        elif inp == 't':
            try:
                speed = float(input("  Enter new translation speed (e.g., 0.01): ").strip())
                print(f"  -> Speed updated: {speed}")
            except ValueError:
                print("  [!] Invalid input.")
            continue

        elif inp == 'r':
            try:
                angle = float(input("  Enter new rotation degrees (e.g., 2.0): ").strip())
                print(f"  -> Angle updated: {angle} deg/frame")
            except ValueError:
                print("  [!] Invalid input.")
            continue

        elif inp == 'c':
            print(f"\n{'='*55}")
            print(f"  Chunk: {chunk_idx} | Speed: {speed} | Angle: {angle}")
            print(f"  Frames/chunk: {num_frames} | Cond: {cond_frames} | New: {new_frames_per_chunk}")
            print(f"  Total frames: {len(all_generated_frames)}")
            print(f"  Cache points: {cache_pts}")
            print(f"  Camera pos: {current_c2w[:3, 3]}")
            print(f"  Prompt: {prompt[:80]}...")
            print(f"{'='*55}\n")
            continue

        actions = parse_action_string(inp, default_frames=new_frames_per_chunk)
        if not actions:
            print("  [!] Unknown command. Use W/A/S/D/I/J/K/L/V, or combos like 'w 20 a 10'.")
            continue

        total_action_frames = sum(n for _, n in actions)
        direction_str = " + ".join(f"{d}({n}f)" for d, n in actions)
        print(f"\n  -> Chunk {chunk_idx}: {direction_str} = {total_action_frames} action frames")

        cam_c2w_chunk = generate_camera_trajectory(
            current_c2w, actions, speed, angle, num_frames_total=num_frames
        )

        print("  [Render] Rendering tracking maps from 3D cache...")
        rendered_frames, rendered_masks = spatial_cache.render_tracking_maps(
            cam_c2w_chunk, height, width
        )

        tracking_pil_list = []
        for fr, mk in zip(rendered_frames, rendered_masks):
            masked = (mk[..., None] * fr.astype(float)).astype(np.uint8)
            tracking_pil_list.append(Image.fromarray(masked))

        cond_pil = all_generated_frames[-cond_frames:]

        full_tracking = list(cond_pil) + tracking_pil_list[cond_frames:num_frames]
        full_tracking = full_tracking[:num_frames]

        if args.naive_ref:
            keyframe_indices = None
            ref_pil = all_generated_frames[-ref_frames:]
            print(f"  [Ref] Naive mode: using last {len(ref_pil)} frames")
        else:
            memory_c2w_arr = np.array(all_cam_c2w, dtype=np.float64)
            keyframe_indices = select_keyframe_indices(
                memory_c2w_arr, cam_c2w_chunk,
                N=ref_frames, fov_deg=90.0, aspect=width / height,
                num_samples=32, overlap_threshold=0.3,
                force_count=args.force_ref_count,
            )

            if keyframe_indices is not None and len(keyframe_indices) > 0:
                ref_pil = [all_generated_frames[i] for i in keyframe_indices]
                print(f"  [Ref] Using {len(ref_pil)} keyframes by frustum overlap: {keyframe_indices}")
            else:
                ref_pil = all_generated_frames[-ref_frames:]
                print(f"  [Ref] Frustum selection returned nothing, falling back to last {ref_frames} frames")

        ref_t = pil_list_to_tensor(ref_pil, target_size=(height, width))
        ref_t = ref_t.unsqueeze(0).permute(0, 2, 1, 3, 4).to(dtype).to(device)

        print("  [Generate] Running CogVideoX inference...")
        start_t = time.time()

        video_gen = generate_chunk(
            pipe=pipe, prompt=prompt,
            cond_frames_pil=cond_pil,
            tracking_pil_list=full_tracking,
            reference_tensor=ref_t,
            height=height, width=width,
            cond_frames=cond_frames, num_frames=num_frames,
            seed=seed, device=device, dtype=dtype,
            max_ref_frames=ref_frames,
        )

        elapsed = time.time() - start_t
        print(f"  [Generate] Done in {elapsed / 60:.1f} min")

        chunk_dir = os.path.join(args.output_dir, f"chunk_{chunk_idx:03d}")
        os.makedirs(chunk_dir, exist_ok=True)

        export_to_video(
            resize_pil_list(video_gen, (height, width)),
            os.path.join(chunk_dir, "generated.mp4"), fps=20
        )
        export_to_video(
            resize_pil_list(tracking_pil_list[:num_frames], (height, width)),
            os.path.join(chunk_dir, "tracking_map.mp4"), fps=20
        )
        export_to_video(
            resize_pil_list(full_tracking, (height, width)),
            os.path.join(chunk_dir, "full_tracking.mp4"), fps=20
        )

        ref_info = {
            "keyframe_indices": keyframe_indices if keyframe_indices is not None else [],
            "num_memory_frames": len(all_cam_c2w),
            "fallback": keyframe_indices is None or len(keyframe_indices) == 0,
        }
        with open(os.path.join(chunk_dir, "ref_selection.json"), "w") as f:
            json.dump(ref_info, f, indent=2)

        ref_img_dir = os.path.join(chunk_dir, "ref_frames")
        os.makedirs(ref_img_dir, exist_ok=True)
        for rank, pil_img in enumerate(ref_pil):
            idx_label = keyframe_indices[rank] if (keyframe_indices and rank < len(keyframe_indices)) else f"last{rank}"
            pil_img.save(os.path.join(ref_img_dir, f"ref_{rank:02d}_frame{idx_label}.png"))

        new_gen_frames = video_gen[cond_frames:]
        all_generated_frames.extend(new_gen_frames)

        export_to_video(
            resize_pil_list(all_generated_frames, (height, width)),
            os.path.join(args.output_dir, "long_video.mp4"), fps=20
        )


        for ci in range(cond_frames, num_frames):
            if ci < len(cam_c2w_chunk):
                all_cam_c2w.append(cam_c2w_chunk[ci].copy())
        current_c2w = cam_c2w_chunk[-1].copy()


        print("  [Update] Running DA3 on long video for full cache rebuild...")

        try:
            long_video_path = os.path.join(args.output_dir, "long_video.mp4")

            long_da3_dir = os.path.join(args.output_dir, f"da3_long_chunk{chunk_idx:03d}")
            os.makedirs(long_da3_dir, exist_ok=True)

            long_images, long_depths, long_intrinsics, _, long_cam_c2w_da3 = run_da3_multiview(
                long_video_path, long_da3_dir,
                process_res=args.da3_process_res,
                n_frames=len(all_generated_frames),
            )

            long_intrinsic = long_intrinsics[0] if len(long_intrinsics.shape) > 2 else long_intrinsics


            n_da3 = long_images.shape[0]
            n_user = len(all_cam_c2w)
            n_use = min(n_da3, n_user)

            print(f"  [Update] DA3 returned {n_da3} frames, user poses: {n_user}, using: {n_use}")


            user_cam_for_tsdf = np.array(all_cam_c2w[:n_use], dtype=np.float64)

            np.savez(os.path.join(long_da3_dir, "da3_long_data.npz"),
                     images=long_images[:n_use],
                     depths=long_depths[:n_use],
                     intrinsic=long_intrinsic,
                     cam_c2w_da3=long_cam_c2w_da3[:n_use],
                     cam_c2w_user=user_cam_for_tsdf)

            print("  [Update] Rebuilding 3D spatial memory cache from full long video...")
            spatial_cache.rebuild_from_long_video(
                long_images[:n_use],
                long_depths[:n_use],
                long_cam_c2w_da3[:n_use],
                intrinsic=long_intrinsic,
                save_dir=args.output_dir,
                chunk_idx=chunk_idx,
            )


            current_c2w = long_cam_c2w_da3[n_use - 1].copy()

        except Exception as e:
            print(f"  [!] 3D cache rebuild failed: {e}")
            import traceback
            traceback.print_exc()
            print("      Continuing with existing cache...")

        chunk_idx += 1
        print(f"\n  [OK] Chunk {chunk_idx - 1} complete. Total frames: {len(all_generated_frames)}")

    print("\n" + "=" * 60)
    print("  Saving Final Outputs")
    print("=" * 60)

    export_to_video(
        resize_pil_list(all_generated_frames, (height, width)),
        os.path.join(args.output_dir, "final_long_video.mp4"), fps=20
    )
    if spatial_cache.points is not None:
        spatial_cache.save(os.path.join(args.output_dir, "3d_cache.ply"))

    # Save all camera poses
    all_cam_c2w_arr = np.array(all_cam_c2w)
    np.save(os.path.join(args.output_dir, "all_cam_c2w.npy"), all_cam_c2w_arr)

    # Save metadata
    metadata = {
        "total_chunks": chunk_idx,
        "total_frames": len(all_generated_frames),
        "prompt": prompt,
        "speed": speed,
        "angle": angle,
        "cond_frames": cond_frames,
        "ref_frames": ref_frames,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "seed": seed,
        "input_video": args.input_video,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[DONE] Total chunks: {chunk_idx}")
    print(f"[DONE] Total frames: {len(all_generated_frames)}")
    print(f"[DONE] Output dir: {args.output_dir}")
    print(f"[DONE] Final video: {os.path.join(args.output_dir, 'final_long_video.mp4')}")
    print(f"[DONE] 3D cache: {os.path.join(args.output_dir, '3d_cache.ply')}")


if __name__ == "__main__":
    main()
