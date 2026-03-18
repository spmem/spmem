
import time
import open3d as o3d
import argparse
import os
import cv2
import numpy as np
from PIL import Image
import imageio
import torch
import shutil
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.structures import Pointclouds
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


def _ensure_nvcc_on_path() -> str | None:
    """
    PyCUDA invokes `nvcc` by command name. If CUDA is installed but PATH
    doesn't include the cuda bin dir, add it so PyCUDA can compile kernels.
    """
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    candidates = (
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.4/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11.8/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
    )
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            os.environ["PATH"] = os.path.dirname(p) + os.pathsep + os.environ.get("PATH", "")
            return p
    return None


_NVCC_PATH = _ensure_nvcc_on_path()

import fusion

RUN_DATA_VERSION = "2026-03-18-nvccpath-maxvoldim"

def depth_to_point_cloud(intrinsic, depth, c2w):

    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    uv = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)  # (H*W, 3)

    K_inv = np.linalg.inv(intrinsic)
    points_cam = (K_inv @ uv.T).T  # (H*W, 3)
    points_cam *= depth.reshape(-1, 1)  

    ones = np.ones((points_cam.shape[0], 1))
    points_cam_homo = np.hstack([points_cam, ones])  # (H*W, 4)
    points_world = (c2w @ points_cam_homo.T).T[:, :3]  # (H*W, 3)

    return points_world

def save_ply(points, colors, filename):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)




def get_random_revisit():
    numbers = list(range(48))


    weights = []
    for i in numbers:
        if i <= 10:       # 0-10: 权重较高（斜率较平缓）
            weights.append(50 - i * 2)
        elif i <= 25:     # 11-25: 权重适中（斜率中等）
            weights.append(30 - (i - 10) * 1.5)
        else:             # 21-47: 权重很低（斜率陡峭）
            weights.append(15 - (i - 20) * 0.5)

    selected = random.choices(numbers, weights=weights, k=1)[0]
    return selected

def load_pointcloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)  # (P, 3)
    
    if pcd.has_colors():
        colors = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32)  # (P, 3)
    else:
        colors = torch.ones_like(points) 
    return points, colors

def convert_extrinsics(cam_c2w):
    N = cam_c2w.shape[0]


    cam_c2w[:, :,0] = -cam_c2w[:, :,0]
    cam_c2w[:, :,1] = -cam_c2w[:, :,1]

    R = cam_c2w[:, :3, :3]  
    
    w2c = cam_c2w.inverse()
    T = w2c[:, :3, -1]


    return R, T

def convert_intrinsics(intrinsic, image_size):
    """将 3x3 内参矩阵转换为 PyTorch3D 的归一化 focal_length 和 principal_point"""
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    focal_length = torch.tensor([[fx / image_size, fy / image_size]], dtype=torch.float32)
    principal_point = torch.tensor([[cx / image_size, cy / image_size]], dtype=torch.float32)

    return focal_length, principal_point

def render_pointcloud(pointcloud_path, intrinsic, cam_c2w, H, W, image_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cam_c2w = torch.tensor(cam_c2w, dtype=torch.float32) 

    points, colors = load_pointcloud(pointcloud_path)
    points, colors = points.to(device), colors.to(device)
    
    N = cam_c2w.shape[0]  # Number of cameras
    R, T = convert_extrinsics(cam_c2w)

    # Prepare intrinsic matrix template (reuse for all views)
    K_template = np.zeros((4, 4))
    K_template[0, 0] = intrinsic[0, 0]
    K_template[1, 1] = intrinsic[1, 1]
    K_template[0, 2] = W / 2  # intrinsic[0,2]
    K_template[1, 2] = H / 2  # intrinsic[1,2]
    K_template[2, 3] = 1
    K_template[3, 2] = 1
    K_template = torch.from_numpy(K_template).float().to(device)

    raster_settings = PointsRasterizationSettings(
        bin_size=0,
        image_size=(H, W),
        radius=0.008, 
        points_per_pixel=10
    )

    # Create a single point cloud and render per-camera (lower peak memory)
    point_cloud = Pointclouds(
        points=points.unsqueeze(0),
        features=colors.unsqueeze(0),
    )

    frames_list = []
    masks_list = []
    for i in range(N):
        K = K_template.unsqueeze(0)
        cameras = PerspectiveCameras(
            K=K,
            R=R[i:i+1],
            T=T[i:i+1],
            in_ndc=False,
            image_size=((H, W),),
            device=device,
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

        frame = renderer(point_cloud)  # (1, H, W, 4)
        fragments = rasterizer(point_cloud)
        mask = (fragments.idx[..., 0] >= 0).float()

        frames_list.append(frame.squeeze(0))
        masks_list.append(mask.squeeze(0))

    frames = torch.stack(frames_list, dim=0)
    masks = torch.stack(masks_list, dim=0)
    return frames, masks

def calculate_optimal_voxel_size(depth, height, width, k=10):
    depth_min = np.min(depth[depth > 0]) 
    depth_max = np.max(depth)
    scene_scale = depth_max - depth_min

    pixel_density = min(width, height) / scene_scale
    k = k  # 可以根据实验结果调整
    voxel_size = scene_scale / (k * pixel_density)
    # 限制 voxel_size 的范围，避免过小或过大
    voxel_size = max(0.02, min(voxel_size, 0.02))  # 通常在 1mm 到 10mm 之间
    return voxel_size


def depth_cap_percentile(depths, percentile=98.0):
    """Cap extreme depths to shrink TSDF bounds (set capped values to 0)."""
    valid = depths[depths > 0]
    if valid.size == 0:
        return depths, None
    cap = np.percentile(valid, percentile)
    out = depths.copy()
    out[out > cap] = 0
    return out, float(cap)


def remove_outliers_iqr(depth_map, k=1.5):
    """Remove per-frame depth outliers using IQR rule (set to 0)."""
    depths = depth_map.flatten()
    depths = depths[depths > 0]
    if depths.size == 0:
        return depth_map
    q1 = np.percentile(depths, 25)
    q3 = np.percentile(depths, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    outlier_mask = (depth_map < lower) | (depth_map > upper)
    depth_map[outlier_mask] = 0
    return depth_map
    
if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_dir", type=str, default="")
  parser.add_argument("--name", type=str, default="")
  parser.add_argument("--n_imgs", type=int, default=None)
  parser.add_argument("--voxel_size", type=float, default=0.02, help="TSDF voxel size (meters).")
  parser.add_argument("--max_vol_dim", type=int, default=1200, help="TSDF max volume dimension (legacy behavior uses 1200).")
  parser.add_argument("--depth_percentile", type=float, default=None, help="Cap depth above this percentile to shrink TSDF bounds (default: disabled).")
  parser.add_argument("--depth_max", type=float, default=None, help="Clamp depth >= this value to 0 (disabled by default).")
  parser.add_argument("--iqr_filter", action="store_true", help="Apply per-frame IQR outlier removal when estimating bounds (recommended).")
  parser.add_argument("--iqr_k", type=float, default=1.5, help="IQR outlier threshold k (default 1.5).")
  parser.add_argument("--fast_mode", action="store_true", help="Enable faster TSDF (depth cap + auto max_vol_dim to avoid voxel explosion).")
  parser.add_argument("--save_dir", type=str, default="outputs")
  parser.add_argument("--random_select_target", action="store_true", help="help for revisit")

  args = parser.parse_args()
  save_dir = args.save_dir
  
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  # TSDFVolume uses PyCUDA for GPU mode and requires `nvcc` at runtime to compile kernels.
  # If `nvcc` is not available, fall back to CPU mode to avoid PyCUDA compilation errors.
  use_gpu = (DEVICE == 'cuda') and (_NVCC_PATH is not None)
  
  print("===> run_data version:", RUN_DATA_VERSION)
  print("===> use_gpu:", use_gpu)
  if DEVICE == "cuda":
    print("===> nvcc:", _NVCC_PATH)
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  
  print("Estimating voxel volume bounds...")
  
  input_dir = args.input_dir
  name = args.name
  results = np.load(input_dir)
  images = results['images']
  depths = results['depths']
  intrinsic = results['intrinsic'] #(3, 3)
  cam_c2w = results['cam_c2w'] #(97, 4, 4)
  os.makedirs(f"{save_dir}/{name}", exist_ok=True)

  depth_min = depths.min()
  depth_max = depths.max()
  n_imgs, height, width = images.shape[0], images.shape[1], images.shape[2]
  
  
  scale = 1
  cam_intr = intrinsic
  vol_bnds = np.zeros((3,2))
  
  if args.random_select_target:
    n_imgs = 49 + get_random_revisit()
  
  if args.n_imgs is not None:
    n_imgs = args.n_imgs
  
  voxel_size = float(args.voxel_size)
  depth_percentile = args.depth_percentile
  if args.fast_mode and depth_percentile is None:
    depth_percentile = 98.0
  if depth_percentile is not None:
    depths, depth_cap = depth_cap_percentile(depths, percentile=float(depth_percentile))
    if depth_cap is not None:
      print(f"==> depth capped at p{float(depth_percentile)}: {depth_cap:.4f}")
  print("==> depth value range:", depths.min(), depths.max())
  print("==> num of images:", n_imgs)
  print(f"==> image resolution: {height}*{width}")
  print("==> voxel_size:", voxel_size)
  
  for i in range(n_imgs):
    # Read depth image and camera pose
    # depth_im = cv2.imread(f"{input_dir}/depth/{i:06d}.npy").astype(float)
    depth_im = depths[i].astype(float)
    if args.depth_max is not None:
      depth_im[depth_im >= float(args.depth_max)] = 0
    if args.iqr_filter:
      depth_im = remove_outliers_iqr(depth_im.copy(), k=float(args.iqr_k))
    depth_im /= scale  # depth is saved in 16-bit PNG in millimeters
    # print(f"===={i}===== depth max:{depth_im.max()}")
    cam_pose = cam_c2w[i]  # 4x4 rigid transformation matrix
    
    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  
  
  print("vol_bnds:", vol_bnds)
  print("Initializing voxel volume...")
  # Legacy behavior: use user-provided max_vol_dim (default 1200). This may shrink voxel_size
  # inside fusion.TSDFVolume and produce a very dense grid (slow on CPU but higher detail).
  # Fast mode: override max_vol_dim to avoid unintended voxel_size shrinking.
  max_vol_dim = int(args.max_vol_dim)
  if args.fast_mode:
    vol_dim_pre = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).astype(int)
    max_vol_dim_pre = int(max(vol_dim_pre))
    print("==> pre vol_dim:", vol_dim_pre, "max:", max_vol_dim_pre)
    max_vol_dim = max_vol_dim_pre
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size, max_vol_dim=max_vol_dim, use_gpu=use_gpu)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  for i in range(n_imgs):

    color_image = images[i]
    depth_im = depths[i].astype(float)
    depth_im /= scale  # depth is saved in 16-bit PNG in millimeters
    # print(f"===={i}===== depth max:{depth_im.max()}")
    cam_pose = cam_c2w[i]  # 4x4 rigid transformation matrix
    img_shape = color_image.shape

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  # Default: render point cloud (render_ply)
  point_cloud = tsdf_vol.get_point_cloud()
  print("done get points cloud")
  fusion.pcwrite(f"{save_dir}/{name}/pc.ply", point_cloud)
  print("Saving point cloud to pc.ply...")
  frames, masks = render_pointcloud(f"{save_dir}/{name}/pc.ply", intrinsic, cam_c2w, height, width)


  original_frames_list = [images[i] for i in range(images.shape[0])]  
  render_frames_list = [(frames[i]*255).cpu().numpy().astype(np.uint8) for i in range(frames.shape[0])]
  mask_frames_list = [(masks[i]*255).cpu().numpy().astype(np.uint8) for i in range(masks.shape[0])]
  masked_frames_list = [(masks[i][...,None]*frames[i]*255).cpu().numpy().astype(np.uint8) for i in range(masks.shape[0])]
  
  imageio.mimwrite(f"{save_dir}/{name}/Vid.mp4", original_frames_list, fps=20, quality=10)
  imageio.mimwrite(f"{save_dir}/{name}/Vid_mask.mp4", mask_frames_list[48:], fps=20, quality=10)
  imageio.mimwrite(f"{save_dir}/{name}/Vid_masktarget.mp4", masked_frames_list[48:], fps=20, quality=10)

