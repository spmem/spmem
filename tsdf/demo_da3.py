"""
demo_da3.py — TSDF Fusion + Dynamic/Static Separation using Depth-Anything-3 output.

Supports two modes:
  1. Chunk 0 (initial): Build TSDF from DA3 multi-view depth of the input video
     - Input: video frames + DA3 depth + DA3 camera poses
     - Output: static point cloud (pc.ply) + rendered tracking maps (Vid_masktarget.mp4)

  2. Chunk >= 1 (streaming update): Build TSDF from DA3 streaming depth of generated video
     - Input: per-frame depth + known camera poses (from user trajectory)
     - Output: new static point cloud for merging into spatial memory cache
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np
import torch
import imageio
import open3d as o3d
from PIL import Image
from tqdm import tqdm

from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.structures import Pointclouds

TSDF_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TSDF_ROOT)
import fusion


# ─── Helpers ──────────────────────────────────────────────────────────────

def remove_outliers_iqr(depth_map, k=1.5):
    depths = depth_map.flatten()
    depths = depths[depths > 0]
    if len(depths) == 0:
        return depth_map
    Q1 = np.percentile(depths, 25)
    Q3 = np.percentile(depths, 75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    mask = (depth_map < lower) | (depth_map > upper)
    depth_map[mask] = 0
    return depth_map


def calculate_optimal_voxel_size(depth, height, width, k=10, max_vol_dim=512):
    """Compute voxel size that keeps the volume within max_vol_dim^3."""
    valid = depth[depth > 0]
    if len(valid) == 0:
        return 0.05
    depth_min = np.min(valid)
    depth_max = np.percentile(valid, 98)  # clip outlier depths
    scene_scale = depth_max - depth_min
    # ensure voxel grid fits within max_vol_dim
    voxel_size = max(scene_scale / max_vol_dim, 0.02)
    print(f"[TSDF-DA3] depth range: {depth_min:.2f} ~ {depth_max:.2f}, "
          f"scene_scale={scene_scale:.2f}, voxel_size={voxel_size:.4f}")
    return voxel_size


def convert_extrinsics_pytorch3d(cam_c2w):
    """Convert c2w to PyTorch3D convention (flip X,Y) and return R, T."""
    cam_c2w = cam_c2w.clone()
    cam_c2w[:, :, 0] = -cam_c2w[:, :, 0]
    cam_c2w[:, :, 1] = -cam_c2w[:, :, 1]
    R = cam_c2w[:, :3, :3]
    w2c = cam_c2w.inverse()
    T = w2c[:, :3, -1]
    return R, T


def save_ply(points, colors, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"[TSDF-DA3] Saved point cloud to {filename}")


# ─── TSDF Fusion Core ────────────────────────────────────────────────────

def run_tsdf_fusion(images, depths, intrinsic, cam_c2w, n_imgs=None,
                    voxel_size=None, max_vol_dim=512, depth_percentile=98):
    """
    Run TSDF fusion on RGB-D frames and return point cloud + mesh data.

    Args:
        images: (N, H, W, 3) uint8
        depths: (N, H, W) float
        intrinsic: (3, 3) camera intrinsic matrix
        cam_c2w: (N, 4, 4) camera-to-world poses
        n_imgs: number of frames to fuse (None = all)
        voxel_size: override voxel size (None = auto)
        max_vol_dim: max voxel grid dimension (default 512 to keep volume tractable)
        depth_percentile: clip depth above this percentile to reduce volume size

    Returns:
        point_cloud: (P, 6) [x, y, z, r, g, b]
        voxel_size_used: actual voxel size used
    """
    total = images.shape[0]
    if n_imgs is None:
        n_imgs = total
    n_imgs = min(n_imgs, total)

    height, width = images.shape[1], images.shape[2]

    # Clip extreme depths to shrink the bounding volume
    all_valid = depths[:n_imgs][depths[:n_imgs] > 0]
    if len(all_valid) > 0:
        depth_cap = np.percentile(all_valid, depth_percentile)
        depths = depths.copy()
        depths[depths > depth_cap] = 0
        print(f"[TSDF-DA3] Depth capped at percentile {depth_percentile}: {depth_cap:.3f}")

    if voxel_size is None:
        voxel_size = calculate_optimal_voxel_size(
            depths[:n_imgs], height, width, max_vol_dim=max_vol_dim)

    vol_bnds = np.zeros((3, 2))
    for i in range(n_imgs):
        depth_im = depths[i].astype(float)
        depth_im = remove_outliers_iqr(depth_im.copy())
        cam_pose = cam_c2w[i]
        view_frust_pts = fusion.get_view_frustum(depth_im, intrinsic, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    use_gpu = torch.cuda.is_available()
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size,
                                  max_vol_dim=max_vol_dim, use_gpu=use_gpu)

    for i in tqdm(range(n_imgs), desc="[TSDF] Integrating frames"):
        color_image = images[i]
        depth_im = depths[i].astype(float)
        cam_pose = cam_c2w[i]
        tsdf_vol.integrate(color_image, depth_im, intrinsic, cam_pose, obs_weight=1.0)

    point_cloud = tsdf_vol.get_point_cloud()
    return point_cloud, voxel_size


# ─── Point Cloud Rendering ───────────────────────────────────────────────

def render_pointcloud(points, colors, intrinsic, cam_c2w, H, W, radius=0.008):
    """
    Render a colored point cloud from given camera views using PyTorch3D.

    Args:
        points: (P, 3) numpy array or torch tensor
        colors: (P, 3) in [0,1] numpy array or torch tensor
        intrinsic: (3, 3) numpy intrinsic matrix
        cam_c2w: (N, 4, 4) numpy camera-to-world poses
        H, W: render resolution

    Returns:
        frames_list: list of (H, W, 3) uint8 numpy arrays
        masks_list: list of (H, W) float [0,1] numpy arrays
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cam_c2w_t = torch.tensor(cam_c2w, dtype=torch.float32)
    N = cam_c2w_t.shape[0]

    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()
    if isinstance(colors, np.ndarray):
        colors = torch.from_numpy(colors).float()
    points = points.to(device)
    colors = colors.to(device)

    R, T = convert_extrinsics_pytorch3d(cam_c2w_t)

    K_template = np.zeros((4, 4))
    K_template[0, 0] = intrinsic[0, 0]
    K_template[1, 1] = intrinsic[1, 1]
    K_template[0, 2] = W / 2
    K_template[1, 2] = H / 2
    K_template[2, 3] = 1
    K_template[3, 2] = 1
    K_template = torch.from_numpy(K_template).float().to(device)

    raster_settings = PointsRasterizationSettings(
        bin_size=0, image_size=(H, W), radius=radius, points_per_pixel=10
    )
    point_cloud = Pointclouds(
        points=points.unsqueeze(0), features=colors.unsqueeze(0)
    )

    frames_list = []
    masks_list = []

    for i in range(N):
        K = K_template.unsqueeze(0)
        cameras = PerspectiveCameras(
            K=K, R=R[i:i+1].to(device), T=T[i:i+1].to(device),
            in_ndc=False, image_size=((H, W),), device=device
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

        frame = renderer(point_cloud)
        fragments = rasterizer(point_cloud)
        mask = (fragments.idx[..., 0] >= 0).float()

        frame_np = (frame.squeeze(0).cpu().numpy()[..., :3] * 255).astype(np.uint8)
        mask_np = mask.squeeze(0).cpu().numpy()

        frames_list.append(frame_np)
        masks_list.append(mask_np)

    return frames_list, masks_list


# ─── DA3 => TSDF (Chunk 0): full multi-view reconstruction ──────────────

def da3_to_tsdf_chunk0(images, depths, intrinsics, extrinsics_w2c,
                        n_imgs=49, save_dir=None, name="chunk0"):
    """
    Convert DA3 output to TSDF for chunk 0 (initial video).

    DA3 outputs w2c extrinsics (N, 3, 4). We convert to c2w (N, 4, 4) for TSDF.
    DA3 may output per-frame intrinsics; we take the first one as canonical.

    Args:
        images: (N, H, W, 3) uint8
        depths: (N, H, W) float
        intrinsics: (N, 3, 3) per-frame intrinsics
        extrinsics_w2c: (N, 3, 4) world-to-camera (DA3 convention)
        n_imgs: frames to fuse
        save_dir: directory to save outputs
        name: output name prefix

    Returns:
        pc_points: (P, 3) world coordinates
        pc_colors: (P, 3) RGB [0, 255]
        intrinsic: (3, 3) canonical intrinsic
        cam_c2w: (N, 4, 4) camera-to-world
        voxel_size: used voxel size
    """
    N = images.shape[0]
    n_imgs = min(n_imgs, N)

    # Convert w2c (3x4) -> c2w (4x4)
    cam_c2w = np.zeros((N, 4, 4), dtype=np.float64)
    for i in range(N):
        w2c_4x4 = np.eye(4)
        w2c_4x4[:3, :4] = extrinsics_w2c[i]
        cam_c2w[i] = np.linalg.inv(w2c_4x4)

    intrinsic = intrinsics[0]  # canonical intrinsic

    pc, voxel_size = run_tsdf_fusion(
        images, depths, intrinsic, cam_c2w, n_imgs=n_imgs
    )

    pc_points = pc[:, :3]
    pc_colors = pc[:, 3:6]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_ply(pc_points, pc_colors / 255.0, os.path.join(save_dir, f"{name}_pc.ply"))

    return pc_points, pc_colors, intrinsic, cam_c2w, voxel_size


def da3_to_tsdf_streaming(frames_rgb, depths, intrinsic, cam_c2w,
                           voxel_size=0.02, save_dir=None, name="update"):
    """
    Build TSDF from streaming DA3 depth output (chunk >= 1).

    Here cam_c2w is known from user-specified trajectory.
    Depths come from DA3 streaming inference on the generated video.

    Args:
        frames_rgb: (N, H, W, 3) uint8
        depths: (N, H, W) float
        intrinsic: (3, 3) canonical intrinsic
        cam_c2w: (N, 4, 4) camera-to-world
        voxel_size: voxel size (should match chunk 0)
        save_dir: directory to save outputs
        name: output name prefix

    Returns:
        pc_points: (P, 3)
        pc_colors: (P, 3) in [0, 255]
    """
    pc, _ = run_tsdf_fusion(
        frames_rgb, depths, intrinsic, cam_c2w,
        voxel_size=voxel_size
    )
    pc_points = pc[:, :3]
    pc_colors = pc[:, 3:6]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_ply(pc_points, pc_colors / 255.0, os.path.join(save_dir, f"{name}_pc.ply"))

    return pc_points, pc_colors


def render_tracking_maps_from_cache(pc_points, pc_colors, intrinsic, cam_c2w,
                                      H, W, radius=0.008):
    """
    Render tracking maps from the 3D spatial memory cache.

    Args:
        pc_points: (P, 3) numpy
        pc_colors: (P, 3) in [0, 255] or [0, 1]
        intrinsic: (3, 3)
        cam_c2w: (N, 4, 4)
        H, W: resolution

    Returns:
        masked_frames: list of (H, W, 3) uint8 — masked rendered views
        masks: list of (H, W) float
    """
    colors = pc_colors.copy()
    if colors.max() > 1.0:
        colors = colors / 255.0

    frames, masks = render_pointcloud(pc_points, colors, intrinsic, cam_c2w, H, W, radius)

    masked_frames = []
    for fr, mk in zip(frames, masks):
        masked = (mk[..., None] * fr.astype(float)).astype(np.uint8)
        masked_frames.append(masked)

    return masked_frames, masks


# ─── CLI: standalone usage ────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSDF fusion with DA3 depth")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to DA3 output directory (with results_output/, camera_poses.txt, etc.)")
    parser.add_argument("--name", type=str, default="da3_tsdf")
    parser.add_argument("--n_imgs", type=int, default=49)
    parser.add_argument("--save_dir", type=str, default="outputs_da3_tsdf")
    parser.add_argument("--render_ply", action="store_true")
    args = parser.parse_args()

    print("=== demo_da3.py: TSDF fusion from Depth-Anything-3 output ===")

    output_dir = args.input_dir
    results_dir = os.path.join(output_dir, "results_output")

    poses_raw = np.loadtxt(os.path.join(output_dir, "camera_poses.txt"))
    num_total = poses_raw.shape[0]
    poses_c2w = [poses_raw[i].reshape(4, 4) for i in range(num_total)]
    cam_c2w = np.array(poses_c2w)

    intrinsics_raw = np.loadtxt(os.path.join(output_dir, "intrinsic.txt"))

    all_images = []
    all_depths = []
    all_intrinsics = []

    n_imgs = min(args.n_imgs, num_total)
    for i in tqdm(range(n_imgs), desc="Loading DA3 frames"):
        npz_path = os.path.join(results_dir, f"frame_{i}.npz")
        data = np.load(npz_path)
        all_images.append(data["image"])
        all_depths.append(data["depth"])
        all_intrinsics.append(data["intrinsics"])

    images = np.stack(all_images)
    depths = np.stack(all_depths)
    intrinsic = all_intrinsics[0]

    print(f"  images: {images.shape}, depths: {depths.shape}")
    print(f"  intrinsic:\n{intrinsic}")

    pc, voxel_size = run_tsdf_fusion(images, depths, intrinsic, cam_c2w[:n_imgs], n_imgs=n_imgs)
    pc_points = pc[:, :3]
    pc_colors = pc[:, 3:6]

    os.makedirs(f"{args.save_dir}/{args.name}", exist_ok=True)
    save_ply(pc_points, pc_colors / 255.0, f"{args.save_dir}/{args.name}/pc.ply")

    if args.render_ply:
        H, W = images.shape[1], images.shape[2]
        target_cam_c2w = cam_c2w[:n_imgs]

        frames, masks = render_pointcloud(
            pc_points, pc_colors / 255.0, intrinsic, target_cam_c2w, H, W
        )

        source_list = [images[i] for i in range(min(49, n_imgs))]
        masked_list = []
        for fr, mk in zip(frames, masks):
            masked = (mk[..., None] * fr.astype(float)).astype(np.uint8)
            masked_list.append(masked)

        imageio.mimwrite(f"{args.save_dir}/{args.name}/Vid_source.mp4",
                         source_list, fps=20, quality=8)
        imageio.mimwrite(f"{args.save_dir}/{args.name}/Vid_masktarget.mp4",
                         masked_list, fps=20, quality=8)
        print(f"  Videos saved to {args.save_dir}/{args.name}/")

    print("=== Done ===")
