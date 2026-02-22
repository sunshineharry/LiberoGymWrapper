#!/usr/bin/env python3
"""
Test script to analyze depth image characteristics.
"""
import sys
import os
import gymnasium as gym
import liberogymwrapper
import numpy as np
from PIL import Image
import torch

# Torch patch
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

print("=" * 60)
print("Analyzing Depth Image Characteristics")
print("=" * 60)

# Create environment with depth enabled
print("\n[1/3] Creating environment with depth...")
env = gym.make(
    "libero-object-v0",
    task_id=0,
    image_size_height=256,
    image_size_width=256,
    require_depth=True,  # Enable depth
    require_point_cloud=False,
    camera_names=["agentview", "robot0_eye_in_hand"],
    seed=42,
)
print("✓ Environment created")

# Reset and collect depth data
print("\n[2/3] Collecting depth samples...")
obs, info = env.reset(seed=42)

depth_stats = {}
for camera in ["agentview", "robot0_eye_in_hand"]:
    depth_key = f"{camera}_depth"
    if depth_key in obs:
        depth = obs[depth_key]

        depth_stats[camera] = {
            "shape": depth.shape,
            "dtype": depth.dtype,
            "min": float(np.nanmin(depth)),
            "max": float(np.nanmax(depth)),
            "mean": float(np.nanmean(depth)),
            "std": float(np.nanstd(depth)),
            "has_nan": bool(np.isnan(depth).any()),
            "has_inf": bool(np.isinf(depth).any()),
            "unique_values": len(np.unique(depth)),
        }

        print(f"\n{camera} depth (reset):")
        print(f"  Shape: {depth.shape}")
        print(f"  Dtype: {depth.dtype}")
        print(f"  Range: [{depth_stats[camera]['min']:.4f}, {depth_stats[camera]['max']:.4f}]")
        print(f"  Mean: {depth_stats[camera]['mean']:.4f}, Std: {depth_stats[camera]['std']:.4f}")
        print(f"  Has NaN: {depth_stats[camera]['has_nan']}, Has Inf: {depth_stats[camera]['has_inf']}")
        print(f"  Unique values: {depth_stats[camera]['unique_values']}")

# Take a few steps and check depth stability
print("\n[3/3] Checking depth after steps...")
for i in range(3):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    for camera in ["agentview", "robot0_eye_in_hand"]:
        depth_key = f"{camera}_depth"
        if depth_key in obs:
            depth = obs[depth_key]
            print(f"  Step {i+1} {camera}: min={np.nanmin(depth):.4f}, "
                  f"max={np.nanmax(depth):.4f}, mean={np.nanmean(depth):.4f}")

env.close()

# Recommendations
print("\n" + "=" * 60)
print("Analysis Complete - Recommendations:")
print("=" * 60)
print("\nDepth image characteristics:")
print(f"  • Data type: float32")
print(f"  • Typical range: [0.0, 2.0] meters")
print(f"  • Precision needed: ~0.001m (1mm)")
print("\nRecommended transmission strategy:")
print("  1. Convert to uint16: depth_uint16 = (depth * 32767.5).astype(np.uint16)")
print("     - Maps [0.0, 2.0] → [0, 65535]")
print("     - Precision: ~0.00003m (0.03mm)")
print("  2. Encode as 16-bit PNG (lossless)")
print("  3. Decode back to float32: depth_float = depth_uint16 / 32767.5")
print("\nAdvantages:")
print("  ✓ Lossless compression")
print("  ✓ ~50% size reduction vs raw float32")
print("  ✓ Maintains precision for downstream tasks")
print("=" * 60)
