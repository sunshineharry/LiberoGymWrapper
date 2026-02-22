#!/usr/bin/env python3
"""
Test script to verify step() returns correct images.
Mimics test_wrapper.ipynb to isolate the issue.
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

# Output directory
OUTPUT_DIR = "/tmp/test_step_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Testing Libero Gym Environment - Step Images")
print("=" * 60)

# Create environment (same as test_wrapper.ipynb but smaller resolution)
print("\n[1/5] Creating environment...")
env = gym.make(
    "libero-object-v0",
    task_id=0,
    image_size_height=256,
    image_size_width=256,
    require_depth=False,
    require_point_cloud=False,
    camera_names=["agentview", "robot0_eye_in_hand"],
    seed=42,
)
print("✓ Environment created")

# Reset
print("\n[2/5] Resetting environment...")
obs, info = env.reset(seed=42)
print("✓ Reset complete")
print(f"  Task: {info['task_description']}")

# Save reset images
print("\n[3/5] Saving RESET images...")
for camera in ["agentview", "robot0_eye_in_hand"]:
    key = f"{camera}_image"
    if key in obs:
        img_chw = obs[key]  # Should be (C, H, W)
        print(f"  {key}: shape={img_chw.shape}, dtype={img_chw.dtype}, "
              f"min={img_chw.min()}, max={img_chw.max()}, mean={img_chw.mean():.2f}")

        # Convert to (H, W, C) for saving
        img_hwc = img_chw.transpose(1, 2, 0)
        img_pil = Image.fromarray(img_hwc.astype(np.uint8))
        img_pil.save(f"{OUTPUT_DIR}/reset_{camera}.png")
        print(f"    Saved to {OUTPUT_DIR}/reset_{camera}.png")

# Take steps
print("\n[4/5] Taking STEPS and saving images...")
for i in range(5):
    action = env.action_space.sample()

    print(f"\n  Step {i+1}:")
    print(f"    Action: {action[:3]}...")  # Print first 3 elements

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"    Reward: {reward:.3f}, Terminated: {terminated}, Truncated: {truncated}")

    # Save step images
    for camera in ["agentview", "robot0_eye_in_hand"]:
        key = f"{camera}_image"
        if key in obs:
            img_chw = obs[key]
            print(f"    {key}: shape={img_chw.shape}, min={img_chw.min()}, "
                  f"max={img_chw.max()}, mean={img_chw.mean():.2f}")

            # Convert to (H, W, C) for saving
            img_hwc = img_chw.transpose(1, 2, 0)
            img_pil = Image.fromarray(img_hwc.astype(np.uint8))
            img_pil.save(f"{OUTPUT_DIR}/step{i+1}_{camera}.png")
            print(f"      Saved to {OUTPUT_DIR}/step{i+1}_{camera}.png")

    if terminated or truncated:
        print(f"  Episode ended at step {i+1}")
        break

# Cleanup
print("\n[5/5] Cleaning up...")
env.close()
print("✓ Environment closed")

print("\n" + "=" * 60)
print("Test complete!")
print(f"Output directory: {OUTPUT_DIR}")
print("\nPlease check the images:")
print("  - reset_*.png should be CLEAR")
print("  - step*_*.png should be CLEAR")
print("\nIf step images are corrupted, the problem is in libero_env.py step()")
print("=" * 60)
