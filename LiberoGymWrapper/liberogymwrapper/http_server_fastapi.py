#!/usr/bin/env python3
"""
Libero Gym HTTP Server - FastAPI Version

Uses FastAPI instead of Flask to avoid potential threading/request handling issues.
"""
import sys
import uuid
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import base64
import numpy as np
import gymnasium as gym
from PIL import Image
import liberogymwrapper

# Torch compatibility patch
import torch
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Create temp directory for image storage
IMAGE_STORAGE_DIR = Path(tempfile.gettempdir()) / "libero_fastapi_images"
IMAGE_STORAGE_DIR.mkdir(exist_ok=True)
print(f"[INFO] Image storage directory: {IMAGE_STORAGE_DIR}")


def _encode_observation(obs: Dict[str, Any], image_quality: int = 100, save_debug: bool = False,
                        instance_id: str = "", step_counter: int = 0) -> Dict[str, Any]:
    """
    Encode observation for transmission.

    Images: JPEG compression for efficient transmission (80-90% smaller than PNG).
    Depth: 16-bit PNG encoding (lossless, ~50% size reduction vs float32).
    Other data: JSON lists.

    Args:
        obs: Observation dictionary from environment
        image_quality: JPEG quality for RGB images (1-100, default 85)
        save_debug: Whether to save debug images to disk
        instance_id: Environment instance ID (for debug saving)
        step_counter: Step counter (for debug saving)

    Returns:
        Dictionary with base64-encoded data ready for JSON transmission
    """
    import io
    result = {}

    for key, value in obs.items():
        if "image" in key and isinstance(value, np.ndarray):
            # RGB images: JPEG compression
            img_copy = np.array(value, copy=True, order='C')

            # Convert to (H, W, C) format
            if img_copy.ndim == 3 and img_copy.shape[0] == 3:
                img_hwc = img_copy.transpose(1, 2, 0)
            else:
                img_hwc = img_copy

            # Encode to PNG in memory
            img_pil = Image.fromarray(img_hwc.astype(np.uint8), mode='RGB')
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG', quality=image_quality, optimize=False)
            jpeg_bytes = buffer.getvalue()

            result[key] = base64.b64encode(jpeg_bytes).decode('ascii')
            print(f"[ENCODE] {key}: shape={img_hwc.shape}, JPEG={len(jpeg_bytes)} bytes "
                  f"(quality={image_quality})")

            if save_debug and instance_id:
                debug_path = IMAGE_STORAGE_DIR / f"{instance_id}_{key}_{step_counter}.jpg"
                img_pil.save(str(debug_path), quality=image_quality, optimize=True)

        elif "depth" in key and isinstance(value, np.ndarray):
            # Depth images: 16-bit PNG encoding (lossless)
            depth_copy = np.array(value, copy=True, order='C')

            # Convert float32 [0.0, 2.0] to uint16 [0, 65535]
            # Scale factor: 65535 / 2.0 = 32767.5
            depth_uint16 = np.clip(depth_copy * 32767.5, 0, 65535).astype(np.uint16)

            # Encode to 16-bit PNG
            img_pil = Image.fromarray(depth_uint16, mode='I;16')
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            png_bytes = buffer.getvalue()

            result[key] = base64.b64encode(png_bytes).decode('ascii')
            print(f"[ENCODE] {key}: shape={depth_copy.shape}, 16-bit PNG={len(png_bytes)} bytes")

            if save_debug and instance_id:
                debug_path = IMAGE_STORAGE_DIR / f"{instance_id}_{key}_{step_counter}.png"
                img_pil.save(str(debug_path))

        elif isinstance(value, np.ndarray):
            # Other numpy arrays: convert to lists
            result[key] = value.tolist()
        else:
            # Keep other types as-is
            result[key] = value

    return result


class Envs:
    """Container for environment instances"""

    def __init__(self):
        self.envs = {}
        self.step_counters = {}
        self.id_len = 8

    def _lookup_env(self, instance_id):
        if instance_id not in self.envs:
            raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
        return self.envs[instance_id]

    def create(
        self,
        env_name: str,
        task_id: int = 0,
        image_size_height: int = 256,
        image_size_width: int = 256,
        require_depth: bool = False,
        require_point_cloud: bool = False,
        num_points: int = 8192,
        camera_names: List[str] = None,
        max_episode_steps: int = None,
        seed: int = 0,
        enable_pytorch3d_fps: bool = False,
        pointcloud_process_device: str = "cpu",
    ):
        if camera_names is None:
            camera_names = ["agentview", "robot0_eye_in_hand"]

        print(f"[CREATE] env_name={env_name}, task_id={task_id}, cameras={camera_names}")

        env = gym.make(
            env_name,
            task_id=task_id,
            image_size_height=image_size_height,
            image_size_width=image_size_width,
            require_depth=require_depth,
            require_point_cloud=require_point_cloud,
            num_points=num_points,
            camera_names=camera_names,
            max_episode_steps=max_episode_steps,
            seed=seed,
            enable_pytorch3d_fps=enable_pytorch3d_fps,
            pointcloud_process_device=pointcloud_process_device,
        )

        instance_id = str(uuid.uuid4().hex)[:self.id_len]
        self.envs[instance_id] = env
        self.step_counters[instance_id] = 0
        return instance_id

    def reset(self, instance_id: str, seed: int):
        env = self._lookup_env(instance_id)
        obs, info = env.reset(seed=seed)
        self.step_counters[instance_id] = 0
        return obs, info

    def step(self, instance_id: str, action):
        env = self._lookup_env(instance_id)
        nice_action = np.array(action)
        obs, reward, terminated, truncated, info = env.step(nice_action)
        self.step_counters[instance_id] += 1
        return obs, reward, terminated, truncated, info

    def get_step_counter(self, instance_id: str):
        return self.step_counters.get(instance_id, 0)

    def get_action_space_sample(self, instance_id: str):
        env = self._lookup_env(instance_id)
        action = env.action_space.sample()
        return action.tolist() if hasattr(action, 'tolist') else action

    def env_close(self, instance_id: str):
        env = self._lookup_env(instance_id)
        env.close()
        del self.envs[instance_id]
        del self.step_counters[instance_id]


# FastAPI app
app = FastAPI(title="Libero Gym HTTP Server")
envs = Envs()


# Request models
class StepRequest(BaseModel):
    action: List[float]


@app.post("/v1/envs/")
async def env_create(
    env_name: str,
    task_id: int = 0,
    image_size_height: int = 256,
    image_size_width: int = 256,
    require_depth: str = "false",
    require_point_cloud: str = "false",
    num_points: int = 8192,
    camera_names: str = "agentview,robot0_eye_in_hand",
    max_episode_steps: int = None,
    seed: int = 0,
    enable_pytorch3d_fps: str = "false",
    pointcloud_process_device: str = "cpu",
):
    """Create environment instance"""
    camera_list = [name.strip() for name in camera_names.split(",")]
    require_depth_bool = require_depth.lower() == "true"
    require_point_cloud_bool = require_point_cloud.lower() == "true"
    enable_pytorch3d_fps_bool = enable_pytorch3d_fps.lower() == "true"

    instance_id = envs.create(
        env_name=env_name,
        task_id=task_id,
        image_size_height=image_size_height,
        image_size_width=image_size_width,
        require_depth=require_depth_bool,
        require_point_cloud=require_point_cloud_bool,
        num_points=num_points,
        camera_names=camera_list,
        max_episode_steps=max_episode_steps,
        seed=seed,
        enable_pytorch3d_fps=enable_pytorch3d_fps_bool,
        pointcloud_process_device=pointcloud_process_device,
    )

    return {"instance_id": instance_id}


@app.post("/v1/envs/{instance_id}/reset/")
async def env_reset(instance_id: str, seed: int = 0):
    """Reset environment and return observation"""
    print(f"[RESET] instance_id={instance_id}, seed={seed}")

    obs, info = envs.reset(instance_id, seed)
    step_counter = envs.get_step_counter(instance_id)

    # Encode observation (JPEG compression, no disk IO)
    obs_encoded = _encode_observation(obs, image_quality=100, save_debug=False,
                                      instance_id=instance_id, step_counter=step_counter)

    return {"observation": obs_encoded, "info": info}


@app.post("/v1/envs/{instance_id}/step/")
async def env_step(instance_id: str, request: StepRequest):
    """Execute one step"""
    print(f"[STEP] instance_id={instance_id}")

    obs, reward, terminated, truncated, info = envs.step(instance_id, request.action)
    step_counter = envs.get_step_counter(instance_id)

    # Encode observation (PNG compression, no disk IO)
    obs_encoded = _encode_observation(obs, image_quality=100, save_debug=False,
                                      instance_id=instance_id, step_counter=step_counter)

    # Convert info dict to simple types
    info_simple = {}
    for k, v in info.items():
        if isinstance(v, np.ndarray):
            info_simple[k] = v.tolist()
        else:
            info_simple[k] = v

    return {
        "observation": obs_encoded,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info_simple
    }


@app.get("/v1/envs/{instance_id}/action_space/sample/")
async def env_action_space_sample(instance_id: str):
    """Sample random action"""
    action = envs.get_action_space_sample(instance_id)
    return {"action": action}


@app.delete("/v1/envs/{instance_id}/")
async def env_close(instance_id: str):
    """Close environment"""
    envs.env_close(instance_id)
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Libero Gym FastAPI Server")
    parser.add_argument("-l", "--listen", default="0.0.0.0")
    parser.add_argument("-p", "--port", default=40004, type=int)

    args = parser.parse_args()
    print(f"Server starting at: http://{args.listen}:{args.port}")
    print(f"Image storage: {IMAGE_STORAGE_DIR}")

    # Use single worker to avoid any multiprocessing issues
    uvicorn.run(app, host=args.listen, port=args.port, workers=1)
