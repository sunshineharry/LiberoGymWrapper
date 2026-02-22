"""
Libero Gym Wrapper Client - Simplified Version

Uses disk-based image transfer to ensure data integrity.
Images are received as base64, saved to disk, then loaded - simple and reliable.

Observation Structure:
    The observation returned by reset() and step() is a nested named tuple with the following structure:

    obs.robot0.joint.pos           - Joint positions (7,) ndarray
    obs.robot0.joint.pos_cos       - Cosine of joint positions (7,) ndarray
    obs.robot0.joint.pos_sin       - Sine of joint positions (7,) ndarray
    obs.robot0.joint.vel           - Joint velocities (7,) ndarray
    obs.robot0.eef.pos             - End-effector position (3,) ndarray
    obs.robot0.eef.quat            - End-effector quaternion (4,) ndarray
    obs.robot0.gripper.qpos        - Gripper joint positions (2,) ndarray
    obs.robot0.gripper.qvel        - Gripper joint velocities (2,) ndarray
    obs.robot0.proprio_state       - Proprioceptive state vector ndarray

    obs.agentview.image            - RGB image (3, H, W) ndarray
    obs.agentview.depth            - Depth map (H, W) ndarray [optional]
    obs.agentview.pointcloud       - Point cloud (N, 3) ndarray [optional]
    obs.agentview.intrinsic_matrix - Camera intrinsics (3, 3) ndarray
    obs.agentview.extrinsic_matrix - Camera extrinsics (4, 4) ndarray

    obs.robot0_eye_in_hand.image            - RGB image (3, H, W) ndarray [optional]
    obs.robot0_eye_in_hand.depth            - Depth map (H, W) ndarray [optional]
    obs.robot0_eye_in_hand.pointcloud       - Point cloud (N, 3) ndarray [optional]
    obs.robot0_eye_in_hand.intrinsic_matrix - Camera intrinsics (3, 3) ndarray [optional]
    obs.robot0_eye_in_hand.extrinsic_matrix - Camera extrinsics (4, 4) ndarray [optional]

    obs.objects.{object_name}.pos                  - Object position (3,) ndarray
    obs.objects.{object_name}.quat                 - Object quaternion (4,) ndarray
    obs.objects.{object_name}.to_robot0_eef_pos    - Position relative to EEF (3,) ndarray
    obs.objects.{object_name}.to_robot0_eef_quat   - Quaternion relative to EEF (4,) ndarray

    obs.object_state - Concatenated object state vector ndarray

Example:
    >>> client = LiberoGymWrapperClient("http://localhost:8000")
    >>> client.create_env("libero_spatial_no_noops", task_id=0)
    >>> obs, info = client.reset()
    >>>
    >>> # Access robot state with autocomplete
    >>> joint_pos = obs.robot0.joint.pos
    >>> eef_pos = obs.robot0.eef.pos
    >>>
    >>> # Access camera data
    >>> rgb = obs.agentview.image
    >>> depth = obs.agentview.depth
    >>>
    >>> # Access object states
    >>> for obj_name in obs.objects._fields:
    >>>     obj = getattr(obs.objects, obj_name)
    >>>     print(f"{obj_name}: {obj.pos}")
"""
import base64
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, Tuple
from collections import namedtuple

import numpy as np
import requests
from PIL import Image


# Create temp directory for received images
IMAGE_RECEIVE_DIR = Path(tempfile.gettempdir()) / "libero_http_images_client"
IMAGE_RECEIVE_DIR.mkdir(exist_ok=True)


# Fixed key mapping rules
FIXED_KEY_MAPPING = {
    # robot0 related
    'robot0_joint_pos': 'robot0.joint.pos',
    'robot0_joint_pos_cos': 'robot0.joint.pos_cos',
    'robot0_joint_pos_sin': 'robot0.joint.pos_sin',
    'robot0_joint_vel': 'robot0.joint.vel',
    'robot0_eef_pos': 'robot0.eef.pos',
    'robot0_eef_quat': 'robot0.eef.quat',
    'robot0_gripper_qpos': 'robot0.gripper.qpos',
    'robot0_gripper_qvel': 'robot0.gripper.qvel',
    'robot0_proprio-state': 'robot0.proprio_state',

    # agentview camera
    'agentview_image': 'agentview.image',
    'agentview_depth': 'agentview.depth',
    'agentview_pointcloud': 'agentview.pointcloud',
    'agentview_intrinsic_matrix': 'agentview.intrinsic_matrix',
    'agentview_extrinsic_matrix': 'agentview.extrinsic_matrix',

    # robot0_eye_in_hand camera
    'robot0_eye_in_hand_image': 'robot0_eye_in_hand.image',
    'robot0_eye_in_hand_depth': 'robot0_eye_in_hand.depth',
    'robot0_eye_in_hand_pointcloud': 'robot0_eye_in_hand.pointcloud',
    'robot0_eye_in_hand_intrinsic_matrix': 'robot0_eye_in_hand.intrinsic_matrix',
    'robot0_eye_in_hand_extrinsic_matrix': 'robot0_eye_in_hand.extrinsic_matrix',

    # global state
    'object-state': 'object_state',
}

# Dynamic object field suffixes
OBJECT_SUFFIXES = [
    '_pos',
    '_quat',
    '_to_robot0_eef_pos',
    '_to_robot0_eef_quat'
]


def _decode_observation(obs_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode observation data from server.

    RGB images: Decoded from JPEG.
    Depth images: Decoded from 16-bit PNG and converted back to float32.
    Other data: Converted from lists to numpy arrays.

    Args:
        obs_data: Observation dictionary from server

    Returns:
        Decoded observation dictionary with numpy arrays
    """
    import io
    result = {}

    for key, value in obs_data.items():
        if "image" in key and isinstance(value, str):
            # RGB images: decode from JPEG
            img_bytes = base64.b64decode(value.encode('ascii'))
            img_pil = Image.open(io.BytesIO(img_bytes))
            img_hwc = np.array(img_pil)  # (H, W, C) format

            # Convert to (C, H, W) format to match environment's output
            if img_hwc.ndim == 3:
                img_chw = img_hwc.transpose(2, 0, 1)
            else:
                img_chw = img_hwc

            result[key] = img_chw
            # print(f"[CLIENT] Decoded {key}: shape={img_chw.shape}, size={len(img_bytes)} bytes")

        elif "depth" in key and isinstance(value, str):
            # Depth images: decode from 16-bit PNG
            depth_bytes = base64.b64decode(value.encode('ascii'))
            depth_pil = Image.open(io.BytesIO(depth_bytes))
            depth_uint16 = np.array(depth_pil)  # (H, W) uint16

            # Convert uint16 [0, 65535] back to float32 [0.0, 2.0]
            # Scale factor: 2.0 / 65535 = 1 / 32767.5
            depth_float = (depth_uint16 / 32767.5).astype(np.float32)

            result[key] = depth_float
            # print(f"[CLIENT] Decoded {key}: shape={depth_float.shape}, "
            #       f"range=[{depth_float.min():.3f}, {depth_float.max():.3f}]")

        elif isinstance(value, list):
            # Convert lists back to numpy arrays
            result[key] = np.array(value)
        else:
            result[key] = value

    return result


def _convert_obs_to_namedtuple(obs_dict: Dict[str, Any]) -> Any:
    """
    Convert observation dictionary to nested named tuple structure.

    Args:
        obs_dict: Observation dictionary with flat keys

    Returns:
        Named tuple with hierarchical structure
    """
    # Build hierarchical structure
    hierarchical = {}

    for key, value in obs_dict.items():
        # Check if it's a fixed key
        if key in FIXED_KEY_MAPPING:
            new_key = FIXED_KEY_MAPPING[key]
        else:
            # Check if it's a dynamic object field
            # Sort suffixes by length (longest first) to match more specific patterns first
            matched = False
            for suffix in sorted(OBJECT_SUFFIXES, key=len, reverse=True):
                if key.endswith(suffix):
                    object_name = key[:-len(suffix)]
                    # Remove leading underscore from suffix
                    attr_name = suffix[1:]
                    new_key = f'objects.{object_name}.{attr_name}'
                    matched = True
                    break

            if not matched:
                # Unknown key, keep as-is
                new_key = key

        # Split key by dots and build nested structure
        parts = new_key.split('.')
        current = hierarchical

        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    # Recursively convert dict to namedtuple
    def dict_to_namedtuple(d: Dict[str, Any], name: str = 'Observation') -> Any:
        if not isinstance(d, dict):
            return d

        # Sort keys for consistent namedtuple field ordering
        sorted_keys = sorted(d.keys())

        # Recursively convert nested dicts
        converted_values = []
        for k in sorted_keys:
            v = d[k]
            if isinstance(v, dict):
                converted_values.append(dict_to_namedtuple(v, name=k.capitalize()))
            else:
                converted_values.append(v)

        # Create namedtuple class
        NT = namedtuple(name, sorted_keys)
        return NT(*converted_values)

    return dict_to_namedtuple(hierarchical)


class LiberoGymWrapperClient:
    """
    Client for Libero Gym HTTP Server.

    Simple implementation using disk-based image transfer for maximum reliability.

    Attributes:
        base_url: Base URL of the server (e.g., "http://localhost:8000")
        instance_id: ID of the current environment instance
        timeout: Request timeout in seconds
        step_counter: Number of steps executed in current episode
        sess: Persistent HTTP session

    Example:
        >>> client = LiberoGymWrapperClient("http://localhost:8000")
        >>> client.create_env("libero_spatial_no_noops", task_id=0)
        >>> obs, info = client.reset()
        >>> action = client.sample_action()
        >>> obs, reward, terminated, truncated, info = client.step(action)
        >>> client.delete()
    """

    def __init__(
        self,
        base_url: str,
        instance_id: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url: Base URL of the Libero Gym HTTP server (e.g., "http://localhost:8000")
            instance_id: Optional environment instance ID to use
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip("/")
        self.instance_id = instance_id
        self.timeout = timeout
        self.step_counter = 0
        self.sess = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, *, params=None, json=None) -> requests.Response:
        resp = self.sess.request(
            method=method,
            url=self._url(path),
            params=params,
            json=json,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp

    @staticmethod
    def _bool_q(v: bool) -> str:
        return "true" if v else "false"

    def create_env(
        self,
        env_name: str,
        task_id: int = 0,
        image_size_height: int = 256,
        image_size_width: int = 256,
        require_depth: bool = False,
        require_point_cloud: bool = False,
        num_points: int = 8192,
        camera_names: Union[str, List[str]] = ["agentview", "robot0_eye_in_hand"],
        max_episode_steps: int = 600,
        seed: int = 0,
        enable_pytorch3d_fps: bool = False,
        pointcloud_process_device: str = "cpu",
    ) -> Tuple["LiberoGymWrapperClient", str]:
        """
        Create environment on server.

        Args:
            env_name: Name of the environment (e.g., "libero_spatial_no_noops")
            task_id: Task ID within the suite (default: 0)
            image_size_height: Height of rendered images in pixels (default: 256)
            image_size_width: Width of rendered images in pixels (default: 256)
            require_depth: Whether to include depth maps in observations (default: False)
            require_point_cloud: Whether to include point clouds in observations (default: False)
            num_points: Number of points in point cloud if enabled (default: 8192)
            camera_names: List of camera names or comma-separated string (default: ["agentview", "robot0_eye_in_hand"])
            max_episode_steps: Maximum steps per episode (default: 600)
            seed: Random seed for environment (default: 0)
            enable_pytorch3d_fps: Use PyTorch3D for faster point cloud sampling (default: False)
            pointcloud_process_device: Device for point cloud processing, "cpu" or "cuda" (default: "cpu")

        Returns:
            Tuple of (self, instance_id) where instance_id is the created environment ID

        Example:
            >>> client = LiberoGymWrapperClient("http://localhost:8000")
            >>> client, instance_id = client.create_env(
            ...     env_name="libero_spatial_no_noops",
            ...     task_id=0,
            ...     require_depth=True,
            ...     require_point_cloud=True
            ... )
            >>> print(f"Created environment: {instance_id}")
        """

        if isinstance(camera_names, list):
            camera_names = ",".join([c.strip() for c in camera_names])

        params = {
            "env_name": env_name,
            "task_id": task_id,
            "image_size_height": image_size_height,
            "image_size_width": image_size_width,
            "require_depth": self._bool_q(require_depth),
            "require_point_cloud": self._bool_q(require_point_cloud),
            "num_points": num_points,
            "camera_names": camera_names,
            "max_episode_steps": max_episode_steps,
            "seed": seed,
            "enable_pytorch3d_fps": self._bool_q(enable_pytorch3d_fps),
            "pointcloud_process_device": pointcloud_process_device,
        }

        data = self._request("POST", "/v1/envs/", params=params).json()
        iid = data["instance_id"]
        self.instance_id = iid
        self.step_counter = 0
        return (self, iid)

    def reset(
        self,
        seed: int = 0,
        instance_id: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for environment reset (default: 0)
            instance_id: Optional environment instance ID, uses self.instance_id if None

        Returns:
            Tuple of (observation, info) where:
                - observation: Named tuple with hierarchical structure containing:
                    - robot0: Robot state (joint positions, velocities, end-effector pose, gripper state)
                    - agentview: Camera data (image, depth, pointcloud, intrinsics, extrinsics)
                    - robot0_eye_in_hand: Eye-in-hand camera data (optional, same structure as agentview)
                    - objects: Dynamic object states with fields like pos, quat, to_robot0_eef_pos, etc.
                    - object_state: Concatenated object state vector
                - info: Dictionary with additional information from the environment

        Raises:
            ValueError: If no instance_id is available

        Example:
            >>> obs, info = client.reset(seed=42)
            >>> # Access robot state
            >>> joint_pos = obs.robot0.joint.pos  # shape: (7,)
            >>> eef_pos = obs.robot0.eef.pos      # shape: (3,)
            >>> # Access camera data
            >>> rgb = obs.agentview.image          # shape: (3, 256, 256)
            >>> # Access object states
            >>> for obj_name in obs.objects._fields:
            ...     obj = getattr(obs.objects, obj_name)
            ...     print(f"{obj_name}: position={obj.pos}")
        """

        iid = instance_id or self.instance_id
        if not iid:
            raise ValueError("No instance_id available")

        self.step_counter = 0
        data = self._request("POST", f"/v1/envs/{iid}/reset/", params={"seed": seed}).json()

        obs_dict = _decode_observation(data["observation"])
        obs = _convert_obs_to_namedtuple(obs_dict)
        info = data["info"]

        return obs, info

    def step(
        self,
        action: Any,
        instance_id: Optional[str] = None,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step with the given action.

        Args:
            action: Action to execute. Can be numpy array or list.
                   For Libero environments, typically a 7-dimensional array for joint control.
            instance_id: Optional environment instance ID, uses self.instance_id if None

        Returns:
            Tuple of (observation, reward, terminated, truncated, info) where:
                - observation: Named tuple with same structure as reset() (see reset() for details)
                - reward: Float reward value for this step
                - terminated: Boolean indicating if episode ended due to success/failure
                - truncated: Boolean indicating if episode ended due to time limit
                - info: Dictionary with additional step information

        Raises:
            ValueError: If no instance_id is available

        Example:
            >>> # Random action
            >>> action = client.sample_action()
            >>> obs, reward, terminated, truncated, info = client.step(action)
            >>> print(f"Reward: {reward}, Done: {terminated or truncated}")
            >>>
            >>> # Custom action
            >>> import numpy as np
            >>> action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            >>> obs, reward, terminated, truncated, info = client.step(action)
            >>> new_eef_pos = obs.robot0.eef.pos
        """

        iid = instance_id or self.instance_id
        if not iid:
            raise ValueError("No instance_id available")

        # Convert action to list if needed
        if isinstance(action, np.ndarray):
            action = action.tolist()

        self.step_counter += 1
        data = self._request("POST", f"/v1/envs/{iid}/step/", json={"action": action}).json()

        obs_dict = _decode_observation(data["observation"])
        obs = _convert_obs_to_namedtuple(obs_dict)
        info = data["info"]

        return obs, data["reward"], data["terminated"], data["truncated"], info

    def sample_action(self, instance_id: Optional[str] = None) -> Any:
        """
        Sample a random action from the environment's action space.

        Args:
            instance_id: Optional environment instance ID, uses self.instance_id if None

        Returns:
            Random action sampled from the action space (typically a list of floats)

        Raises:
            ValueError: If no instance_id is available

        Example:
            >>> action = client.sample_action()
            >>> print(f"Sampled action: {action}")
            >>> obs, reward, terminated, truncated, info = client.step(action)
        """
        iid = instance_id or self.instance_id
        if not iid:
            raise ValueError("No instance_id available")

        data = self._request("GET", f"/v1/envs/{iid}/action_space/sample/").json()
        return data["action"]

    def delete(self, instance_id: Optional[str] = None) -> None:
        """
        Delete environment instance from server.

        Args:
            instance_id: Optional environment instance ID, uses self.instance_id if None

        Raises:
            ValueError: If no instance_id is available

        Example:
            >>> client.delete()  # Delete current environment
            >>> # Or delete a specific instance
            >>> client.delete(instance_id="specific-id")
        """
        iid = instance_id or self.instance_id
        if not iid:
            raise ValueError("No instance_id available")

        self._request("DELETE", f"/v1/envs/{iid}/")
        if iid == self.instance_id:
            self.instance_id = None

    def __repr__(self) -> str:
        return f"LiberoGymWrapperClient(base_url='{self.base_url}', instance_id={self.instance_id!r})"
