"""
Type stubs for LiberoGymWrapperClient to provide better IDE autocomplete support.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple
import numpy as np
import numpy.typing as npt


# Type aliases for common array shapes
NDArrayFloat = npt.NDArray[np.float32]
NDArrayUInt8 = npt.NDArray[np.uint8]


# Robot state types
class RobotJoint(NamedTuple):
    """Robot joint state."""
    pos: NDArrayFloat  # (7,) - Joint positions
    pos_cos: NDArrayFloat  # (7,) - Cosine of joint positions
    pos_sin: NDArrayFloat  # (7,) - Sine of joint positions
    vel: NDArrayFloat  # (7,) - Joint velocities


class RobotEEF(NamedTuple):
    """End-effector state."""
    pos: NDArrayFloat  # (3,) - Position in world frame
    quat: NDArrayFloat  # (4,) - Quaternion orientation (w, x, y, z)


class RobotGripper(NamedTuple):
    """Gripper state."""
    qpos: NDArrayFloat  # (2,) - Gripper joint positions
    qvel: NDArrayFloat  # (2,) - Gripper joint velocities


class Robot0(NamedTuple):
    """Complete robot state."""
    joint: RobotJoint
    eef: RobotEEF
    gripper: RobotGripper
    proprio_state: NDArrayFloat  # Proprioceptive state vector


# Camera types
class CameraData(NamedTuple):
    """Camera observation data (all cameras share this base structure)."""
    image: NDArrayUInt8  # (3, H, W) - RGB image in CHW format
    intrinsic_matrix: NDArrayFloat  # (3, 3) - Camera intrinsic matrix
    extrinsic_matrix: NDArrayFloat  # (4, 4) - Camera extrinsic matrix
    # Optional fields (may or may not be present depending on configuration)
    # depth: NDArrayFloat  # (H, W) - Depth map
    # pointcloud: NDArrayFloat  # (N, 3) - Point cloud


# Object state type (dynamic - actual fields depend on scene)
class ObjectState(NamedTuple):
    """State of a single object in the scene."""
    pos: NDArrayFloat  # (3,) - Object position
    quat: NDArrayFloat  # (4,) - Object quaternion
    to_robot0_eef_pos: NDArrayFloat  # (3,) - Position relative to end-effector
    to_robot0_eef_quat: NDArrayFloat  # (4,) - Quaternion relative to end-effector


class LiberoGymWrapperClient:
    """
    Client for Libero Gym HTTP Server.

    Provides access to Libero manipulation environments over HTTP.
    Observations are returned as nested named tuples for type safety and IDE support.
    """

    base_url: str
    instance_id: Optional[str]
    timeout: int
    step_counter: int

    def __init__(
        self,
        base_url: str,
        instance_id: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url: Base URL of the server (e.g., "http://localhost:8000")
            instance_id: Optional environment instance ID
            timeout: Request timeout in seconds
        """
        ...

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
    ) -> Tuple[LiberoGymWrapperClient, str]:
        """
        Create environment on server.

        Args:
            env_name: Environment name (e.g., "libero_spatial_no_noops")
            task_id: Task ID within suite
            image_size_height: Image height in pixels
            image_size_width: Image width in pixels
            require_depth: Include depth maps in observations
            require_point_cloud: Include point clouds in observations
            num_points: Number of points in point cloud
            camera_names: Camera names to use
            max_episode_steps: Maximum steps per episode
            seed: Random seed
            enable_pytorch3d_fps: Use PyTorch3D for point cloud sampling
            pointcloud_process_device: Device for point cloud processing

        Returns:
            Tuple of (self, instance_id)
        """
        ...

    def reset(
        self,
        seed: int = 0,
        instance_id: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset environment.

        Args:
            seed: Random seed for reset
            instance_id: Optional instance ID

        Returns:
            Tuple of (observation, info)

            observation is a named tuple with structure:
                - robot0.joint.pos: NDArrayFloat (7,)
                - robot0.joint.pos_cos: NDArrayFloat (7,)
                - robot0.joint.pos_sin: NDArrayFloat (7,)
                - robot0.joint.vel: NDArrayFloat (7,)
                - robot0.eef.pos: NDArrayFloat (3,)
                - robot0.eef.quat: NDArrayFloat (4,)
                - robot0.gripper.qpos: NDArrayFloat (2,)
                - robot0.gripper.qvel: NDArrayFloat (2,)
                - robot0.proprio_state: NDArrayFloat
                - agentview.image: NDArrayUInt8 (3, H, W)
                - agentview.depth: NDArrayFloat (H, W) [optional]
                - agentview.pointcloud: NDArrayFloat (N, 3) [optional]
                - agentview.intrinsic_matrix: NDArrayFloat (3, 3)
                - agentview.extrinsic_matrix: NDArrayFloat (4, 4)
                - robot0_eye_in_hand: similar structure [optional]
                - objects.{name}.pos: NDArrayFloat (3,)
                - objects.{name}.quat: NDArrayFloat (4,)
                - objects.{name}.to_robot0_eef_pos: NDArrayFloat (3,)
                - objects.{name}.to_robot0_eef_quat: NDArrayFloat (4,)
                - object_state: NDArrayFloat
        """
        ...

    def step(
        self,
        action: Union[np.ndarray, List[float]],
        instance_id: Optional[str] = None,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.

        Args:
            action: Action to execute (7-dimensional for Libero)
            instance_id: Optional instance ID

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            observation has same structure as reset()
        """
        ...

    def sample_action(self, instance_id: Optional[str] = None) -> List[float]:
        """
        Sample random action.

        Args:
            instance_id: Optional instance ID

        Returns:
            Random action from action space
        """
        ...

    def delete(self, instance_id: Optional[str] = None) -> None:
        """
        Delete environment instance.

        Args:
            instance_id: Optional instance ID
        """
        ...

    def __repr__(self) -> str: ...
