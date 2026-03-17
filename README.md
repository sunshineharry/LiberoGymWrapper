# A Gymnasium Wrapper for LIBERO
This project provides a wrapper to make the [LIBERO](https://github.com/sunshineharry/lerobot-libero) robotics benchmark compatible with the modern [Gymnasium](https://gymnasium.farama.org/) API.

This wrapper allows you to create, reset, and step through LIBERO environments using the standard `gymnasium.make()` interface.

## Dependencies
Before using this wrapper, ensure you have the required dependencies. The primary dependencies are:

- `gymnasium>=0.29.1`

- `libero @ git+https://github.com/sunshineharry/lerobot-libero.git#egg=libero`

- `torch`

Optional dependencies for point cloud support:

- `open3d` (required if `require_point_cloud=True`)

- `pytorch3d` (required if `enable_pytorch3d_fps=True`)

You can typically install this wrapper and its dependencies by running `pip install .` from the project's root directory.

## Usage
To use the wrapper, you must first import `liberogymwrapper` in your script. This will automatically register the LIBERO environments with the Gymnasium framework.

**Example 1: Basic Initialization and Data Access**

This example shows how to create an environment, reset it, and access the initial observation and task description.

```python
import gymnasium as gym
import liberogymwrapper  # 1. Import wrapper to register environments
import numpy as np
from tqdm import tqdm

print("Iterating through 'libero-10-v0' tasks...")

# We can loop through all 10 tasks in the 'libero-10-v0' suite
for i in tqdm(range(10)):

    # 2. Create the environment using gym.make()
    env = gym.make(
        "libero-10-v0",                # Task suite name
        task_id=i,                     # The ID of the task to load (0-9)
        image_size_height=224,         # Image height
        image_size_width=224,          # Image width
        require_depth=True,            # Include depth observations
        camera_names=["agentview"],    # A list of cameras to use
        seed=0,
    )

    # 3. Reset the environment to get the first observation
    # obs is a dictionary containing robot state and camera images
    obs, info = env.reset()

    # 4. Access the natural language task description (prompt)
    prompt = env.unwrapped.task_description

    # 5. Access data from the observation dictionary
    # Images are in (C, H, W) format and are RGB
    agentview_image = obs["agentview_image"]           # Shape: (3, 224, 224)
    agentview_depth = obs["agentview_depth"]           # Shape: (224, 224)
    agentview_intrinsic = obs["agentview_intrinsic_matrix"]  # Shape: (3, 3)
    agentview_extrinsic = obs["agentview_extrinsic_matrix"]  # Shape: (4, 4)

    print(f"\n[Task ID {i}]")
    print(f"  Prompt: {prompt}")
    print(f"  Image shape: {agentview_image.shape}")
    print(f"  Depth shape: {agentview_depth.shape}")

    # 6. Always close the environment when done
    env.close()

print("\nAll tasks iterated successfully.")
```

**Example 2: Basic Environment Step Loop**

This example shows the standard Gymnasium reset-step loop. It demonstrates how to interact with the environment by sending actions and receiving new observations until the episode ends.

For demonstration, `env.action_space.sample()` is used to generate random actions. In a real application, you would replace this with your own policy.

```python
import gymnasium as gym
import liberogymwrapper
import numpy as np
import time

# 1. Create a single environment
print("Creating 'libero-goal-v0' environment...")
env = gym.make(
    "libero-goal-v0",              # Use the "goal" task suite
    task_id=5,                     # Load task #5
    image_size_height=224,         # Image height
    image_size_width=224,          # Image width
    require_depth=True,            # Include depth observations
    camera_names=["agentview", "robot0_eye_in_hand"],  # Use multiple cameras
    max_episode_steps=600,         # Maximum steps per episode
    seed=0,
)

# 2. Reset the environment
obs, info = env.reset()
print(f"Starting task: {env.unwrapped.task_description}")

# --- 3. (Optional) Stabilize the Environment ---
# In simulation, it is often good practice to let objects
# settle before starting the control loop.
print("Stabilizing environment...")
dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])  # [dx, dy, dz, droll, dpitch, dyaw, gripper]
for _ in range(20):
    obs, _, _, _, _ = env.step(dummy_action)
# --- End Stabilization ---

# 4. Run the main control loop
print("Starting main control loop...")
done = False
step_count = 0
start_time = time.time()

while not done:
    # 5. Get an action
    # REPLACE THIS with your policy's action
    # action = my_policy.get_action(obs, env.unwrapped.task_description)
    action = env.action_space.sample()

    # 6. Step the environment with the action
    obs, reward, terminated, truncated, info = env.step(action)

    # 7. Check if the episode is finished
    done = terminated or truncated
    step_count += 1

end_time = time.time()
print(f"\nEpisode finished in {end_time - start_time:.2f} seconds.")
print(f"Total steps: {step_count}")

# 8. Close the environment
env.close()
```

## Environment Reference

**Available Environment IDs:**

The following environment IDs are registered by this wrapper:

- `libero-10-v0` (A subset of 10 tasks)
- `libero-90-v0` (The full set of 90 tasks)
- `libero-goal-v0`
- `libero-object-v0`
- `libero-spatial-v0`

**`gym.make()` Keyword Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `task_id` | `int` | `0` | The specific task to load from the suite (0-9 for most suites, 0-89 for libero-90). |
| `image_size_height` | `int` | `224` | The height of camera image observations. |
| `image_size_width` | `int` | `224` | The width of camera image observations. |
| `require_depth` | `bool` | `True` | Whether to include metric depth maps in observations. |
| `require_point_cloud` | `bool` | `False` | Whether to generate point clouds from depth. Requires `require_depth=True` and `open3d`. |
| `num_points` | `int` | `8192` | Number of points in the downsampled point cloud. Only used when `require_point_cloud=True`. |
| `camera_names` | `list[str]` | `["agentview", "robot0_eye_in_hand"]` | A list of camera names to include in observations. |
| `max_episode_steps` | `int` | `600` | Maximum number of steps before the episode is truncated. |
| `seed` | `int` | `0` | A seed for the environment's random number generators. |
| `enable_pytorch3d_fps` | `bool` | `False` | Use pytorch3d's farthest point sampling instead of open3d's. Requires `pytorch3d`. |
| `pointcloud_process_device` | `str` | `"cpu"` | Device for pytorch3d FPS computation (e.g., `"cpu"`, `"cuda"`). Only used when `enable_pytorch3d_fps=True`. |

**Observation Dictionary:**

The observation returned by `reset()` and `step()` is a dictionary. For each camera name in `camera_names`, the following keys are available:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `{camera}_image` | `(3, H, W)` | `uint8` | RGB image, flipped to correct orientation. |
| `{camera}_depth` | `(H, W)` | `float32` | Metric depth map in meters (clipped to [0, 2.0]). NaN values are replaced with 0. Only present when `require_depth=True`. |
| `{camera}_intrinsic_matrix` | `(3, 3)` | `float64` | Camera intrinsic matrix. |
| `{camera}_extrinsic_matrix` | `(4, 4)` | `float64` | Camera extrinsic matrix. |
| `{camera}_pointcloud` | `(num_points, 3)` | `float64` | Point cloud in world coordinates. Only present when `require_point_cloud=True`. |

Additional keys from the underlying LIBERO environment (e.g., robot joint states) are also included.

**Properties and Methods:**

- `env.unwrapped.task_description` -- The natural language task description string.
- `env.unwrapped.num_init_states` -- Number of available initial states for the current task.
- `env.unwrapped.check_success()` -- Returns whether the task is currently in a success state.
- `env.reset(seed=None, options=None)` -- Reset the environment. Pass `options={"init_state_id": int}` to specify an initial state.
- `env.step(action)` -- Step with a 7-dim action `[dx, dy, dz, droll, dpitch, dyaw, gripper]`. Returns `(obs, reward, terminated, truncated, info)`.

## Remark

Some users may face the error:

```
UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray._reconstruct was not an allowed global by default. Please use `torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])` or the `torch.serialization.safe_globals([numpy.core.multiarray._reconstruct])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
```

You can find the code in `libero/libero/benchmark/__init__.py`:
```python
init_states = torch.load(init_states_path)
```

and change it to:
```python
init_states = torch.load(init_states_path, weights_only=False)
```

## Thanks

This project is updated from

```
git@github.com:CleanDiffuserTeam/CleanDiffuser.git -b lightning
```
