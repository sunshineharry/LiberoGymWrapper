# A Gymnasium Wrapper for LIBERO
This project provides a wrapper to make the LIBERO robotics benchmark compatible with the old Gym API.

This wrapper allows you to create, reset, and step through LIBERO environments using the standard gym.make() interface.

## Dependencies
Before using this wrapper, ensure you have the required dependencies. The primary dependencies are:

- `libero`

- `torch`

You can typically install this wrapper and its dependencies by running `pip install .` from the project's root directory, assuming a valid pyproject.toml is present.

## Usage
To use the wrapper, you must first import liberogymwrapper in your script. This will automatically register the LIBERO environments with the Gymnasium framework.

**Example: Basic Initialization and Data Access**

This example shows how to create an environment, reset it to a specific task, and access the initial observation and task description.

```Python

import gym
import liberogymwrapper  # 1. Import wrapper to register environments
import numpy as np
from tqdm import tqdm

print("Iterating through 'libero-10-v0' tasks...")

# We can loop through all 10 tasks in the 'libero-10-v0' suite
for i in tqdm(range(10)):
    
    # 2. Create the environment using gym.make()
    # You must provide task_id, image_size, and camera_names
    env = gym.make(
        "libero-10-v0",             # Task suite name
        task_id=i,                  # The ID of the task to load (0-9)
        image_size=224,             # Image resolution (H and W)
        camera_names=["agentview"], # A list of cameras to use
        seed=0,
    )

    # 3. Reset the environment to get the first observation
    # obs is a dictionary containing robot state and camera images
    obs = env.reset()

    # 4. Access the natural language task description (prompt)
    prompt = env.task_description
    
    # 5. Access data from the observation dictionary
    # Images are in (C, H, W) format and are RGB
    agentview_image = obs["agentview_image"] # Shape: (3, 224, 224)

    print(f"\n[Task ID {i}]")
    print(f"  Prompt: {prompt}")
    print(f"  Image shape: {agentview_image.shape}")

    # 6. Always close the environment when done
    env.close()

print("\nAll tasks iterated successfully.")
```

**Example 2: Basic Environment Step Loop**

This example shows the standard Gymnasium reset-step loop. It demonstrates how to interact with the environment by sending actions and receiving new observations until the episode ends.

For demonstration, env.action_space.sample() is used to generate random actions. In a real application, you would replace this with your own policy.

```Python

import gym
import liberogymwrapper
import numpy as np
import time

# 1. Create a single environment
print("Creating 'libero-goal-v0' environment...")
env = gym.make(
    "libero-goal-v0",  # Use the "goal" task suite
    task_id=5,         # Load task #5
    image_size=224,
    camera_names=["agentview", "robot0_eye_in_hand"], # Use multiple cameras
    seed=0,
)

# 2. Reset the environment
obs = env.reset()
print(f"Starting task: {env.task_description}")

# --- 3. (Optional) Stabilize the Environment ---
# In simulation, it is often good practice to let objects
# settle before starting the control loop.
# We step a few times with a neutral action (e.g., open gripper).
print("Stabilizing environment...")
dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])  # [dx, dy, dz, droll, dpitch, dyaw, gripper]
for _ in range(20):
    obs, _, _, _ = env.step(dummy_action)
# --- End Stabilization ---


# 4. Run the main control loop
print("Starting main control loop...")
done = False
step_count = 0
start_time = time.time()

while not done:
    # 5. Get an action
    # REPLACE THIS with your policy's action
    # action = my_policy.get_action(obs, env.task_description)
    action = env.action_space.sample() 
    
    # 6. Step the environment with the action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 7. Check if the episode is finished
    done = terminated or truncated
    
    step_count += 1
    
    # (Optional) Add a safety break for long-running episodes
    if step_count > 1000:
        print("Reached max step limit.")
        break

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

When calling gym.make(), you must provide the following keyword arguments:

- `task_id (int)`: The specific task to load from the suite (e.g., 0 for the first task).

- `image_size (int)`: The width and height for all camera observations (e.g., 224).

- `camera_names (list[str])`: A list of camera names to include in the observation dictionary.

- Common names: "agentview", "robot0_eye_in_hand"

- `seed (int, optional)`: A seed for the environment's random number generators.

- `render_mode (str, optional)`: Set to "human" to open a window for live rendering.

## Remark

Some users may face the error:

```
UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray._reconstruct was not an allowed global by default. Please use `torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])` or the `torch.serialization.safe_globals([numpy.core.multiarray._reconstruct])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
```

You can find the code
```
File /opt/miniforge3/envs/pi0_torch/lib/python3.12/site-packages/libero/libero/benchmark/__init__.py:164, in Benchmark.get_task_init_states(self, i)
    158 def get_task_init_states(self, i):
    159     init_states_path = os.path.join(
    160         get_libero_path("init_states"),
    161         self.tasks[i].problem_folder,
    162         self.tasks[i].init_states_file,
    163     )
--> 164     init_states = torch.load(init_states_path)
    165     return init_states

```

and change the `init_states = torch.load(init_states_path)` to `init_states = torch.load(init_states_path, weights_only=False)`


## Thanks

This project is updated from 

```
git@github.com:CleanDiffuserTeam/CleanDiffuser.git -b lightning

```