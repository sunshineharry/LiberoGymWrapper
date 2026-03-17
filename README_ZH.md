# LIBERO 的 Gymnasium 封装器

本项目为 [LIBERO](https://github.com/sunshineharry/lerobot-libero) 机器人基准测试提供了一个封装器，使其兼容现代 [Gymnasium](https://gymnasium.farama.org/) API。

通过该封装器，你可以使用标准的 `gymnasium.make()` 接口来创建、重置和运行 LIBERO 环境。

## 依赖

使用前请确保安装了以下依赖：

- `gymnasium>=0.29.1`

- `libero @ git+https://github.com/sunshineharry/lerobot-libero.git#egg=libero`

- `torch`

可选依赖（用于点云支持）：

- `open3d`（当 `require_point_cloud=True` 时需要）

- `pytorch3d`（当 `enable_pytorch3d_fps=True` 时需要）

可以在项目根目录下运行 `pip install .` 来安装本封装器及其依赖。

## 使用方法

使用时需要先在脚本中导入 `liberogymwrapper`，这会自动将 LIBERO 环境注册到 Gymnasium 框架中。

**示例 1：基本初始化与数据访问**

本示例展示如何创建环境、重置并访问初始观测和任务描述。

```python
import gymnasium as gym
import liberogymwrapper  # 1. 导入封装器以注册环境
import numpy as np
from tqdm import tqdm

print("遍历 'libero-10-v0' 任务...")

# 遍历 'libero-10-v0' 中的全部 10 个任务
for i in tqdm(range(10)):

    # 2. 使用 gym.make() 创建环境
    env = gym.make(
        "libero-10-v0",                # 任务套件名称
        task_id=i,                     # 要加载的任务 ID（0-9）
        image_size_height=224,         # 图像高度
        image_size_width=224,          # 图像宽度
        require_depth=True,            # 包含深度观测
        camera_names=["agentview"],    # 相机列表
        seed=0,
    )

    # 3. 重置环境，获取初始观测
    # obs 是一个字典，包含机器人状态和相机图像
    obs, info = env.reset()

    # 4. 获取自然语言任务描述
    prompt = env.unwrapped.task_description

    # 5. 从观测字典中获取数据
    # 图像格式为 (C, H, W)，RGB 通道
    agentview_image = obs["agentview_image"]           # 形状: (3, 224, 224)
    agentview_depth = obs["agentview_depth"]           # 形状: (224, 224)
    agentview_intrinsic = obs["agentview_intrinsic_matrix"]  # 形状: (3, 3)
    agentview_extrinsic = obs["agentview_extrinsic_matrix"]  # 形状: (4, 4)

    print(f"\n[任务 ID {i}]")
    print(f"  描述: {prompt}")
    print(f"  图像形状: {agentview_image.shape}")
    print(f"  深度形状: {agentview_depth.shape}")

    # 6. 使用完毕后关闭环境
    env.close()

print("\n所有任务遍历完成。")
```

**示例 2：基本环境交互循环**

本示例展示标准的 Gymnasium reset-step 循环，演示如何向环境发送动作并接收新的观测，直到回合结束。

示例中使用 `env.action_space.sample()` 生成随机动作。实际应用中应替换为你自己的策略。

```python
import gymnasium as gym
import liberogymwrapper
import numpy as np
import time

# 1. 创建单个环境
print("创建 'libero-goal-v0' 环境...")
env = gym.make(
    "libero-goal-v0",              # 使用 "goal" 任务套件
    task_id=5,                     # 加载第 5 个任务
    image_size_height=224,         # 图像高度
    image_size_width=224,          # 图像宽度
    require_depth=True,            # 包含深度观测
    camera_names=["agentview", "robot0_eye_in_hand"],  # 使用多个相机
    max_episode_steps=600,         # 每回合最大步数
    seed=0,
)

# 2. 重置环境
obs, info = env.reset()
print(f"开始任务: {env.unwrapped.task_description}")

# --- 3.（可选）稳定环境 ---
# 在仿真中，通常需要让物体先稳定下来再开始控制循环
print("稳定环境中...")
dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])  # [dx, dy, dz, droll, dpitch, dyaw, gripper]
for _ in range(20):
    obs, _, _, _, _ = env.step(dummy_action)
# --- 稳定结束 ---

# 4. 运行主控制循环
print("开始主控制循环...")
done = False
step_count = 0
start_time = time.time()

while not done:
    # 5. 获取动作
    # 请替换为你的策略动作
    # action = my_policy.get_action(obs, env.unwrapped.task_description)
    action = env.action_space.sample()

    # 6. 执行动作
    obs, reward, terminated, truncated, info = env.step(action)

    # 7. 检查回合是否结束
    done = terminated or truncated
    step_count += 1

end_time = time.time()
print(f"\n回合结束，耗时 {end_time - start_time:.2f} 秒。")
print(f"总步数: {step_count}")

# 8. 关闭环境
env.close()
```

## 环境参考

**可用的环境 ID：**

本封装器注册了以下环境 ID：

- `libero-10-v0`（10 个任务的子集）
- `libero-90-v0`（完整的 90 个任务）
- `libero-goal-v0`
- `libero-object-v0`
- `libero-spatial-v0`

**`gym.make()` 关键字参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `task_id` | `int` | `0` | 要加载的任务 ID（大多数套件为 0-9，libero-90 为 0-89）。 |
| `image_size_height` | `int` | `224` | 相机图像观测的高度。 |
| `image_size_width` | `int` | `224` | 相机图像观测的宽度。 |
| `require_depth` | `bool` | `True` | 是否在观测中包含度量深度图。 |
| `require_point_cloud` | `bool` | `False` | 是否从深度图生成点云。需要 `require_depth=True` 和 `open3d`。 |
| `num_points` | `int` | `8192` | 下采样后的点云点数。仅在 `require_point_cloud=True` 时使用。 |
| `camera_names` | `list[str]` | `["agentview", "robot0_eye_in_hand"]` | 要包含在观测中的相机名称列表。 |
| `max_episode_steps` | `int` | `600` | 回合被截断前的最大步数。 |
| `seed` | `int` | `0` | 环境随机数生成器的种子。 |
| `enable_pytorch3d_fps` | `bool` | `False` | 使用 pytorch3d 的最远点采样代替 open3d。需要安装 `pytorch3d`。 |
| `pointcloud_process_device` | `str` | `"cpu"` | pytorch3d FPS 计算使用的设备（如 `"cpu"`、`"cuda"`）。仅在 `enable_pytorch3d_fps=True` 时使用。 |

**观测字典：**

`reset()` 和 `step()` 返回的观测是一个字典。对于 `camera_names` 中的每个相机名称，包含以下键：

| 键 | 形状 | 数据类型 | 说明 |
|---|---|---|---|
| `{camera}_image` | `(3, H, W)` | `uint8` | RGB 图像，已翻转为正确方向。 |
| `{camera}_depth` | `(H, W)` | `float32` | 度量深度图，单位为米（裁剪到 [0, 2.0]）。NaN 值已替换为 0。仅在 `require_depth=True` 时存在。 |
| `{camera}_intrinsic_matrix` | `(3, 3)` | `float64` | 相机内参矩阵。 |
| `{camera}_extrinsic_matrix` | `(4, 4)` | `float64` | 相机外参矩阵。 |
| `{camera}_pointcloud` | `(num_points, 3)` | `float64` | 世界坐标系下的点云。仅在 `require_point_cloud=True` 时存在。 |

底层 LIBERO 环境的其他键（如机器人关节状态）也会包含在内。

**属性和方法：**

- `env.unwrapped.task_description` -- 自然语言任务描述字符串。
- `env.unwrapped.num_init_states` -- 当前任务可用的初始状态数量。
- `env.unwrapped.check_success()` -- 返回任务当前是否处于成功状态。
- `env.reset(seed=None, options=None)` -- 重置环境。传入 `options={"init_state_id": int}` 可指定初始状态。
- `env.step(action)` -- 执行 7 维动作 `[dx, dy, dz, droll, dpitch, dyaw, gripper]`。返回 `(obs, reward, terminated, truncated, info)`。

## 备注

部分用户可能会遇到以下错误：

```
UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray._reconstruct was not an allowed global by default. Please use `torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])` or the `torch.serialization.safe_globals([numpy.core.multiarray._reconstruct])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
```

可以在 `libero/libero/benchmark/__init__.py` 中找到以下代码：
```python
init_states = torch.load(init_states_path)
```

将其修改为：
```python
init_states = torch.load(init_states_path, weights_only=False)
```

## 致谢

本项目基于以下项目更新：

```
git@github.com:CleanDiffuserTeam/CleanDiffuser.git -b lightning
```
