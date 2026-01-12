# Libero Web Wrapper (Server + Client)

An HTTP layer over the LIBERO manipulation benchmark, plus a lightweight Python client. The server exposes Gymnasium-compatible LIBERO environments over REST; the client hides the HTTP and serialization details so you can `.reset()` / `.step()` remotely as if the env were local.

## Components
- `liberogymwrapper` (server): registers LIBERO task suites with Gymnasium and serves them via Flask.
- `liberogymwrapperclient` (client): small wrapper around the HTTP API that transparently decodes observations and infos.

## Requirements
- Server: Python 3.12.12, CUDA-capable GPU recommended, system deps for MuJoCo/robosuite/Open3D. Main Python deps: torch 2.9.1, libero fork (gymnasium-based), open3d, pytorch3d 0.7.9, flask.
- Client: Python >=3.9 (tested), only needs requests/numpy/opencv/Pillow.

## Install
### Notes
- Server and client can live in separate Python environments. The server has heavier native deps; isolate it.

### Server (recommended, uv)
```bash
# From repository root
cd LiberoWebWrapper/LiberoGymWrapper
uv venv .venv && source .venv/bin/activate
uv pip install -e . --compile
```

### Client (lighter)
```bash
cd LiberoWebWrapper/LiberoGymWrapperClient
pip install -e .
```

## Run the server
```bash
cd LiberoWebWrapper/LiberoGymWrapper
python -m liberogymwrapper.http_server --listen 0.0.0.0 --port 40004 --log_level INFO
```
Notes:
- Importing `liberogymwrapper` registers env IDs with Gymnasium: `libero-10-v0`, `libero-90-v0`, `libero-goal-v0`, `libero-object-v0`, `libero-spatial-v0`.
- Responses encode observations/infos as `pickle -> zlib -> base64`; trust the server before decoding.
- Torch is patched to load weights with `weights_only=False` by default to avoid PyTorch 2.6 safety change.

## Run the clinent

### Code
```python
from liberogymwrapperclient import LiberoGymWrapperClient

client = LiberoGymWrapperClient(base_url="http://localhost:40004")
client, iid = client.create_env(
	env_name="libero-10-v0",
	task_id=0,
	image_size_height=224,
	image_size_width=224,
	camera_names=["agentview", "robot0_eye_in_hand"],
	require_depth=True,
	require_point_cloud=False,
)

obs, info = client.reset(seed=0)
done = False
while not done:
	action = client.sample_action()
	obs, reward, terminated, truncated, info = client.step(action)
	done = terminated or truncated

client.delete()  # optional; context manager also works
```

### More infomation about the `obs`:
#### 机器人本体状态 (Robot Proprio-state)

| 键名                   | 形状 | 说明                     |
| ---------------------- | ---- | ------------------------ |
| `robot0_joint_pos`​     | (7,) | 7个关节的位置（弧度）    |
| `robot0_joint_pos_cos`​ | (7,) | 关节位置的余弦值         |
| `robot0_joint_pos_sin`​ | (7,) | 关节位置的正弦值         |
| `robot0_joint_vel`​     | (7,) | 7个关节的速度            |
| `robot0_eef_pos`​       | (3,) | 末端执行器位置 (x, y, z) |
| `robot0_eef_quat`​      | (4,) | 末端执行器姿态（四元数） |
| `robot0_gripper_qpos`​  | (2,) | 夹爪位置（左/右指）      |
| `robot0_gripper_qvel`​  | (2,) | 夹爪速度                 |

#### 视觉观测 (Visual Observations)

##### 相机 1: `agentview`（第三人称视角）

| 键名                         | 形状            | 类型    | 说明              |
| ---------------------------- | --------------- | ------- | ----------------- |
| `agentview_image`​            | (3, 1080, 1920) | uint8   | RGB图像 (CHW格式) |
| `agentview_depth`​            | (1080, 1920)    | float32 | 深度图（米）      |
| `agentview_pointcloud`​       | (8192, 3)       | float64 | 点云 (x, y, z)    |
| `agentview_extrinsic_matrix`​ | (3,3)           | float64 | 外参矩阵          |
| `agentview_intrinsic_matrix`​ | (3,3)           | float64 | 内参矩阵          |

> 在 LIBERO 常见配置里：
>
> - `agentview_image`：`agentview` 相机的 RGB
> - `agentview_depth`：同一个 `agentview` 相机的 depth
> - `agentview_pointcloud`：由同一个 `agentview` 相机的 depth 反投影（或同等方式）得到的点云
>
> 所以它们的 **内参/外参一致**（同一台相机）。  
> 只有在你配置了不同 camera（比如 `agentview` vs `robot0_eye_in_hand`）时，才会不同。

##### 相机 2: `robot0_eye_in_hand`（手眼相机）

| 键名                                  | 形状            | 类型    | 说明     |
| ------------------------------------- | --------------- | ------- | -------- |
| `robot0_eye_in_hand_image`​            | (3, 1080, 1920) | uint8   | RGB图像  |
| `robot0_eye_in_hand_depth`​            | (1080, 1920)    | float32 | 深度图   |
| `robot0_eye_in_hand_pointcloud`​       | (8192, 3)       | float64 | 点云     |
| `robot0_eye_in_hand_extrinsic_matrix`​ | (3,3)           | float64 | 外参矩阵 |
| `robot0_eye_in_hand_intrinsic_matrix`​ | (3,3)           | float64 | 内参矩阵 |

> `eye_in_hand` 是机器人领域的标准术语，指安装在末端执行器（End-Effector）上的相机

#### 物体状态 (Object States)

场景中有每**个物体**都包含 4 个属性：

| 物体名            | `*_pos`(3,)  | `*_quat`(4,) | `*_to_robot0_eef_pos`(3,) | `*_to_robot0_eef_quat`(4,) |
| ----------------- | ------------ | ------------ | ------------------------- | -------------------------- |
| `alphabet_soup_1`​ | 世界坐标位置 | 世界坐标姿态 | 相对于末端的位置          | 相对于末端的姿态           |

| 汇总键名       | 形状 | 说明                   |
| -------------- | ---- | ---------------------- |
| `object-state`​ |      | 所有物体状态的拼接向量 |

‍


## Some detail info
### REST API (server)
#### ListEnv
- Method: [GET] `/v1/envs/`
- Response 200: `{ all_envs: {<instance_id>: <env_id>, ...} }`

#### CreateEnv
- Method: POST /v1/envs/ (query params)
- Required: env_name (string)
- Optional: 
    - `task_id` (int, default 0), 
    - `image_size_width` (int, default 1920), 
    - `image_size_height` (int, default 1080), 
    - `require_depth` (bool, default true), 
    - `require_point_cloud` (bool, default false), 
    - `num_points` (int, default 8192), 
    - `camera_names` (comma string, default `agentview,robot0_eye_in_hand`), 
    - `max_episode_steps` (int, default 600), 
    - `seed` (int, default 0), 
    - `enable_pytorch3d_fps` (bool, default `false`), 
    - `pointcloud_process_device` (string, default `cpu`)
- Response 200: { instance_id: string }

#### DeleteEnv
- Method: [DELETE] `/v1/envs/{instance_id}/`
- Path: instance_id (string)
- Response 200: empty body

#### ResetEnv
- Method: POST /v1/envs/{instance_id}/reset/ (query seed)
- Path: instance_id (string)
- Query: seed (int, default 0)
- Response 200: { observation: encoded, info: passthrough }

#### Step
- Method: [POST] `/v1/envs/{instance_id}/step/`
- Path: instance_id (string)
- Body JSON: { action: array or scalar }
- Response 200: `{ observation: encoded, reward: float, terminated: bool, truncated: bool, info: encoded }`

#### SampleAction
- Method: [GET] `/v1/envs/{instance_id}/action_space/sample/`
- Path: instance_id (string)
- Response 200: `{ action: array or scalar }`

#### Encoding
- observation and info: pickle -> zlib -> base64; client already decodes.
- reward is cast to Python float.

### Troubleshooting
- PyTorch 2.6 `torch.load` safety error: server already sets `weights_only=False`; ensure you trust the checkpoint source.
- Point cloud: requires depth; if `require_point_cloud=True`, make sure Open3D and (optionally) PyTorch3D are available and GPU matches the wheel.
- Large obs: responses are base64-encoded pickles; avoid proxy middleware that rewrites bodies.
