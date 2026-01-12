#!/usr/bin/env python3
import sys
import uuid
import argparse
import logging

from flask import Flask, request, jsonify
import base64
import json
import zlib
import pickle

import numpy as np
import gymnasium as gym
import liberogymwrapper

logger = logging.getLogger("libero_gym_wrapper.http_server")


# Torch 补丁
import torch
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # 如果没有明确指定 weights_only，就设为 False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load


def _anything_to_json(obs):
    """使用 pickle 序列化 observation"""
    # pickle 序列化
    pickled = pickle.dumps(obs, protocol=pickle.HIGHEST_PROTOCOL)
    # 压缩
    compressed = zlib.compress(pickled, level=6)
    # base64 编码（使 JSON 兼容）
    b64_encoded = base64.b64encode(compressed).decode('ascii')
    return {
        "type": "pickle",
        "data": b64_encoded
    }


def _unjson(jsonable):
    """反序列化 observation"""
    # base64 解码
    compressed = base64.b64decode(jsonable["data"].encode('ascii'))
    # 解压
    pickled = zlib.decompress(compressed)
    # pickle 反序列化
    obs = pickle.loads(pickled)
    return obs


########## Container for environments ##########
class Envs:
    """
    Container and manager for the environments instantiated
    on this server.

    When a new environment is created, such as with
    envs.create('CartPole-v1'), it is stored under a short
    identifier (such as '3c657dbc'). Future API calls make
    use of this instance_id to identify which environment
    should be manipulated.
    """

    def __init__(self):
        self.envs = {}
        self.id_len = 8

    def _lookup_env(self, instance_id):
        try:
            return self.envs[instance_id]
        except KeyError as e:
            raise InvalidUsage(f"Instance_id {instance_id} unknown") from e

    def _remove_env(self, instance_id):
        try:
            del self.envs[instance_id]
        except KeyError as e:
            raise InvalidUsage(f"Instance_id {instance_id} unknown") from e

    def _add_alpha_channel(self, rf):
        if isinstance(rf, np.ndarray) and rf.dtype == np.uint8:
            return np.dstack((rf, np.full((rf.shape[0], rf.shape[1]), 255, dtype=np.uint8)))
        return rf


    def create(
        self,
        env_name,
        task_id:                int = 0,
        image_size_height:      int = 1080,
        image_size_width:       int = 1920,
        require_depth:          bool = True,
        require_point_cloud:    bool = False,
        num_points:             int = 8192,
        camera_names:           list = ["agentview", "robot0_eye_in_hand"],
        max_episode_steps:      int = 600,
        seed:                   int = 0,
        enable_pytorch3d_fps:   bool = False,
        pointcloud_process_device: str = "cpu",
    ):
        if camera_names is None:
            camera_names = ["agentview", "robot0_eye_in_hand"]
        
        try:
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
        except gym.error.Error as e:
            raise InvalidUsage(f"Attempted to look up malformed environment ID '{env_name}'") from e

        instance_id = str(uuid.uuid4().hex)[: self.id_len]
        self.envs[instance_id] = env
        return instance_id

    def list_all(self):
        return {instance_id: env.spec.id for (instance_id, env) in self.envs.items()}

    def reset(self, instance_id, seed):
        env = self._lookup_env(instance_id)
        seed = int(seed) if seed is not None else None
        obs = env.reset(seed=seed)
        return obs

    def get_id(self, instance_id):
        _id = self._lookup_env(instance_id).spec.id
        return _id


    def step(self, instance_id, action):
        env = self._lookup_env(instance_id)
        if isinstance(action, int):
            nice_action = action
        else:
            nice_action = np.array(action)
        observation, reward, terminated, truncated, info = env.step(nice_action)
        return [observation, reward, terminated, truncated, info]


    def get_action_space_sample(self, instance_id):
        env = self._lookup_env(instance_id)
        action = env.action_space.sample()
        if isinstance(action, (list, tuple)) or ("numpy" in str(type(action))):
            try:
                action = action.tolist()
            except TypeError:
                print(type(action))
                print("TypeError")
        return action

    def env_close(self, instance_id):
        env = self._lookup_env(instance_id)
        env.close()
        self._remove_env(instance_id)


########## App setup ##########
app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
envs = Envs()


########## Error handling ##########
class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


def get_required_param(json_, param):
    if json_ is None:
        logger.info("Request is not a valid json")
        raise InvalidUsage("Request is not a valid json")
    value = json_.get(param, None)
    if (value is None) or (value == "") or (value == []):
        logger.info("A required request parameter '%s' had value %s", param, value)
        raise InvalidUsage(f"A required request parameter '{param}' was not provided")
    return value


def get_optional_param(json_, param, default):
    if json_ is None:
        logger.info("Request is not a valid json")
        raise InvalidUsage("Request is not a valid json")
    value = json_.get(param, None)
    if (value is None) or (value == "") or (value == []):
        logger.info(
            "An optional request parameter '%s' had value %s and was replaced with default value %s",
            param,
            value,
            default,
        )
        value = default
    return value


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


########## API route definitions ##########
########## API 路由定义 ##########

@app.route("/v1/envs/", methods=["POST"])
def env_create():
    """
    创建环境实例，所有参数从 URL Query 参数中获取
    
    Args:
        env_id (str): 环境名称（必需）
        task_id (int): 任务 ID，默认 0
        image_size_height (int): 图像高度，默认 1080
        image_size_width (int): 图像宽度，默认 1920
        require_depth (bool): 是否需要深度图，默认 true
        require_point_cloud (bool): 是否需要点云，默认 false
        num_points (int): 点云采样点数默认 true
        camera_names (str): 相机列表，逗号分隔，默认 "agentview,robot0_eye_in_hand"
        max_episode_steps (int): 最大步数，默认 600
        seed (int): 随机种子，默认 0
        enable_pytorch3d_fps (bool): 是否启用 PyTorch3D FPS，默认 false
        pointcloud_process_device (str): 点云处理设备，默认 "cpu"
    
    Returns:
        JSON: {"instance_id": "环境实例的唯一标识符"}
    
    Example:
        POST /v1/envs/?env_id=libero-90-v0&task_id=0&seed=42
    """
    args = request.args
    
    # 必需参数
    env_name = args.get("env_name", None)
    if env_name is None:
        raise InvalidUsage("缺少必需参数 'env_name'")
    
    # 可选参数（带类型转换）
    task_id = int(args.get("task_id", 0))
    image_size_height = int(args.get("image_size_height", 1080))
    image_size_width = int(args.get("image_size_width", 1920))
    require_depth = args.get("require_depth", "true").lower() == "true"
    require_point_cloud = args.get("require_point_cloud", "false").lower() == "true"
    num_points = int(args.get("num_points", 8192))
    max_episode_steps = int(args.get("max_episode_steps", 600))
    seed = int(args.get("seed", 0))
    enable_pytorch3d_fps = args.get("enable_pytorch3d_fps", "false").lower() == "true"
    pointcloud_process_device = args.get("pointcloud_process_device", "cpu")
    
    # camera_names: 逗号分隔的字符串转为列表
    camera_names_str = args.get("camera_names", "agentview,robot0_eye_in_hand")
    camera_names = [name.strip() for name in camera_names_str.split(",")]
    print(f"[info]创建环境: env_name={env_name}, task_id={task_id}, \n\t\timage={image_size_width}x{image_size_height}, depth={require_depth}, \n\t\tpointcloud={require_point_cloud}, num_points={num_points}, \n\t\tcameras={camera_names}, max_steps={max_episode_steps}, \n\t\tseed={seed}, fps={enable_pytorch3d_fps}, device={pointcloud_process_device}")
    
    instance_id = envs.create(
        env_name=env_name,
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
    
    return jsonify(instance_id=instance_id)


@app.route("/v1/envs/", methods=["GET"])
def env_list_all():
    """
    列出所有运行中的环境
    
    Returns:
        JSON: {"all_envs": {instance_id: env_id, ...}}
    """
    all_envs = envs.list_all()
    return jsonify(all_envs=all_envs)


@app.route("/v1/envs/<instance_id>/", methods=["GET"])
def env_get_id(instance_id):
    """
    获取环境的 spec ID
    
    Args:
        instance_id (str): 环境实例标识符
    
    Returns:
        JSON: {"id": "环境 ID，如 libero-90-v0"}
    """
    _id = envs.get_id(instance_id)
    return jsonify(id=_id)


@app.route("/v1/envs/<instance_id>/reset/", methods=["POST"])
def env_reset(instance_id):
    """
    重置环境并返回初始观测
    
    Args:
        instance_id (str): 环境实例标识符（URL 路径参数）
        seed (int, optional): 随机种子（Query 参数）
    
    Returns:
        JSON: {
            "observation": pickle + zlib 压缩 + base64 编码的观测数据,
            "info": pickle + zlib 压缩 + base64 编码的环境信息
        }
    
    Example:
        POST /v1/envs/3c657dbc/reset/?seed=42
    """
    args = request.args
    
    seed = args.get("seed", 0)
    if seed is not None:
        seed = int(seed)
    
    logger.info(f"重置环境: instance_id={instance_id}, seed={seed}")
    
    obs, info = envs.reset(instance_id, seed)
    
    obs_jsonable = _anything_to_json(obs)
    
    return jsonify(observation=obs_jsonable, info=info)



@app.route("/v1/envs/<instance_id>/step/", methods=["POST"])
def env_step(instance_id):
    """
    执行一步动作
    
    Args:
        instance_id (str): 环境实例标识符
        action (list): 动作向量
    
    Returns:
        JSON: {
            "observation": 压缩编码的观测,
            "reward": float 奖励值,
            "terminated": bool 是否终止,
            "truncated": bool 是否截断,
            "info": 压缩编码的附加信息
        }
    """
    json_data = request.get_json()
    action = get_required_param(json_data, "action")
    
    [obs, reward, terminated, truncated, info] = envs.step(instance_id, action)
    
    obs_jsonable = _anything_to_json(obs)
    info_jsonable = _anything_to_json(info)
    
    # reward 转为 Python float，避免 numpy 类型序列化问题
    if hasattr(reward, 'item'):
        reward = reward.item()
    
    return jsonify(
        observation=obs_jsonable,
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=info_jsonable
    )


@app.route("/v1/envs/<instance_id>/action_space/sample/", methods=["GET"])
def env_action_space_sample(instance_id):
    """
    从动作空间随机采样一个动作
    
    Args:
        instance_id (str): 环境实例标识符
    
    Returns:
        JSON: {"action": [随机采样的动作向量]}
    """
    action = envs.get_action_space_sample(instance_id)
    return jsonify(action=action)


@app.route("/v1/envs/<instance_id>/", methods=["DELETE"])
def env_close(instance_id):
    """
    关闭并删除环境实例
    
    Args:
        instance_id (str): 环境实例标识符
    
    Returns:
        HTTP 200 空响应
    """
    envs.env_close(instance_id)
    return ("", 200)


def run_main():
    parser = argparse.ArgumentParser(description="Start a Gym HTTP API server")
    parser.add_argument("-l", "--listen", help="interface to listen to", default="0.0.0.0")
    parser.add_argument("-p", "--port", default=40004, type=int, help="port to bind to")
    parser.add_argument("-g", "--log_level", default="ERROR", type=str, help="server log level")

    args = parser.parse_args()
    print(f"Server starting at:  http://{args.listen}:{args.port}. Loglevel: {args.log_level}.")
    logger.setLevel(args.log_level)
    app.run(host=args.listen, port=args.port)


if __name__ == "__main__":
    run_main()