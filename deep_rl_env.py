from base_env import BaseEnv
from utils import make_reward_function, ssim_index_to_db, normalize
from typing import Callable, Union
import numpy as np
import torch as th
from collections import deque
from deep_rl_model import DeepRlModel
from dataclasses import dataclass

HISTORY_LEN = 10
DEFAULT_VIDEO_INDEX = 1.0

@dataclass
class Channel:
    name: str = ""
    chunks_sent: int = 0


class DeepRlEnv(BaseEnv):
    """
        Env for a model with individual heads
    """
    def setup_env(self, model_path: str) -> Callable:
        self.past_ssim_db = 0
        self.past_cum_rebuf = 0
        self.channel = Channel()
        self.throughput_history = deque(maxlen=HISTORY_LEN)
        self.delay_history = deque(maxlen=HISTORY_LEN)
        self.reward_history = deque(maxlen=HISTORY_LEN)
        self.reward_func = make_reward_function()
        self.max_video_size = 3 # Mb
        self.max_reward = 5
        self.max_quality = 30 # SSIM dB
        self.max_delay = 20 # s
        self.max_throughput = 25 # Mb/s
        self.max_buffer = 15 # s
        for _ in range(HISTORY_LEN):
            self.throughput_history.append(0)
            self.delay_history.append(0)
            self.reward_history.append(0)
        model = DeepRlModel()
        model.load_state_dict(th.load(model_path))
        model.eval()
        th.set_num_threads(1) # follow Pensieve code in third_party/
        return model

    def process_env_info(self, env_info: dict) -> dict:
        if self.channel.name == env_info["channel_name"]:
            self.channel.chunks_sent += 1
        else:
            self.channel.name = env_info["channel_name"]
            self.channel.chunks_sent = 0
        is_init = self.channel.chunks_sent <= 1 # first or second chunk

        if self.past_action is not None: # avoid doing extra work on zeros for very first action
            env_info["past_chunk"]["ssim_db"] = ssim_index_to_db(
                env_info["past_chunk"]["ssim"])
            if not is_init:
                rebuf = max(0, env_info["cum_rebuf"] - self.past_cum_rebuf)
                quality_change = abs(
                    self.past_ssim_db - env_info["past_chunk"]["ssim_db"])
            else:
                # first or second action in the environment
                rebuf = 0 # ignore start-up delay
                quality_change = 0 # quality change penalty for first action
            reward = self.reward_func(
                env_info["past_chunk"]["ssim_db"], rebuf, quality_change)
            self.past_ssim_db = env_info["past_chunk"]["ssim_db"]
            self.past_cum_rebuf = env_info["cum_rebuf"]
            self.delay_history.append(
                normalize(env_info["past_chunk"]["delay"], max=self.max_delay))
            env_info["past_chunk"]["delay"] = max(
                env_info["past_chunk"]["delay"], 1e-6) # avoid division by 0
            throughput = env_info["past_chunk"]["size"] / \
                env_info["past_chunk"]["delay"]
            self.throughput_history.append(
                normalize(throughput, max=self.max_throughput))
            self.reward_history.append(normalize(reward, max=self.max_reward))
        
        past_quality = normalize(self.past_ssim_db, max=self.max_quality)
        buffer = normalize(env_info["buffer"], max=self.max_buffer)
        video_sizes = normalize(
            np.array(env_info["sizes"]), max=self.max_video_size)
        ssim_dbs = [
            [ssim_index_to_db(ssim) for ssim in timestamp] 
            for timestamp in env_info["ssims"]]
        ssim_dbs = normalize(np.array(ssim_dbs), max=self.max_quality)

        obs = {"past_quality": [past_quality],
                "buffer": [buffer],
                "throughputs": np.array(self.throughput_history, ndmin=2),
                "delays": np.array(self.delay_history, ndmin=2),
                "video_sizes": video_sizes,
                "video_index": [DEFAULT_VIDEO_INDEX],
                "ssim_dbs": ssim_dbs,
                "rewards": np.array(self.reward_history, ndmin=2)}
        obs = {key: th.as_tensor(val, dtype=th.float32).unsqueeze_(0)
            for key, val in obs.items()}

        return obs

