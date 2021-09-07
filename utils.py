from typing import NamedTuple, Callable, Union
import numpy as np


class RewardWeights(NamedTuple):
    quality_weight: float = 1
    rebuf_weight: float = 100
    quality_change_weight: float = 1

# defined initially in https://arxiv.org/abs/1805.11593
def sqrt_norm_function(reward: float, min_reward: float = 0, 
        max_reward: float = 5) -> float:
    reward = np.sign(reward) * ((np.sqrt(np.abs(reward) + 1) - 1) + \
                                                        0.001 * reward)
    reward = np.clip(reward, min_reward, max_reward)
    return reward

def make_reward_function(weights: RewardWeights= RewardWeights(), 
            norm_function: Callable= sqrt_norm_function) -> Callable:
    def reward_function(quality, rebuf, quality_change) -> float:
        reward = weights.quality_weight * quality - \
            weights.rebuf_weight * rebuf - \
            weights.quality_change_weight * quality_change
        reward = norm_function(reward)
        return reward
    return reward_function

def normalize(raw: Union[float, np.ndarray], 
        max: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    norm = raw / max
    norm = np.clip(norm, 0, 1)
    return norm

# taken partly from https://github.com/StanfordSNR/puffer/blob/master/src/scripts/helpers.py
def ssim_index_to_db(ssim_index: float) -> float:
    if not 0 < ssim_index < 1: # avoid division by 0
        return 0 
    return -10 * np.log10(1 - ssim_index)

