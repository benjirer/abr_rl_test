import torch as th
from torch import nn
from torch.distributions import Categorical


class DeepRlModel(nn.Module):
    """
        Conv-1D extractor for ABR with individual heads,
        similar in part to https://github.com/hongzimao/pensieve/blob/master/sim/a3c.py
    """

    def __init__(self, look_ahead_horizon: int = 5, n_actions: int = 10,
                feature_dim: int = 897):
        super(DeepRlModel, self).__init__()
        self.quality_net = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        self.buffer_net = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        self.throughput_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(), nn.Conv1d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(), nn.Flatten())
        self.delay_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(), nn.Conv1d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(), nn.Flatten())
        self.video_size_net = nn.Sequential(
            nn.Conv1d(look_ahead_horizon, 32, kernel_size=4, stride=1),
            nn.ReLU(), nn.Conv1d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(), nn.Flatten())
        self.index_net = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        self.ssim_net = nn.Sequential(
            nn.Conv1d(look_ahead_horizon, 32, kernel_size=4, stride=1),
            nn.ReLU(), nn.Conv1d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(), nn.Flatten())
        self.reward_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(), nn.Conv1d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(), nn.Flatten())
        self.policy_net = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU())
        self.action_net = nn.Linear(128, n_actions)

    @th.no_grad() # ideally, this should be in a wrapper  
    def forward(self, observation: dict) -> int:
        quality_out = self.quality_net(observation["past_quality"])
        buffer_out = self.buffer_net(observation["buffer"])
        throughput_out = self.throughput_net(observation["throughputs"])
        delay_out = self.delay_net(observation["delays"])
        video_out = self.video_size_net(observation["video_sizes"])
        index_out = self.index_net(observation["video_index"])
        ssim_out = self.ssim_net(observation["ssim_dbs"])
        reward_out = self.reward_net(observation["rewards"])
        features = th.cat((observation["past_quality"], quality_out, buffer_out,
            throughput_out, delay_out, video_out, index_out,
            ssim_out, reward_out), dim=1)
        action_logits = self.action_net(self.policy_net(features))
        dist = Categorical(logits=action_logits)
        action = th.argmax(dist.probs, dim=1)
        return action.item()
