import numpy as np


class OrnsteinUhlenbeckActionNoise(object):
    """
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    Source: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/utils.py
    连续噪音采样器算法
    """

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2, random_machine=np.random):
        '''
        action_dim: 所有连续动作的维度总和
        mu: 噪音均值
        theta: 噪音的均值回复速度
        sigma: 噪音的标准差
        random_machine: 随机数生成器，默认为numpy的随机模块
        '''
        super(OrnsteinUhlenbeckActionNoise, self).__init__()
        self.random = random_machine
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu # 初始化噪音状态为均值，shape 为 (action_dim,)

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * self.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X
