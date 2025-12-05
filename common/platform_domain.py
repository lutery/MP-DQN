import numpy as np
import gym


class PlatformFlattenedActionWrapper(gym.ActionWrapper):
    """
    Changes the format of the parameterised action space to conform to that of Goal-v0 and Platform-v0
    将连续动作的每一个动作都提取到和离散动作同级，不放在space[1]下面
    """
    def __init__(self, env):
        super(PlatformFlattenedActionWrapper, self).__init__(env)
        old_as = env.action_space # 存储旧的动作空间
        num_actions = old_as.spaces[0].n # 获取离散动作数， space[0]对应离散动作
        # 构建新的动作空间,将连续动作每一个动作提取到和离散动作相同的维度，而不是在space[1]下面
        self.action_space = gym.spaces.Tuple((
            old_as.spaces[0],  # actions
            *(gym.spaces.Box(old_as.spaces[1].spaces[i].low, old_as.spaces[1].spaces[i].high, dtype=np.float32)
              for i in range(0, num_actions))
        ))

    def action(self, action):
        return action
