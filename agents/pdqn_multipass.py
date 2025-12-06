import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.pdqn import PDQNAgent
from agents.utils import hard_update_target_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mp-dqn的动作价值网络，使用多次传递的方式计算所有动作的Q值
class MultiPassQActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size_list, hidden_layers=(100,),
                 output_layer_init_std=None, activation="relu", **kwargs):
        '''
        Docstring for __init__
        
        :param self: Description
        :param state_size: 环境状态的维度
        :param action_size: 离散动作的数量
        :param action_parameter_size_list: 连续动作参数的维度列表
        :param hidden_layers: 隐藏层的结构
        :param output_layer_init_std: 输出层的初始化标准差
        :param activation: 激活函数
        :param kwargs: 外部传入的其他参数
        '''
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size_list = action_parameter_size_list
        self.action_parameter_size = sum(action_parameter_size_list) # 这个依旧是存储所有连续动作参数的总维度
        self.activation = activation

        # create layers
        self.layers = nn.ModuleList()
        # 看来这里的输入是结合了状态和所有动作参数的维度，估计是要来个cat
        inputSize = self.state_size + self.action_parameter_size
        # 以下部分看父类即可
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))

        # initialise layer weights
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        # else:
        #     nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

        self.offsets = self.action_parameter_size_list.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)

    def forward(self, state, action_parameters):
        '''
        Docstring for forward
        
        :param self: Description
        :param state: 环境观察
        :param action_parameters: 所有连续动作参数
        '''
        # implement forward
        negative_slope = 0.01

        Q = []
        # duplicate inputs so we can process all actions in a single pass
        batch_size = state.shape[0]
        # with torch.no_grad():
        x = torch.cat((state, torch.zeros_like(action_parameters)), dim=1)
        x = x.repeat(self.action_size, 1)
        for a in range(self.action_size):
            x[a*batch_size:(a+1)*batch_size, self.state_size + self.offsets[a]: self.state_size + self.offsets[a+1]] \
                = action_parameters[:, self.offsets[a]:self.offsets[a+1]]

        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        Qall = self.layers[-1](x)

        # extract Q-values for each action
        for a in range(self.action_size):
            Qa = Qall[a*batch_size:(a+1)*batch_size, a]
            if len(Qa.shape) == 1:
                Qa = Qa.unsqueeze(1)
            Q.append(Qa)
        Q = torch.cat(Q, dim=1)
        return Q


class MultiPassPDQNAgent(PDQNAgent):
    NAME = "Multi-Pass P-DQN Agent"

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # 构建自己的动作策略网络，其余的部分和普通的PDQN一样
        self.actor = MultiPassQActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_sizes,
                                     **kwargs['actor_kwargs']).to(device)
        self.actor_target = MultiPassQActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_sizes,
                                            **kwargs['actor_kwargs']).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
