import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.pdqn import PDQNAgent
from agents.utils import hard_update_target_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiPassQActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size_list, hidden_layers=(100,),
                 output_layer_init_std=None, activation="relu", **kwargs):
        '''
        Docstring for __init__
        这个类实现了MP-DQN算法中的Q-Actor网络结构
        
        :param self: Description
        :param state_size: 观察空间维度
        :param action_size: 离散动作数量
        :param action_parameter_size_list: 连续动作参数维度列表
        :param hidden_layers: 隐藏层神经元数量列表，用来控制隐藏层的维度
        :param output_layer_init_std: 输出层权重初始化的标准差，这边没有指定则使用默认初始化
        :param activation: 激活函数类型
        :param kwargs: Description 这个类中无用
        '''

        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size_list = action_parameter_size_list
        self.action_parameter_size = sum(action_parameter_size_list)
        self.activation = activation

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size # 看来模型的输入是环境观察和执行的动作
        lastHiddenLayerSize = inputSize
        # 特征提取使用功能的是全连接层
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))

        # initialise layer weights
        # 初始化权重
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            # 如果输出层有单独指定则进行单独指定，没有则统一使用前面的初始化方法
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        # else:
        #     nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias) # 偏置统一使用的是0初始化 

        # 这里估计是一个累加和操作，用来计算每个连续动作参数在整体参数中的偏移位置
        self.offsets = self.action_parameter_size_list.cumsum()
        # 插入一个0到最前面，方便后续索引
        self.offsets = np.insert(self.offsets, 0, 0)

    def forward(self, state, action_parameters):
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
        self.actor = MultiPassQActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_sizes,
                                     **kwargs['actor_kwargs']).to(device)
        self.actor_target = MultiPassQActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_sizes,
                                            **kwargs['actor_kwargs']).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
