import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.pdqn import PDQNAgent
from agents.utils import hard_update_target_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=(100,), action_input_layer=0,
                 output_layer_init_std=None, activation="relu", init_type="kaiming", init_std=0.01):
        '''
        state_size: 环境观察的维度，而在SplitQActor中传入的是经过特征提取后的维度
        action_size: 离散动作的个数 在SplitQActor中传入的是1，表示只处理一个离散动作的信息
        action_parameter_size: 连续动作参数的维度 在SplitQActor这里传入的对应离散动作的连续动作的维度
        hidden_layers: 隐藏层结构
        action_input_layer: 从第几个隐藏层开始输入连续动作参数（这个参数在这个类中无用）
        output_layer_init_std: 输出层初始化的标准差
        activation: 激活函数类型
        init_type: 初始化类型
        init_std: 如果初始化类型是normal，则指定标准差
        '''
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation

        # create layers
        self.layers = nn.ModuleList()
        # 组合特征提取后的环境观察和连续动作的维度进行特征提取
        inputSize = self.state_size + self.action_parameter_size
        lastHiddenLayerSize = inputSize
        if hidden_layers:  # non-empty
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))

        # initialise layer weights
        # 初始化权重
        for i in range(0, len(self.layers) - 1):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            elif init_type == "normal":
                nn.init.normal_(self.layers[i].weight.data, std=init_std)
            else:
                raise ValueError("Unknown init_type "+str(init_type))
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        # else:
        #     nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state, action_parameters):
        '''
        state: 在SplitQActor中这里传入的是特征提取后的环境观察
        action_paramters: 在SplitQActor中这里传入的是单个离散动作对应的连续动作

        最终返回的是预测的离散动作的Q值
        '''
        # implement forward
        negative_slope = 0.01
        x = torch.cat((state, action_parameters), dim=1)
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        Q = self.layers[-1](x)
        return Q


class SplitQActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size_list, hidden_layers=(64, 32), action_input_layer=0,
                 output_layer_init_std=None, activation="relu", init_type="kaiming", init_std=None):
        '''
        state_size: 环境观察的维度  
        action_size: 离散动作的个数
        action_parameter_size_list: 每个离散动作对应的连续动作参数维度列表
        hidden_layers: 隐藏层结构
        action_input_layer: 从第几个隐藏层开始输入连续动作参数
        output_layer_init_std: 输出层初始化的标准差
        activation: 激活函数类型
        init_type: 初始化类型
        init_std: 如果初始化类型是normal，则指定标准差
        '''
        super(SplitQActor, self).__init__()
        self.state_size = state_size

        assert len(action_parameter_size_list) == action_size
        assert len(hidden_layers) >= action_input_layer#+1
        self.action_size = action_size
        self.action_parameter_size_list = np.array(action_parameter_size_list, dtype=int)
        self.activation = activation
        if init_type == "normal":
            assert init_std is not None and init_std > 0

        # shared hidden (feature) layers
        shared_layer_sizes = hidden_layers[:action_input_layer]
        split_layer_sizes = hidden_layers[action_input_layer:]

        # create shared layers
        self.shared_layers = nn.ModuleList()
        inputSize = self.state_size # 于其他的不同之处在于这里输入的只有环境观察
        lastSharedLayerSize = inputSize # 存储上一层的输出维度
        # 构建共同的特征提取层
        if len(shared_layer_sizes) > 0:
            n = len(shared_layer_sizes)
            self.shared_layers.append(nn.Linear(inputSize, shared_layer_sizes[0]))
            for i in range(1, n):
                self.shared_layers.append(nn.Linear(shared_layer_sizes[i - 1], shared_layer_sizes[i]))
            lastSharedLayerSize = shared_layer_sizes[-1]

        # create separate network for each action
        self.split_networks = nn.ModuleList()
        # 这里应该是构建每一个离散动作+连续动作的独立处理链路
        for k in range(self.action_size): # 遍历每一个离散动作
            self.split_networks.append(QNetwork(lastSharedLayerSize, 1, action_parameter_size_list[k],
                                                hidden_layers=split_layer_sizes, output_layer_init_std=output_layer_init_std,
                                                activation=activation))

        # initialise layer weights
        # 初始化共享层的权重
        for layer in self.shared_layers:
            if init_type == "kaiming":
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity=activation)
            elif init_type == "normal":
                nn.init.normal_(layer.weight.data, std=init_std)
            else:
                raise ValueError("Unknown init_type "+str(init_type))
            nn.init.zeros_(layer.bias)

        # 这里依旧是构建每个连续动作参数在整体参数中的偏移位置
        self.offsets = self.action_parameter_size_list.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)

        print(self)

    def forward(self, state, action_parameters):
        '''
        state: 环境观察， shape (batch_size, state_size)    
        action_parameters: 所有连续动作参数 shape (batch_size, total_action_parameter_size)
        '''
        # implement forward
        negative_slope = 0.01 # 如果使用的leak_relu的激活函数则会使用到本参数
        x = state
        num_layers = len(self.shared_layers)
        # 首先共享层提取观察信息
        for i in range(0, num_layers):
            if self.activation == "relu":
                x = F.relu(self.shared_layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.shared_layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        output = []
        # 其次对每一个离散动作和连续动作的值进行特征提取
        for a in range(self.action_size):
            # action_parameters[:, self.offsets[a]:self.offsets[a+1]]： 从连续动作中提取对应的离散动作的连续动作和环境组合进行特征提取进行Q值估计
            Qa = self.split_networks[a](x, action_parameters[:, self.offsets[a]:self.offsets[a+1]])
            output.append(Qa)
        # 最后将每一个离散动作对应连续动作的Q值都组合起来得到每一个离散动作的Q值组合
        Q = torch.cat(output, dim=1)
        return Q


class SplitPDQNAgent(PDQNAgent):
    NAME = "Split P-DQN N-Step Agent 这里应该也是另一个PDQN的变体，离散动作预测的方法有改变"

    def __init__(self,
                 *args,
                 **kwargs):
        super(SplitPDQNAgent, self).__init__(*args, **kwargs)
        self.actor = SplitQActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_sizes,
                                 **kwargs['actor_kwargs']).to(device)
        self.actor_target = SplitQActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_sizes,
                                        **kwargs['actor_kwargs']).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
