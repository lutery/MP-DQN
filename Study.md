Using "Codebase"

Read [](file:///c%3A/Projects/OpenSource/python/MP-DQN/README.md)

Read [](file:///c%3A/Projects/OpenSource/python/MP-DQN/agents)

Read [](file:///c%3A/Projects/OpenSource/python/MP-DQN/agents/pdqn_multipass.py)

Read [](file:///c%3A/Projects/OpenSource/python/MP-DQN/agents/pdqn.py#1-1), lines 1 to 500

Read [](file:///c%3A/Projects/OpenSource/python/MP-DQN/agents/agent.py)

Read [](file:///c%3A/Projects/OpenSource/python/MP-DQN/agents/pdqn_multipass.py)

Read [](file:///c%3A/Projects/OpenSource/python/MP-DQN/run_platform_pdqn.py#1-1), lines 1 to 150

Searched for regex `MultiPassPDQNAgent|multipass`, 38 results

根据您的工作空间，我建议您按以下顺序阅读 MP-DQN 算法代码：

## 1. 从基类开始
首先看 agent.py，了解所有 Agent 的基本接口：
- `act()`: 选择动作
- `step()`: 学习步骤
- `start_episode()` / `end_episode()`: 回合管理

## 2. 核心算法 - P-DQN 基础
然后阅读 pdqn.py，这是 MP-DQN 的基类：

**关键组件：**
- **`QActor`** (第16行): Q 网络，输入状态和动作参数，输出每个离散动作的 Q 值
- **`ParamActor`** (第64行): 参数网络，输入状态，输出所有动作的连续参数
- **`PDQNAgent`** (第142行): 主算法类
  - `act()` (第306行): epsilon-greedy 动作选择
  - `_optimize_td_loss()` (第380行): 核心训练逻辑，包含 Q 网络更新和参数网络的梯度反转

## 3. MP-DQN 核心实现 ⭐
接着看 pdqn_multipass.py，这是 **MP-DQN 的关键创新**：

````python
class MultiPassQActor(nn.Module):
    # 关键：第 48-67 行的 forward() 方法
    def forward(self, state, action_parameters):
        # 1. 复制 batch，每个动作一份
        x = x.repeat(self.action_size, 1)
        
        # 2. 每个动作只填入对应的参数段
        for a in range(self.action_size):
            x[a*batch_size:(a+1)*batch_size, 
              self.state_size + self.offsets[a]: self.state_size + self.offsets[a+1]] \
                = action_parameters[:, self.offsets[a]:self.offsets[a+1]]
        
        # 3. 单次前向传播得到所有 Q(s,a)
        Qall = self.layers[-1](x)
````

**MP-DQN 解决的问题：**
- P-DQN 会把所有动作参数同时输入 Q 网络，导致"过参数化"问题
- MP-DQN 通过多个"pass"（批处理并行），每个动作只看到自己的参数，避免参数泄露

## 4. 运行脚本示例
最后看一个运行脚本，例如 run_platform_pdqn.py (第117-123行)：

````python
from agents.pdqn_multipass import MultiPassPDQNAgent

# 第122-123行：选择 MP-DQN
elif multipass:
    agent_class = MultiPassPDQNAgent
````

## 推荐阅读顺序总结

1. **agent.py** - 理解接口
2. **pdqn.py** - 理解 P-DQN 基础（重点：`QActor`, `ParamActor`, `_optimize_td_loss()`）
3. **pdqn_multipass.py** - 理解 MP-DQN 创新（重点：`MultiPassQActor.forward()`）
4. **run_platform_pdqn.py** - 理解如何运行

## 快速测试

在 PowerShell 中运行：
```powershell
cd c:\Projects\OpenSource\python\MP-DQN
python run_platform_pdqn.py --multipass True --episodes 100
```

这样您就能看到 MP-DQN 在 Platform 环境上的训练过程了！