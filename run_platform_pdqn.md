# @click是啥？能详细讲一下吗

## 参数来源与含义

### 1. **Platform-v0 环境的动作空间**

从 platform_domain.py 可以看到，Platform 环境有 3 个离散动作，每个动作带一个连续参数：

- **动作 0 - 跳跃 (run)**：参数范围约 `[0, 6]`，**初始值 3.0** ≈ 中等速度
- **动作 1 - 跳跃 (hop)**：参数范围约 `[0, 20]`，**初始值 10.0** ≈ 中等跳跃力
- **动作 2 - 跳跃 (leap)**：参数范围约 `[0, 800]`，**初始值 400.0** ≈ 中等跳跃距离

### 2. **为什么是这些值？**

这些值是**论文实验中的经验性初始化**，目的是：

✅ **加速训练收敛**：
- 从"合理的动作参数"开始，而不是随机初始化
- 避免一开始就探索极端参数（如 leap=0 或 leap=800）

✅ **符合物理直觉**：
- 选择每个动作范围的中间值或略偏下的值
- 让 agent 初期行为更"保守"，减少失败

### 3. **在代码中的使用**

在 run_platform_pdqn.py 第 156-162 行：

````python
if initialise_params:
    initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
    initial_bias = np.zeros(env.action_space.spaces[0].n)
    for a in range(env.action_space.spaces[0].n):
        initial_bias[a] = initial_params_[a]  # 将 [3., 10., 400.] 设为偏置
    agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
````

**关键机制**：
- 通过 `ParamActor` 的 **passthrough layer**（直通层）实现
- 这个层的权重固定为 0，只用偏置 `initial_bias` 来输出初始参数
- 相当于告诉网络："在训练初期，对于任何状态，先输出这些合理的默认参数"

### 4. **动作参数缩放后的值**

当 `--scale-actions True` 时（第 103-107 行），这些参数会被缩放到 `[-1, 1]`：

````python
# 公式：scaled = 2 * (value - low) / (high - low) - 1
initial_params_[0] = 2. * (3. - 0) / (6 - 0) - 1    # = 0.0
initial_params_[1] = 2. * (10. - 0) / (20 - 0) - 1  # = 0.0
initial_params_[2] = 2. * (400. - 0) / (800 - 0) - 1 # = 0.0
````

**结果**：缩放后所有初始参数都变成 **0.0**（即归一化范围的中点），这在神经网络训练中是常见的初始化策略。

## 其他域的对比

| 域 | 初始参数 | 文件 |
|---|---------|------|
| **Platform** | `[3., 10., 400.]` | `run_platform_pdqn.py:101` |
| **Platform (Q-PAMDP)** | `[3., 10., 400.]` | `run_platform_qpamdp.py:51` |
| **Goal** | 基于 `kickto_weights` 矩阵 | `run_goal_paddpg.py:73-92` |
| **Soccer** | 无显式初始化 | 各 soccer 脚本 |

## 如何修改这些初始值？

如果想实验不同的初始化策略，可以：

### 方法 1：直接修改代码
````python
// ...existing code...
initial_params_ = [1.5, 5., 200.]  # 更保守的初始值
// ...existing code...
````

### 方法 2：添加命令行参数（推荐）
````python
// ...existing code...
@click.option('--initial-params', default='[3., 10., 400.]', 
              help='Initial action parameters for [run, hop, leap].', 
              cls=ClickPythonLiteralOption)
def run(..., initial_params, ...):
    initial_params_ = initial_params
    # ...existing code...
````

运行时：
```powershell
python run_platform_pdqn.py --initial-params "[2., 8., 300.]"
```

### 方法 3：关闭初始化（完全随机）
```powershell
python run_platform_pdqn.py --initialise-params False
```

这样 `ParamActor` 会使用网络的默认初始化（通常是小随机值）。

## 总结

`[3., 10., 400.]` 是**基于 Platform 环境物理特性的经验值**，并非通过数学推导得出，而是实验中发现能加速收敛的"好的起始点"。这是强化学习中常见的**领域知识注入**技巧。


# Gym Monitor 详解

`Monitor` 是 OpenAI Gym 提供的**环境监控包装器**，用于记录和保存训练过程中的统计信息、视频等数据。

## 基本功能

### 1. **在代码中的使用**

````python
// ...existing code...
from gym.wrappers import Monitor

# 第 120-121 行
dir = os.path.join(save_dir, title)
env = Monitor(env, directory=os.path.join(dir, str(seed)), 
              video_callable=False, 
              write_upon_reset=False, 
              force=True)
````

### 2. **核心参数说明**

| 参数 | 值 | 作用 |
|------|-----|------|
| `directory` | `results/platform/PDDQN1/1/` | 保存统计数据的目录 |
| `video_callable` | `False` | **不自动录制视频**（本项目用自定义方式保存帧） |
| `write_upon_reset` | `False` | 不在每次 `reset()` 时立即写入文件（提高性能） |
| `force` | `True` | 如果目录已存在则覆盖（避免报错） |

## Monitor 自动记录的数据

### 1. **episode 统计信息**

Monitor 会在目录下生成一个 **`openaigym.episode_batch.*.stats.json`** 文件，包含：

```json
{
  "initial_reset_timestamp": 1701763200.123,
  "timestamps": [1701763210.456, 1701763220.789, ...],
  "episode_lengths": [42, 67, 31, ...],
  "episode_rewards": [150.3, 210.7, 89.2, ...],
  "episode_types": ["t", "t", "t", ...]
}
```

**字段含义**：
- `timestamps`: 每个 episode 结束的时间戳
- `episode_lengths`: 每个 episode 的步数
- `episode_rewards`: 每个 episode 的总奖励
- `episode_types`: `"t"` 表示训练（training）

### 2. **在代码中如何使用这些数据**

````python
// ...existing code...
# 第 226-227 行：获取 Monitor 记录的奖励
returns = env.get_episode_rewards()
print("Ave. return =", sum(returns) / len(returns))
print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)
````

**`env.get_episode_rewards()`** 返回的就是 Monitor 记录的所有 episode 奖励列表。

## 与自定义记录的对比

### 在你的代码中有两套记录系统：

#### 1. **Monitor 自动记录**（后台默默工作）
```python
returns = env.get_episode_rewards()  # 从 Monitor 获取
```

#### 2. **手动记录**（训练循环中显式追踪）
```python
# 第 175、217 行
returns = []  # 手动维护的列表
returns.append(episode_reward)  # 每个 episode 结束后手动添加
```

### 为什么要两套？

| 记录方式 | 优点 | 缺点 | 使用场景 |
|---------|------|------|---------|
| **Monitor** | 自动、标准化、包含时间戳 | 不灵活、无法记录自定义指标 | 通用实验记录 |
| **手动记录** | 灵活、可实时计算滚动平均 | 需要自己写代码维护 | 训练中实时监控、自定义指标 |

在代码第 219-221 行可以看到手动记录的优势：

````python
// ...existing code...
if i % 100 == 0:
    print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(
        str(i), 
        total_reward / (i + 1),           # 平均奖励
        np.array(returns[-100:]).mean()   # 最近 100 个 episode 的平均
    ))
````

这种实时计算的滚动平均是 Monitor 无法直接提供的。

## Monitor 的其他功能（本项目未使用）

### 1. **自动录制视频**

如果设置 `video_callable=lambda episode_id: episode_id % 100 == 0`，Monitor 会自动录制视频：

```python
# 本项目没有使用，因为用了自定义的帧保存方式
env = Monitor(env, directory=dir, 
              video_callable=lambda ep: ep % 100 == 0)  # 每 100 个 episode 录制一次
```

**为什么本项目不用这个功能？**

看第 214-215 行：

````python
// ...existing code...
if save_frames and i % render_freq == 0:
    video_index = env.unwrapped.save_render_states(vidir, title, video_index)
````

项目使用了**自定义的帧保存逻辑**（`save_render_states`），因为：
- 可以控制帧的保存格式（可能是图片序列而非视频）
- 可以自定义保存频率（`render_freq`）
- 更灵活地控制可视化内容

### 2. **manifest 文件**

Monitor 还会生成 `openaigym.manifest.*.manifest.json`：

```json
{
  "env_info": {
    "env_id": "Platform-v0",
    "gym_version": "0.17.3"
  },
  "videos": [],
  "stats": "openaigym.episode_batch.0.12345.stats.json"
}
```

记录了环境信息和关联的统计文件。

## Monitor 的工作原理

### 1. **Wrapper 模式**

Monitor 是一个**环境包装器**（Wrapper），遵循装饰器模式：

```python
原始环境 → ScaledStateWrapper → PlatformFlattenedActionWrapper 
         → ScaledParameterisedActionWrapper → Monitor
```

每次调用 `step()` 或 `reset()` 时，Monitor 会：
1. 拦截调用
2. 记录统计信息
3. 转发给内层环境

### 2. **核心拦截逻辑**（简化示意）

```python
class Monitor(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 记录统计
        self.episode_reward += reward
        self.episode_length += 1
        
        if done:
            # episode 结束，保存记录
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)
            self.write_stats()  # 写入 JSON 文件
            
        return obs, reward, done, info
```

## 在你的项目中的最佳实践

### 1. **训练时的典型配置**

````python
# 训练时：启用 Monitor，禁用视频录制
env = Monitor(env, directory=save_dir, 
              video_callable=False,        # 不录制视频
              write_upon_reset=False,      # 批量写入，提高性能
              force=True)                  # 覆盖已存在的目录
````

### 2. **评估时的配置**

如果需要录制评估视频，可以这样：

````python
# 评估时：录制前 10 个 episode
eval_env = Monitor(eval_env, directory=eval_dir,
                   video_callable=lambda ep: ep < 10,  # 只录制前 10 个
                   force=True)
````

### 3. **数据分析**

训练结束后，可以读取 Monitor 的统计文件进行分析：

```python
import json
import glob

# 读取所有统计文件
stats_files = glob.glob(os.path.join(save_dir, '*.stats.json'))
for f in stats_files:
    with open(f, 'r') as fp:
        stats = json.load(fp)
        print(f"Episode rewards: {stats['episode_rewards']}")
        print(f"Episode lengths: {stats['episode_lengths']}")
```

## 常见问题

### Q1: 为什么训练结束后才看到 JSON 文件？

**A**: 因为设置了 `write_upon_reset=False`，Monitor 会等到程序结束或显式调用 `env.close()` 才批量写入。

第 223 行的 `env.close()` 触发了最终写入：

````python
// ...existing code...
end_time = time.time()
print("Took %.2f seconds" % (end_time - start_time))
env.close()  # 触发 Monitor 写入统计文件
````

### Q2: Monitor 会影响训练速度吗？

**A**: 影响极小。Monitor 只记录必要的标量数据（奖励、步数、时间戳），不会显著降低训练速度。但如果启用 `video_callable`，录制视频会显著拖慢训练。

### Q3: 如何禁用 Monitor？

直接移除这一行包装即可：

```python
# 注释掉 Monitor 包装
# env = Monitor(env, directory=dir, ...)
```

但这样会导致 `env.get_episode_rewards()` 报错，需要完全依赖手动记录。

## 总结

在你的 MP-DQN 项目中，**Monitor 主要用于**：

1. ✅ **标准化实验记录**：自动保存 episode 奖励、步数、时间戳
2. ✅ **结果复现**：通过 JSON 文件可以准确复现实验曲线
3. ✅ **方便对比**：不同 seed/算法的运行可以统一用 Monitor 数据绘图

同时保留了**手动记录系统**用于：
- 训练过程中的实时监控（滚动平均）
- 保存最终的 `.npy` 文件（第 229 行）
- 自定义的帧保存逻辑

这种"Monitor + 手动记录"的组合是强化学习实验的**标准做法**，既保证了标准化，又保持了灵活性。