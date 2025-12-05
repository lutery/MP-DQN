# MP-DQN Copilot Instructions

## 项目概览
- 本仓库实现了多种参数化动作空间强化学习算法：P-DQN、MP-DQN、SP-DQN、PA-DDPG、Q-PAMDP，均围绕离散动作 + 连续参数的 Gym 环境（Platform / Goal / Soccer）。
- 算法核心在 `agents/` 目录，各种变体（`pdqn.py`, `pdqn_multipass.py`, `pdqn_split*.py`, `qpamdp.py`, `paddpg.py` 等）共用统一的 Agent 接口（基类 `agents.agent.Agent`）。
- 环境适配逻辑在 `common/`：针对不同 Gym 域提供 observation / action wrapper，使所有 run 脚本看到的是统一接口（`Tuple(Discrete, Boxes...)` + 可选状态扩展 / 归一化）。
- 入口脚本为 `run_*_*.py`（如 `run_platform_pdqn.py`），使用 `click` 定义大量命令行参数，用于快速做超参搜索和复现实验。

## 运行与工作流
- 训练 / 评估统一通过 run 脚本调用，例如：
  - `python run_platform_pdqn.py`：Platform 域上 P-DQN/MP-DQN（默认 `--multipass True`）
  - `python run_goal_pdqn.py --split True`：Goal 域上 SP-DQN
  - `python run_soccer_pdqn.py --multipass True --layers [1024,512,256,128] --weighted True --indexed True`
- 所有脚本使用 `click`，布尔/列表参数要按现有示例格式书写（如 `--layers [128,]`，`--multipass True`），请保持兼容。
- 主要外部依赖：`torch`, `gym`, `numpy`, `click` 以及三个自定义 Gym 环境（`gym-platform`, `gym-goal`, `gym-soccer`），参见根目录 `README.md` 的安装方式。
- Windows 下推荐用 PowerShell 直接在仓库根目录运行：
  ```powershell
  cd c:\Projects\OpenSource\python\MP-DQN
  python run_platform_pdqn.py
  ```

## 关键结构与约定
- **参数化动作空间表示**：
  - 环境 action_space 统一为 `Tuple(Discrete(num_actions), Box(...) * num_actions)`。
  - Agent 内部将各个 Box 的维度拼接成一维参数向量；偏移量通过 `self.action_parameter_sizes` 和 `self.action_parameter_offsets` 管理，扩展多算法时务必复用这一约定。
  - 多数 run 脚本会在与环境交互前做一次“填充”或“展平”动作，例如 `run_platform_pdqn.py` 中的 `pad_action` 将 `(act, act_param)` 转回环境需要的 `(act, [p0, p1, p2])`。
- **状态/动作缩放**：
  - 状态缩放通过 `common.wrappers.ScaledStateWrapper` 完成；动作参数缩放通过 `ScaledParameterisedActionWrapper`，一般通过 `--scale-actions True` 打开。
  - 如果新增环境或更改空间维度，优先在 wrapper 中处理缩放/形状变更，而不是在 Agent 里写特例逻辑。
- **MP-DQN 关键逻辑**：
  - `agents.pdqn_multipass.MultiPassQActor` 通过多次“传递”实现 per-action Q 计算：复制 batch，使每个动作只填入各自参数段，最后拼回 `Q(s,a)` 矩阵。
  - 这依赖 `self.action_parameter_size_list` 和 `self.offsets` 对每个动作参数切分，修改时要保持与 `PDQNAgent` 中 offset 计算一致。
- **参数网络（ParamActor）与直通层**：
  - `agents.pdqn.ParamActor` 使用两部分线性层（`action_parameters_output_layer` + `action_parameters_passthrough_layer`），后者权重固定为 0，用于以偏置方式实现“初始参数”。
  - run 脚本中有 `initialise_params` 分支，通过 `agent.set_action_parameter_passthrough_weights` 设置这些偏置，以匹配论文中的手工初始化；新增 / 修改初始化策略时，请通过该 API，而不要直接改 `ParamActor` 内部属性。

## 扩展与修改建议
- 新增算法变体时：
  - 尽量复用 `PDQNAgent` 的接口和 replay 机制，仅在 actor 结构或更新规则上做差异（参考 `MultiPassPDQNAgent` 与 `SplitPDQNAgent` 的写法）。
  - 保持 `NAME` 字段语义清晰，方便日志/保存区分不同 agent。
- 新增环境/域时：
  - 在 `common/` 下创建对应 `*_domain.py`，仿照 `goal_domain.py`/`platform_domain.py` 写 wrapper（包含 observation 扩展逻辑、FlattenedActionWrapper 等）。
  - 新建 `run_*_*.py` 时，遵守现有脚本结构：`click` 参数名字/默认值风格一致、训练主循环结构基本相同，便于批量脚本调用和结果对比。
- 文件/导入：
  - 内部模块统一使用包路径导入（如 `from agents.pdqn import PDQNAgent`），避免相对路径穿越（`sys.path.append` 已在部分旧文件中使用，尽量不要扩大这种用法）。

## 对 Copilot / AI Agent 的特别提醒
- 代码依赖旧版 `gym` / `pytorch` API，修改时要兼顾兼容性；如需升级，请集中在单独 PR / 分支中完成，并更新 `README.md`。
- 训练脚本中的打印/保存路径逻辑（`save_dir`, `Monitor`, `np.save`）是论文复现实验的一部分，重构时请保持输出文件名和目录结构尽量兼容。
- 请避免为小改动随意更换网络结构默认值（hidden layers、激活函数、学习率等），以免破坏与论文结果的一致性；如确需改动，请在 run 脚本的 CLI 默认值层面完成，并更新 `README.md` 示例命令。
