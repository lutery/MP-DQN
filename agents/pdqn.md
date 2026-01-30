# _invert_gradients
## `_invert_gradients` 函数详解

### 一、函数源码与参数

```python
def _invert_gradients(self, grad, vals, grad_type, inplace=True):
    # 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)
    if grad_type == "actions":
        max_p = self.action_max
        min_p = self.action_min
        rnge = self.action_range
    elif grad_type == "action_parameters":
        max_p = self.action_parameter_max
        min_p = self.action_parameter_min
        rnge = self.action_parameter_range
    else:
        raise ValueError("Unhandled grad_type: '"+str(grad_type) + "'")

    max_p = max_p.cpu()
    min_p = min_p.cpu()
    rnge = rnge.cpu()
    grad = grad.cpu()
    vals = vals.cpu()

    assert grad.shape == vals.shape

    if not inplace:
        grad = grad.clone()
    with torch.no_grad():
        # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
        index = grad > 0
        grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
        grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

    return grad
```

**参数说明**:
- `grad`: 梯度张量，从 Q 网络反向传播得到的 $\frac{\partial Q}{\partial a}$
- `vals`: 当前动作参数值
- `grad_type`: 梯度类型，`"actions"` 或 `"action_parameters"`
- `inplace`: 是否原地修改梯度

---

### 二、核心思想：有界动作空间的梯度修正

#### **问题背景**

在参数化动作空间中，动作参数通常有界，例如：
$$a \in [a_{\min}, a_{\max}]$$

使用标准的策略梯度 $\nabla_\theta Q(s,a)$ 更新参数时，可能导致动作参数**超出边界**。传统解决方案是：

1. **Squashing Function** (如 tanh)：将输出压缩到有界区间
   - ❌ **缺点**: 在边界附近梯度消失，学习困难
   
2. **Gradient Clipping**: 直接裁剪梯度
   - ❌ **缺点**: 破坏梯度方向，可能阻碍学习

#### **Inverting Gradients 方案** (Hausknecht & Stone 2016)

**核心思想**: 根据**当前参数值与边界的距离**动态缩放梯度，使得：
- 参数接近上界时，向上的梯度被抑制
- 参数接近下界时，向下的梯度被抑制
- 参数在中间时，梯度保持较大幅度

这样既保证了边界约束，又避免了梯度消失。

---

### 三、数学原理

#### **1. 梯度方向判断**

```python
index = grad > 0  # 梯度为正（上升方向）
```

- `grad > 0`: 表示 $\frac{\partial Q}{\partial a} > 0$，Q 值随 $a$ 增大而增大
- `grad < 0`: 表示 $\frac{\partial Q}{\partial a} < 0$，Q 值随 $a$ 减小而增大

注释中提到 "actually > but Adam minimises, so reversed"：
- Adam 优化器执行**梯度下降**（最小化损失）
- 但我们想**最大化 Q 值**
- 因此在后续代码中使用 `-grad` 来反转优化方向

#### **2. 缩放公式**

对于**正梯度**（向上移动）:
```python
grad[index] *= (max_p - vals) / rnge
```

数学形式:
$$\nabla' = \nabla \cdot \frac{a_{\max} - a}{a_{\max} - a_{\min}}$$

- 当 $a \to a_{\max}$: 系数 $\to 0$，梯度被强烈抑制
- 当 $a \to a_{\min}$: 系数 $\to 1$，梯度保持完整
- 当 $a$ 在中间: 系数适中

对于**负梯度**（向下移动）:
```python
grad[~index] *= (vals - min_p) / rnge
```

数学形式:
$$\nabla' = \nabla \cdot \frac{a - a_{\min}}{a_{\max} - a_{\min}}$$

- 当 $a \to a_{\min}$: 系数 $\to 0$，梯度被强烈抑制
- 当 $a \to a_{\max}$: 系数 $\to 1$，梯度保持完整

---

### 四、在训练流程中的位置

#### **在 P-DQN 中的使用** (pdqn.py 第 580 行)

```python
# Step 1: 计算 Q 对动作参数的梯度
action_params.requires_grad = True
Q = self.actor(states, action_params)
Q_loss = torch.mean(torch.sum(Q, 1))  # 或其他聚合方式
self.actor.zero_grad()
Q_loss.backward()
delta_a = deepcopy(action_params.grad.data)  # ∂Q/∂a

# Step 2: 反转梯度（关键步骤）
action_params = self.actor_param(Variable(states))
delta_a[:] = self._invert_gradients(
    delta_a, 
    action_params, 
    grad_type="action_parameters", 
    inplace=True
)

# Step 3: 使用修正后的梯度更新参数网络
out = -torch.mul(delta_a, action_params)  # 注意这里的负号
self.actor_param.zero_grad()
out.backward(torch.ones(out.shape).to(self.device))
self.actor_param_optimiser.step()
```

**流程拆解**:

1. **计算原始梯度**: $\nabla_a Q(s,a)$ — Q 值对动作参数的导数
2. **反转梯度**: 根据边界约束修正梯度方向和幅度
3. **反向传播**: 使用修正后的梯度更新 `actor_param` 网络的参数 $\theta$

这实际上是在执行:
$$\theta \leftarrow \theta + \alpha \cdot \text{inverted}(\nabla_a Q) \cdot \frac{\partial a}{\partial \theta}$$

#### **在 PA-DDPG 中的使用** (paddpg.py 第 433-434 行)

```python
delta_a[:, self.num_actions:] = self._invert_gradients(
    delta_a[:, self.num_actions:].cpu(), 
    action_params[:, self.num_actions:].cpu(), 
    grad_type="action_parameters", 
    inplace=True
)
delta_a[:, :self.num_actions] = self._invert_gradients(
    delta_a[:, :self.num_actions].cpu(), 
    action_params[:, :self.num_actions].cpu(), 
    grad_type="actions", 
    inplace=True
)
```

PA-DDPG 中同时处理：
- **离散动作概率** (`actions`)
- **连续动作参数** (`action_parameters`)

两者分别使用不同的边界约束。

---

### 五、数值示例

假设动作参数 $a \in [0, 10]$，当前值 $a = 8$，原始梯度 $\nabla Q = +2$:

**不使用 Inverting Gradients**:
- 参数更新: $a' = 8 + 0.01 \times 2 = 8.02$ ✅
- 再更新几次: $a' = 8.2, 8.4, ..., 10.5$ ❌ **越界！**

**使用 Inverting Gradients**:
```python
max_p = 10, min_p = 0, rnge = 10
grad = +2, vals = 8

# 正梯度，使用上界约束
scale = (10 - 8) / 10 = 0.2
grad' = 2 * 0.2 = 0.4  # 梯度被缩小到原来的 1/5
```
- 参数更新: $a' = 8 + 0.01 \times 0.4 = 8.004$ ✅ **小步移动**
- 越接近边界，步长越小，自然地"软约束"

**反例**：当 $a = 2$，梯度 $\nabla Q = -3$:
```python
# 负梯度，使用下界约束
scale = (2 - 0) / 10 = 0.2
grad' = -3 * 0.2 = -0.6
```
- 更新: $a' = 2 - 0.01 \times 0.6 = 1.994$ ✅

---

### 六、与 run 脚本的配置

所有 `run_*_*.py` 脚本都有这个参数：

```python
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', 
              type=bool)
```

**默认值为 `True`**，说明这是推荐的做法。

对比两种方案：

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **Inverting Gradients** | ✅ 无梯度消失<br>✅ 保持边界<br>✅ 数值稳定 | ❌ 需要手动实现 | 有界连续动作 |
| **Squashing (tanh)** | ✅ 实现简单<br>✅ 自动有界 | ❌ 边界梯度消失<br>❌ 学习困难 | 无边界敏感性任务 |

---

### 七、代码中的性能注释

```python
# 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)
```

这个注释表明：
- 在 **Soccer** 域上，CPU 执行比 GPU 快 5 倍（可能因为这个操作涉及大量条件分支，不适合 GPU 并行）
- 在 **Goal/Platform** 域上，性能差异不大
- 因此代码中显式将张量移到 CPU: `grad = grad.cpu()`

---

### 八、总结

`_invert_gradients` 函数实现了**有界动作空间的智能梯度缩放**：

1. **核心公式**:
   $$\nabla' = \begin{cases}
   \nabla \cdot \frac{a_{\max} - a}{a_{\max} - a_{\min}} & \text{if } \nabla > 0 \\
   \nabla \cdot \frac{a - a_{\min}}{a_{\max} - a_{\min}} & \text{if } \nabla < 0
   \end{cases}$$

2. **效果**:
   - 自动防止参数越界
   - 避免梯度消失
   - 保持学习效率

3. **使用场景**:
   - P-DQN/MP-DQN: 修正连续动作参数的梯度
   - PA-DDPG: 同时修正离散动作概率和连续参数的梯度

4. **论文出处**: Hausknecht & Stone (2016) "Deep Reinforcement Learning in Parameterized Action Space"

这是参数化动作空间强化学习中的一个重要技术创新，使得算法能够在有界动作空间中稳定高效地学习。

Similar code found with 1 license type