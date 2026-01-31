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


# 继续从代码上下问详细并简单易懂的讲解一下：out = -torch.mul(delta_a, action_params)
这句：

````python
out = -torch.mul(delta_a, action_params)
````

要从它前后的上下文一起看，才能明白“它在干什么”。

---

## 1）此时 `delta_a` 和 `action_params` 分别代表什么？

在 `_optimize_td_loss()` 的 “optimize actor（优化连续参数网络）” 部分，代码分两步走：

### Step A：先“求梯度”，但不更新 `actor_param`
```python
with torch.no_grad():
    action_params = self.actor_param(states)
action_params.requires_grad = True
Q = self.actor(states, action_params)
Q_loss.backward()
delta_a = deepcopy(action_params.grad.data)
```

- 这里的 `action_params` 是一个**临时叶子张量**（因为 `no_grad` 切断了它和 `actor_param` 网络参数的计算图）。
- `Q_loss.backward()` 得到的 `action_params.grad` 本质就是：
  \[
  \delta_a \approx \frac{\partial Q\_\text{loss}}{\partial a}
  \]
  （这里的 \(a\) 就是连续动作参数向量 `action_params`）

接着你又对 `delta_a` 做了边界相关的修正（`_invert_gradients`）以及可选的“非选中动作参数置零”（`_zero_index_gradients`），得到一个**你希望用于更新参数网络的“目标梯度方向”**。

### Step B：再“重算一次 `action_params`”，让它连回 `actor_param` 的参数
```python
action_params = self.actor_param(Variable(states))
```

这次不在 `no_grad` 里，所以 `action_params` **带着从网络参数 θ 到输出 a 的计算图**，可以把梯度传回 `actor_param.parameters()`。

---

## 2）`out = -delta_a * action_params` 的真正作用：构造“伪损失”来注入自定义梯度

PyTorch 的优化器（Adam）默认做的是：**最小化 loss**（梯度下降）。

但我们对 `actor_param` 的目标其实是：**让 Q 变大**（梯度上升方向来自 \(\partial Q / \partial a\)）。

做法就是构造一个“伪损失” \(L\)，使得最小化 \(L\) 等价于沿着你想要的方向更新。

令（按元素）：
\[
L = \sum_i \left(-\delta_{a,i} \cdot a_i\right)
\]

对应代码就是：
- `torch.mul(delta_a, action_params)`：逐元素乘，得到每一维的 \(\delta_{a,i} a_i\)
- 前面的负号 `-`：把方向翻过来

这样一来，对 `action_params` 的梯度是：
\[
\frac{\partial L}{\partial a_i} = -\delta_{a,i}
\]

然后再通过链式法则传回网络参数：
\[
\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial a}\frac{\partial a}{\partial \theta}
= (-\delta_a)\frac{\partial a}{\partial \theta}
\]

优化器做一步梯度下降：
\[
\theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta}
= \theta - \alpha \left( (-\delta_a)\frac{\partial a}{\partial \theta}\right)
= \theta + \alpha \left(\delta_a\frac{\partial a}{\partial \theta}\right)
\]

也就是说：**虽然你在“最小化” `out`，但参数更新效果等价于沿着 `delta_a` 指定的方向去“增大 Q”**。

> 负号的核心意义：把“最小化 loss”的更新规则，变成你想要的“最大化 Q”的效果。

---

## 3）为什么后面要 `out.backward(torch.ones(...))`？

因为 `out` 不是标量，而是形状大概为：

- `action_params`: `[batch_size, action_parameter_size]`
- `delta_a`: 同形状
- `out`: 同形状

对非标量调用 `backward()` 时，需要提供一个同形状的“上游梯度”。传 `torch.ones(out.shape)` 等价于对 `out` 的所有元素求和再反传：

\[
\text{backward(ones)} \;\;\Longleftrightarrow\;\; \nabla_\theta \sum_{b,i} out_{b,i}
\]

这就把每个维度的 \(-\delta_a\) 都注入回去了。

---

## 4）一句话总结

`out = -torch.mul(delta_a, action_params)` 不是在计算真正的 TD loss，它是在**构造一个方便 autograd 反传的“伪损失”**，把你已经算好的、并且经过边界/置零处理后的 `delta_a` 当作“希望 action_params 输出应该往哪里动”的信号，从而**驱动 `actor_param` 网络参数按最大化 Q 的方向更新**。

---

你如果愿意，我也可以把这段更新逻辑用更直观的“等价写法”（例如显式 `sum` 成标量 loss）在不改行为的前提下重写一下，方便读代码。

下面给你两个**等价/近似等价**的更直观写法。

---

## 1）严格等价（梯度完全一致）

原代码：

````python
out = -torch.mul(delta_a, action_params)              # out: [B, P]
out.backward(torch.ones(out.shape).to(self.device))   # 等价于对 out 的所有元素求和再反传
````

**严格等价**就是显式把它变成标量 `loss`（把所有元素求和）：

````python
# delta_a 建议当作常量使用（不需要对它求导）
delta_a_const = delta_a.detach()

loss_param = -(delta_a_const * action_params).sum()  # 标量
self.actor_param.zero_grad()
loss_param.backward()
self.actor_param_optimiser.step()
````

为什么严格等价：  
`out.backward(torch.ones_like(out))` 本质等价于对 `out` 做 `sum(out)` 再 `backward()`；而 `out = -(delta_a * action_params)`，所以就是 `-(delta_a * action_params).sum()`。

---

## 2）常用的“归一化版本”（不严格等价，但更稳）

上面的 `.sum()` 会让梯度规模随 `batch_size` 和参数维度 `P` 线性变大/变小；很多人更喜欢做平均（通常需要相应调整学习率）。例如：

````python
delta_a_const = delta_a.detach()

loss_param = -(delta_a_const * action_params).mean()  # 标量（做了归一化）
self.actor_param.zero_grad()
loss_param.backward()
self.actor_param_optimiser.step()
````

> 注意：这会把梯度缩放一个常数因子（大约是 `1/(B*P)`），**不完全等价**，但训练时可能更“数值直观”。

---

如果你把你这段 `_optimize_td_loss()` 后半段完整贴出来（尤其是 `delta_a` 是否被 `.cpu()` / `.to(self.device)` 过），我也可以顺手帮你确认上面这段在你项目里需要不要加 `.to(self.device)` 来避免 device mismatch。