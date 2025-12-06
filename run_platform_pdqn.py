import os
import click
import time
import gym
import gym_platform
from gym.wrappers import Monitor
from common import ClickPythonLiteralOption
from common.platform_domain import PlatformFlattenedActionWrapper

import numpy as np

from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper


def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)


def evaluate(env, agent, episodes=1000):
    returns = []
    timesteps = []
    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)
    # return np.column_stack((returns, timesteps))
    return np.array(returns)

# 注入式命令行参数解析，主要优势是可以自己编写相关的命令行参数类型解析器，支持不同的命令参数
@click.command()
@click.option('--seed', default=1, help='Random seed.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--episodes', default=20000, help='Number of epsiodes.', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.9, help='Discount factor.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=500, help='Number of transitions required to start learning.',
              type=int)  # may have been running with 500??
@click.option('--use-ornstein-noise', default=True,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=10000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
@click.option('--tau-actor', default=0.1, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)  # 0.001
@click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float) # 0.001/0.0001 learns faster but tableaus faster too
@click.option('--learning-rate-actor-param', default=0.0001, help="Critic network learning rate.", type=float)  # 0.00001
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--clip-grad', default=10., help="Parameter gradient clipping limit.", type=float)
@click.option('--split', default=False, help='Separate action-parameter inputs.', type=bool)
@click.option('--multipass', default=True, help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
@click.option('--layers', default='[128,]', help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/platform", help='Output directory.', type=str)
@click.option('--render-freq', default=100, help='How often to render / save frames of an episode.', type=int)
@click.option('--save-frames', default=False, help="Save render frames from the environment. Incompatible with visualise.", type=bool)
@click.option('--visualise', default=True, help="Render game states. Incompatible with save-frames.", type=bool)
@click.option('--title', default="PDDQN", help="Prefix of output files", type=str)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final, zero_index_gradients, initialise_params, scale_actions,
        clip_grad, split, indexed, layers, multipass, weighted, average, random_weighted, render_freq,
        save_freq, save_dir, save_frames, visualise, action_input_layer, title):

    # 确认保存目录
    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)

    # save_frames和visualise互斥，只能选择一种可视化的方式
    assert not (save_frames and visualise)
    if visualise:
        assert render_freq > 0
    if save_frames:
        assert render_freq > 0
        vidir = os.path.join(save_dir, "frames")
        os.makedirs(vidir, exist_ok=True)

    # 创建环境
    env = gym.make('Platform-v0')
    # 这里是动作的初始参数，手工指定为对应环境动作中连续动作每个动作的中间值
    # 估计是为了加快收敛做准备 todo 后续是怎么用的
    # 其他的信息查看md
    initial_params_ = [3., 10., 400.]
    if scale_actions:
        # 这个应该是把初始动作参数缩放到[-1,1]之间
        for a in range(env.action_space.spaces[0].n):
            # 公式：scaled = 2 * (value - low) / (high - low) - 1
            initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low) / (
                        env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.

    env = ScaledStateWrapper(env) # 将环境归一化到-1 - 1
    env = PlatformFlattenedActionWrapper(env) # 提取连续动作与离散动作同级 todo 为啥？
    if scale_actions:
        env = ScaledParameterisedActionWrapper(env) # 如果开启了缩放动作，则包装对应的环境包装器
    
    # 初始化保存目录
    dir = os.path.join(save_dir,title)
    # 环境监控包装器
    env = Monitor(env, directory=os.path.join(dir,str(seed)), video_callable=False, write_upon_reset=False, force=True)
    env.seed(seed)
    np.random.seed(seed)

    print(env.observation_space)

    from agents.pdqn import PDQNAgent
    from agents.pdqn_split import SplitPDQNAgent
    from agents.pdqn_multipass import MultiPassPDQNAgent # 本次近看MPDQN算法
    assert not (split and multipass)
    agent_class = PDQNAgent
    if split:
        agent_class = SplitPDQNAgent
    elif multipass: # 使用MPDQN强化学习算法标识
        agent_class = MultiPassPDQNAgent
    # 根据不同的算法标识，启动不同的Agent
    agent = agent_class(
                       env.observation_space.spaces[0], env.action_space,
                       batch_size=batch_size,
                       learning_rate_actor=learning_rate_actor,
                       learning_rate_actor_param=learning_rate_actor_param,
                       epsilon_steps=epsilon_steps,
                       gamma=gamma,
                       tau_actor=tau_actor,
                       tau_actor_param=tau_actor_param,
                       clip_grad=clip_grad,
                       indexed=indexed,
                       weighted=weighted,
                       average=average,
                       random_weighted=random_weighted,
                       initial_memory_threshold=initial_memory_threshold,
                       use_ornstein_noise=use_ornstein_noise,
                       replay_memory_size=replay_memory_size,
                       epsilon_final=epsilon_final,
                       inverting_gradients=inverting_gradients,
                       actor_kwargs={'hidden_layers': layers,
                                     'action_input_layer': action_input_layer,},
                       actor_param_kwargs={'hidden_layers': layers,
                                           'squashing_function': False,
                                           'output_layer_init_std': 0.0001,},
                       zero_index_gradients=zero_index_gradients,
                       seed=seed)

    if initialise_params:
        # todo 这个权重去更新啥的？
        # shape is (num_actions, state_size)
        initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
        # shape is (num_actions,)
        initial_bias = np.zeros(env.action_space.spaces[0].n)
        # 将偏置初始化为对应的初始动作参数
        for a in range(env.action_space.spaces[0].n):
            initial_bias[a] = initial_params_[a]
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
    print(agent)
    # todo 以下参数的含义后续再更新
    max_steps = 250
    total_reward = 0.
    returns = []
    start_time = time.time()
    video_index = 0
    # agent.epsilon_final = 0.
    # agent.epsilon = 0.
    # agent.noise = None

    for i in range(episodes):
        if save_freq > 0 and save_dir and i % save_freq == 0:
            # 根据保存频率保存模型
            agent.save_models(os.path.join(save_dir, str(i)))
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        if visualise and i % render_freq == 0:
            # 如果开启了可视化，并且到了渲染频率，则渲染当前环境
            env.render()

        # 根据当前状态选择动作
        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)

        episode_reward = 0.
        agent.start_episode()
        for j in range(max_steps):

            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            agent.step(state, (act, all_action_parameters), reward, next_state,
                       (next_act, next_all_action_parameters), terminal, steps)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state

            episode_reward += reward
            if visualise and i % render_freq == 0:
                env.render()

            if terminal:
                break
        agent.end_episode()

        if save_frames and i % render_freq == 0:
            video_index = env.unwrapped.save_render_states(vidir, title, video_index)

        returns.append(episode_reward)
        total_reward += episode_reward
        if i % 100 == 0:
            print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, str(i)))

    returns = env.get_episode_rewards()
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    np.save(os.path.join(dir, title + "{}".format(str(seed))),returns)

    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.epsilon_final = 0.
        agent.epsilon = 0.
        agent.noise = None
        evaluation_returns = evaluate(env, agent, evaluation_episodes)
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)


if __name__ == '__main__':
    run()
