'''
The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    There are four designated locations in the grid world indicated by R(ed), B(lue), G(reen), and Y(ellow). When the episode starts,
    the taxi starts off at a random square and the passenger is at a random location. The taxi drive to the passenger's location,
    pick up the passenger, drive to the passenger's destination (another one of the four specified locations), and then drop off the passenger.
    Once the passenger is dropped off, the episode ends.

    Observations:
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger

    is the taxi), and 4 destination locations.

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: dropoff passenger

    Rewards:
    There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. There is a reward of -10 for executing actions
    "pickup" and "dropoff" illegally.

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, B and Y): locations for passengers and destinations

    actions:
    - 0: south
    - 1: north
    - 2: east
    - 3: west
    - 4: pickup
    - 5: dropoff

    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
'''


import numpy as np
import gym
import random

env = gym.make("Taxi-v3")
# env.render()

# 新建q表，首先获得这个环境所需的动作的数量以及状态的数量
actions = env.action_space.n
# print('动作数：',actions)
states = env.observation_space.n
# print('状态数：',states)

# 新建一个初始值全为0的Q表,
# actions有六种可能的操作，接乘客，放下乘客，东南西北四个可以移动的方向
# states：环境一共有5*5个格子，有5个可能接乘客的地方与4个下车（R、G、Y、B）地点，所以一共有5*5*5*4=500个状态
q_table = np.zeros((states,actions))
# print(q_table)

# ?建立一些超参数
total_episodes = 5000      # 一共玩多少局游戏
total_test_episodes = 100  # 测试中一共走多少步
max_steps = 99             # 每一局游戏最多走多少步

learning_rate = 0.7        # 学习率
gamma = 0.618              # 未来折扣奖励

epsilon = 1.0       # Exploration rate 探索概率
max_epsilon = 1.0   # Exploration probability at start 一开始的探索概率
min_epsilon = 0.01  # 最低的探索概率
decay_rate = 0.01   # Exponential decay rate for exploration prob 探索概率的指数衰减概率

# 奖励：完成一次成功载客，可以得到20分，每移动一步扣1分，将乘客送至错误地点扣10分，每撞墙一次扣1分
for eposide in range(total_episodes):
    state = env.reset()
    step = 0
    done = False

    # 一局游戏最多99步
    for step in range(max_steps):
        # 在0与1之间随机生成一个数字
        exp_exp_tradeoff = random.uniform(0,1)

        # 如果随机生成的数字大于探索概率，选择Q表中最大的动作
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(q_table[state,:])
        # 否则，进行探索，选择随机性动作
        else:
            action = env.action_space.sample()

        # 得到的动作与环境进行交互后，得到奖励，环境变成新的状态
        new_state, reward, done, info = env.step(action)

        # 跟新Q表，根据公式：Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        q_table[state,action] = q_table[state,action] + learning_rate * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
        # 更新状态
        state = new_state

        if done == True:
            break

    # 减少探索概率（因为不确定性越来越小）
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * eposide)


env.reset()
rewards = []

# 利用训练好的Q表来玩游戏
for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        action = np.argmax(q_table[state, :])

        new_state, reward, done, info = env.step(action)

        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            break
        state = new_state

env.close()
print(q_table)
print("Score over time:" + str(sum(rewards)/total_test_episodes))