import numpy as np
import pandas as pd
import time

N_STATES = 6
ACTIONS = ['left','right']
EPSILON = 0.9        # 90%的时候选择最优动作，10%的时候选择随机动作
env = 7
EPISODES = 10
ALPHA = 0.1
GAMMA = 0.9

# 建立一个q表，行数是建立的一维环境的长度，列表示的是动作
def create_q_table(env, actions):
    table = pd.DataFrame(
        np.zeros((env,len(actions))),
        columns=actions
    )
    return table

# 根据Q表的值选择动作
def choose_actions(state, q_table):
    action = q_table.iloc[state,:]        # 取q表中指定的state行
    # 10%的概率选择随机动作
    if(np.random.uniform() > EPSILON) or ((action == 0).all()):
        action_name =np.random.choice(ACTIONS)
    else:
        # 90%的概率选择最优动作
        action_name =action.idxmax()

    return action_name

# 智能体和坏境的交互，到达终点时给的奖励是1，其余奖励都是0
# 当智能体向右移动的时候，如果当前所处的状态离终点还有一格，再往右就是终点，给奖励为1，否则就是将状态往右移一格
# 当智能体向左移动时，给的奖励都是0，当左边是边界时，智能体保持不动
def get_env_feedback(state, action):
    if action == 'right':
        if state == N_STATES-2:
            S_ = 'terminal'
            R = 1
        else:
            S_ =state + 1
            R = 0
    else:
        R = 0
        if state == 0:
            S_ = state
        else:
            S_ = state - 1
    return S_ , R

# 环境可视化
def update_env(state, episode, step_counter):
    # 构建一维环境‘-----T’
    env_list = ['-']*(N_STATES-1) + ['T']
    if state == 'terminal':
        interaction = '迭代次数%s: 总步数: %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction),end='')  # end=''，print函数不会在字符串末尾添加一个换行符，而是添加一个空字符串
        time.sleep(2)
        print('\r      ',end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction),end='')
        time.sleep(0.3)

# 强化学习主要的迭代步骤
def rl():
    # 新建一个Q表
    q_table = create_q_table(N_STATES,ACTIONS)
    # 不停的迭代更新Q表
    for episode in range(EPISODES):
        # 统计agent需要花多少步才能走到终点
        step_counter = 0
        # 当前智能体所在的地方
        state = 0
        # 是否探索到最终位置
        is_terminated = False
        # 环境可视化，将每步的步骤转化成图像输出
        update_env(state,episode,step_counter)
        while not is_terminated:
            # 从Q表中选择最合适的下一步动作（移动方向）
            action = choose_actions(state,q_table)
            # 根据下一步移动的动作来获得下一步的状态和得到的奖励
            S, reward = get_env_feedback(state,action)
            q_predict = q_table.loc[state,action]
            if S != 'terminal':
                q_target = reward + GAMMA * q_table.iloc[S, :].max()
            else:
                q_target = reward
                is_terminated = True

            # 更新q表的数据
            q_table.loc[state, action] += ALPHA * (q_target - q_predict)
            state = S

            update_env(state, episode, step_counter+1)
            step_counter += 1
    return q_table

if __name__ == "__main__":

    q_table = rl()
    print('\r\nQ表：\n')
    print(q_table)