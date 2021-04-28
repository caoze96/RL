import numpy as np
import pandas as pd

actions = ['left','right']
env = 7

# 建立一个q表，行数是建立的一维环境的长度，列表示的是动作
def create_q_table(env, actions):
    table = pd.DataFrame(
        np.zeros((env,len(actions))),
        columns=actions
    )
    return table



if __name__ == "__main__":
    print(create_q_table(env,actions))