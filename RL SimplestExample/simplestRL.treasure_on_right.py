"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


NUMBER_OF_STATES = 6   # the length of the 1 dimensional world -> 预测的长度
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police -> 大寫 Ε、小寫 ε 或ϵ
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor -> 越大，就会越重视以往经验，越小，就会只重视眼前利益
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move


def build_q_table(number_of_states, actions):
    """
    just build a table with zero in the beginning
    :param number_of_states:
    :param actions:
    :return:
    """
    table = pd.DataFrame(
        np.zeros((number_of_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
     # np.random.uniform() is a random Number
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(Current_State, Current_Action):
    """
    get the result based on the movement, right to +1, left to -1
    :param Current_State:
    :param Current_Action:
    :return: response 1 means End while response 0 means pass
    """
    # This is how agent will interact with the environment
    if Current_Action == 'right':    # move right
        if Current_State == NUMBER_OF_STATES - 2:   # terminate
            Next_State = 'terminal'
            Response = 1
        else:
            Next_State = Current_State + 1
            Response = 0
    else:   # move left
        Response = 0
        if Current_State == 0:
            Next_State = Current_State  # reach the wall
        else:
            Next_State = Current_State - 1
    return Next_State, Response


def update_env(state, episode, step_counter):
    """
    print the screen to the movement
    :param state:
    :param episode:
    :param step_counter:
    :return:
    """
    # This is how environment be updated
    env_list = ['-'] * (NUMBER_OF_STATES - 1) + ['T']   # '---------T' our environment
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(NUMBER_OF_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        current_state = 0
        is_terminated = False
        update_env(current_state, episode, step_counter)
        while not is_terminated:

            action = choose_action(current_state, q_table)
            next_state, R = get_env_feedback(current_state, action)  # take action & get next state and reward
            q_predict = q_table.loc[current_state, action]
            if next_state != 'terminal':
                q_target = R + GAMMA * q_table.iloc[next_state, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[current_state, action] += ALPHA * (q_target - q_predict)  # update
            current_state = next_state  # move to next state

            update_env(current_state, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)