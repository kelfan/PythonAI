import numpy
import pandas


def build_q_table(number_of_states, actions):
    """
    just build a table with zero in the beginning
    :param number_of_states:
    :param actions:
    :return:
    """
    table = pandas.DataFrame(
        numpy.zeros((number_of_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def append_dataframe(dataframe: pandas.DataFrame, headers):
    """
    add one more zero number line into a dataframe
    :param dataframe:
    :param actions:
    :return:
    """
    new_row = pandas.DataFrame(
        numpy.zeros((1, len(headers))),  # q_table initial values
        columns=headers, index=[len(dataframe)]  # actions's name
    )
    dataframe = dataframe.append(new_row)
    return dataframe


def choose_action(state, q_table, epsilon = 0.9, actions = ['left', 'right']   ):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
     # np.random.uniform() is a random Number
    if (numpy.random.uniform() > epsilon) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = numpy.random.choice(actions)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name