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
        numpy.zeros((number_of_states, len(actions))),  # q_table initial values
        columns=actions,  # actions's name
    )
    # print(table)    # show table
    return table


def append_dataframe(dataframe: pandas.DataFrame, actions):
    length = len(dataframe)
    new_row = pandas.DataFrame(
        numpy.zeros((1, len(actions))),  # q_table initial values
        columns=actions, index=[length]  # actions's name
    )
    dataframe = dataframe.append(new_row)
    dataframe.reindex()
    return dataframe


q = build_q_table(3, ["buy", "sell"])
q2 = append_dataframe(q, ["buy", "sell"])
end = 0
