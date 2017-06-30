import numpy as np

X = np.random.rand(5, 6)


def next_index(X, formal_step_x, formal_step_y, current_x, current_y, column, row):
    new_x, new_y = None, None
    if formal_step_x is None and formal_step_y is None:
        new_x = current_x + 1
        new_y = current_y + 1
    elif formal_step_y == current_y and formal_step_x < current_x:
        new_x = current_x + 1
        new_y = current_y + 1
        if current_x > column - 1 or X[new_x][new_y] == None:
            new_x = column - 1
            new_y = current_y + 1





