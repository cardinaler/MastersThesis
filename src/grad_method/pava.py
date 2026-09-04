import numpy as np

def pava(y):
    """Проекция на конус монотонно невозрастающих функций."""
    n = y.shape[0]
    u = y.copy().astype(float)
    blocks = [[i, 1, u[i]] for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] > blocks[i+1][2]:
            start, n1, val1 = blocks[i]
            _, n2, val2 = blocks[i+1]
            new_val = (n1 * val1 + n2 * val2) / (n1 + n2)
            blocks[i] = [start, n1 + n2, new_val]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1

    for start, count, val in blocks:
        u[start: start + count] = val
    return u
