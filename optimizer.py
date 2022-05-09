import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time


class Res:
    def __init__(self, x, fun):
        self.x = x
        self.fun = fun


def optimize_nelder_mead(func, x0: np.array, num_points: int, alpha: float = 1., betta: float = 0.5, gamma: float = 2.,
                         rude_param: float = 0.1):
    # np.random.seed(19)
    # Подготовка
    start_f = func(x0)
    start_x = {ind + 1: x0 + rude_param * np.random.randn(x0.shape[0]) for ind in range(num_points)}
    start_pairs = [(start_x[key], func(start_x[key])) for key in start_x.keys()]
    while True:

        # Сортировка
        sorted_pairs = sorted(start_pairs, key=lambda x: x[1], reverse=True)
        x_h, f_h = sorted_pairs[0]
        x_g, f_g = sorted_pairs[1]
        x_l, f_l = sorted_pairs[-1]

        # Найдем центр тяжести всех точек за исключением x_h
        x_c = None
        for i, tup in enumerate(sorted_pairs):
            if i == 0:
                x_c = np.zeros(tup[0].shape)
            else:
                x_c += tup[0]
        x_c /= len(sorted_pairs) - 1

        # Отражение
        x_r = (1 + alpha) * x_c - alpha * x_h
        f_r = func(x_r)
        if f_r < f_l:
            # Растяжение
            x_e = (1 - gamma) * x_c + gamma * x_r
            f_e = func(x_e)
            if f_e < f_r:
                x_h = x_e
            else:
                x_h = x_r
        elif f_l < f_r and f_r < f_g:
            x_h = x_r
        else:
            if f_g < f_r and f_r < f_h:
                x_r, x_h = x_h, x_r
            # Сжатие
            x_s = betta * x_h + (1 - betta) * x_c
            f_s = func(x_s)
            if f_s < f_h:
                x_h = x_s
            else:
                # Глобальное сжатие
                x_homotet = dict()
                for k, v in start_x.items():
                    x_homotet[k] = x_l + (v - x_l) / 2
                start_x = x_homotet
        # Проверка сходимости - например, оценка дисперсии набора точек
        pairs = sorted([(start_x[key], func(start_x[key])) for key in start_x.keys()],
                       key=lambda x: x[1])
        new_f = pairs[0][1]
        relative_diff = (start_f - new_f) / (start_f + 0.0000001)
        if relative_diff < 0.00005:
            return Res(pairs[0][0], new_f)
        else:
            start_f = new_f
            start_pairs = pairs
