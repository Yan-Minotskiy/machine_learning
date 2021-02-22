import itertools as it
import numpy as np


class MGAA(object):
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.count_var = self.x.shape[1]
        self.length = self.x.shape[0]
        self.coef = []
        self.estimation = []
        self.selection_best = []
        self.num_selection = -1
        self.best = np.inf
        next_selection = True
        while next_selection:
            self.coef.append([])
            self.estimation.append(([]))
            self.num_selection += 1
            for i in it.combinations(range(self.count_var), 2):
                var1 = self.x[:, i[0]]
                var2 = self.x[:, i[1]]
                a = np.array([[self.length, sum(var1), sum(var2)],
                              [sum(var1), sum(var1 ** 2), sum(var1 * var2)],
                              [sum(var2), sum(var1 * var2), sum(var2 ** 2)]])
                b = np.array([sum(self.y), sum(self.y * var1), sum(self.y * var2)])
                c = np.linalg.solve(a, b)
                self.coef[self.num_selection].append(c)  # сохранение коэффициентов
                error = []
                for j in range(self.length):
                    error.append((c[0] + c[1] * var1[j] + c[2] * var2[j]) - y)  # оптимизация по метрике MAE
                self.estimation[self.num_selection] = sum(error) / self.length
            self.selection_best = min(self.estimation[self.num_selection])
            if self.selection_best > self.best:
                self.best = self.selection_best
            else:
                next_selection = False

