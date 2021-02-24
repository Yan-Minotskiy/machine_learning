import itertools as it
import numpy as np
from sklearn import metrics


class MGAA(object):
    class MGAAElement(object):
        class Kernel(object):
            def __init__(self, func, get_coef, size_var, size_coef):
                self.func = func
                self.get_coef = get_coef
                self.size_var = size_var
                self.size_coef = size_coef

        @staticmethod
        def classic_kernel_func(x, c):
            return c[0] + c[1] * x[0] + c[2] * x[1]

        @staticmethod
        def classic_kernel_get_coef(var1, var2, y):
            a = np.array([[len(var1), sum(var1), sum(var2)],
                          [sum(var1), sum(var1 ** 2), sum(var1 * var2)],
                          [sum(var2), sum(var1 * var2), sum(var2 ** 2)]])
            b = np.array([sum(y), sum(y * var1), sum(y * var2)])
            return np.linalg.solve(a, b)

        classic_kernel = Kernel(classic_kernel_func, classic_kernel_get_coef, 2, 3)

        # TODO: добавить другие ядра

        def __init__(self, kernel: Kernel = classic_kernel, metric=metrics.mean_absolute_error):
            self.kernel = kernel
            self.metric = metric
            self.coef = []
            self.result = np.inf

        def predict(self, x):
            return [self.kernel.func(i, self.coef) for i in x]

        def estimate(self, y_true, y_pred):
            return self.metric(y_true, y_pred)

        def fit(self, x, y):
            self.coef = self.kernel.get_coef(np.array(x)[:, [0]], np.array(x)[:, [1]], y)
            self.result = self.estimate(y, self.predict(x))

    class MGAANode(MGAAElement):
        def __init__(self, kernel, metric, connect=None):
            super().__init__(kernel, metric)
            self.connect = connect

        def get_data(self, x):
            if self.connect is None:
                return None
            if type(self.connect[0]) is int:
                return [x[i] for i in self.connect]
            return [i.function(x) for i in self.connect]

        @property
        def function(self):
            if self.connect is None:
                return None
            if type(self.connect[0]) is int:
                return lambda x: self.kernel.func([x[i] for i in self.connect], self.coef)
            return lambda x: self.kernel.func([i.function(x) for i in self.connect], self.coef)

    def __init__(self, kernel: MGAAElement.Kernel = MGAAElement.classic_kernel, metric=metrics.mean_absolute_error,
                 cross_epoch_select=False):
        self.kernel = kernel
        self.metric = metric
        self.cross_epoch_select = cross_epoch_select
        self.select_rate = select_rate
        self.function = None

    def fit(self, x, y):
        x = np.transpose(np.array(x))
        y = np.array(y)
        elements = []
        epoch = 0
        flag = True
        best_result = np.inf
        while flag:
            epoch_element = []
            epoch_result = []
            epoch += 1
            if epoch == 1:
                c = range(len(x))
            elif self.cross_epoch_select:
                c = elements
            else:
                c = elements[-1]
            for comb in it.combinations(c, self.kernel.size_var):
                element = self.MGAANode(self.kernel, self.metric, comb)
                element.fit(element.get_data(x), y)
                epoch_result.append(element.result)
                epoch_element.append(element)
            best_epoch_result = min(epoch_result)
            if best_epoch_result < best_result:
                best_result = best_epoch_result
                self.function = epoch_element[epoch_result.index(best_epoch_result)].function
            else:
                flag = False
            if self.cross_epoch_select:
                elements.extend(epoch_element)
            else:
                elements.append(epoch_element)

    def predict(self, x):
        return self.function(x)
