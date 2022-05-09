import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from optimizer import optimize_nelder_mead, Res


class TCH:
    """
        ML алгоритм "Tree of Cutting Hyperplanes" или "Дерево секущих гиперплоскостей"
        основанный на теории Комитета старшинства
    """
    
    def __init__(self, N: int) -> None:
        """
            :param N: достаточное число наблюдений, находящихся выше гиперплоскости, для отсечения
        """
        
        self.X_train = None
        self.y_train = None
        self.L = -1
        self.N = N
        self.weights_hp = dict()  # здесь будут записаны оптимальные веса гиперплоскостей в виде:
        # {'num_hyperplane': (voted_class, optim_weights, probability, probability of rest samples)}
        self.optim_params = dict()  # здесь будут записаны параметры оптимизации гиперплоскостей в виде:
        # {'cycle_range': int, 'disp': bool, 'adaptive': bool, 'maxiter': int or None, 'xatol': float or None}
        # значение параметров дано в описании метода make_hyperplane
        self.feature_importance = None
    
    def scale(self, X: np.array) -> np.array:
        """
            Статический метод, применяющий min-max-scaler

            :param X: матрица признаков
            :returns: отшкалированная матрица признаков
        """
        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())
        return X
    
    def expand(self, X: np.array) -> np.array:
        """
            Concatenates the feature matrix with the column of ones

            :param X: матрица признаков
            :returns: исходная матрица признаков со столбцом единиц (добавили bias)
        """
        
        X_copy = deepcopy(X)
        return np.hstack((X_copy, np.ones(X_copy.shape[0]).reshape(-1, 1)))
    
    def probability(self, X: np.array, w: np.array) -> np.array:
        """
            Принимает на вход матрицу фичей и вектор весов
            Возвращает предсказание - вероятность того, что y = 1 при фиксированных x, P(y=1|x)

            :param X: матрица признаков
            :param w: вектор весов
            :returns: вероятность того, что y = 1 при фиксированных x, P(y=1|x), т.е. вектор вероятностей = ReLU1(X.T*w)
        """
        # ReLu1 function
        linear = np.dot(X, w)
        linear[linear < 0] = 0
        linear[linear > 1] = 1
        return linear
    
    def compute_loss(self, X: np.array, y: np.array, w: np.array) -> np.array:
        """
            Вычисление кастомного лосса

            :param X: матрица признаков
            :param y: вектор целевой переменной
            :param w: вектор весов
            :returns: значение функции потерь
        """
        
        p1 = self.probability(X, w)
        loss = np.sum((self.L - (self.L + 1) * y) * p1)
        return loss
    
    def compute_train_loss_class_0(self, w: np.array) -> np.array:
        """
            Вычисление кастомного лосса для точек класса 0

            Function that we want to minimize, the committee member votes for class 0

            :param w: вектор весов
            :returns: значение функции потерь на обучающей выборке, когда член коммитета голосует за класс 0
        """
        
        if self.X_train is None or self.y_train is None:
            raise Exception('Model is not fitted')
        return self.compute_loss(self.X_train, 1 - self.y_train, w)
    
    def compute_train_loss_class_1(self, w: np.array) -> np.array:
        """
            Вычисление кастомного лосса для точек класса 1

            Function that we want to minimize, the committee member votes for class 1

            :param w: вектор весов
            :returns: значение функции потерь на обучающей выборке, когда член коммитета голосует за класс 1
        """
        
        if self.X_train is None or self.y_train is None:
            raise Exception('Model is not fitted')
        return self.compute_loss(self.X_train, self.y_train, w)
    
    def make_hyperplane(self, class_num: int, X_train: np.array, optim_method: str, c: float = 0.1,
                        cycle_range: int = 100, disp: bool = False, adaptive: bool = True, maxiter: int = None,
                        xatol: float = None, verbose: int = 0) -> None:
        """
            Function that makes hyperplane

            :param class_num: класс, за который голосует данный член комитета: [0, 1]
            :param X_train: матрица признаков обучающей выборки
            :param optim_method: метод оптимизации: ['Nelder-Mead', 'differential_evolution', 'BFGS', 'CG', 'SLSQP', \
                                                    'COBYLA', 'TNC']
            :param с: вспомогательный коэффициент для выбора начального приближения
            :param cycle_range: количество итераций минимизации функции потерь
            Параметры оптимизации с помощью алгоритма Нелдера-Мида:
                :param disp: bool: печать сообщения о сходимости
                :param adaptive: bool: адаптация параметров алгоритма для размерности задачи (полезно при больших размерностях)
                :param maxiter: максимально допустимое количество итераций при оптимизации
                :param xatol: абсолютная ошибка на оптимальных точках между итерациями, приемлемая для сходимости
            :param verbose: подробный вывод описания обучения: [0, 1, 2]:
                0 - не печатать ничего
                1 - печатать общее время обучения
                2 - подробный вывод информации о процессе обучения
            :returns: значение функции потерь на тестовой выборке
        """
        
        if self.X_train is None or self.y_train is None:
            raise Exception('Using make_hyperplane method before fitting')
        if class_num not in [0, 1]:
            raise Exception('Only binary classification is available, class_num should be 0 or 1')
        if optim_method not in ['Nelder-Mead', 'differential_evolution', 'BFGS', 'CG', 'SLSQP', 'COBYLA', 'TNC']:
            raise Exception(
                """Unavailable optimization method, only ['Nelder-Mead', 'differential_evolution', 'BFGS', 'CG', 
                'SLSQP', 'COBYLA', 'TNC'] are available""")
        
        optim_result = []  # оптимальные точки первого приближения
        optim_result_more_precise = []  # оптимальные точки второго приближения
        start_time = time.time()
        
        if verbose == 2:
            print('Optimization is started')
        
        # Оптимизация методом Нелдера-Мида
        if optim_method == 'Nelder-Mead':
            
            # Строим первое приближение
            for i in range(cycle_range):
                
                start_w = np.array((np.random.rand(X_train.shape[1]) - 0.5) * c)
                
                # Строим две гиперплоскости: для классов 0 и 1 отдельно
                res = None
                if class_num == 1:
                    res = optimize_nelder_mead(self.compute_train_loss_class_1, x0=start_w, num_points=30,
                                               alpha=0.9)
                    
                    # res = minimize(self.compute_train_loss_class_1, x0=start_w, method='Nelder-Mead',
                    #                options={'disp': disp, 'adaptive': adaptive, 'maxiter': maxiter, 'xatol': xatol})
                
                elif class_num == 0:
                    res = optimize_nelder_mead(self.compute_train_loss_class_0, x0=start_w, num_points=30,
                                               alpha=0.9)
                    
                    # res = minimize(self.compute_train_loss_class_0, x0=start_w, method='Nelder-Mead',
                    #                options={'disp': disp, 'adaptive': adaptive, 'maxiter': maxiter, 'xatol': xatol})
                
                optim_result.append([res.fun, res.x])
            
            if not optim_result:
                raise Exception('We\'ve not get any satisfying first approximation')
            
            # Отбираем 8% (эпмирика) лучших точек
            optim_result.sort(key=lambda x: x[0])
            optim_result = [x[1] for x in optim_result][:int(0.08 * cycle_range)]
            
            if verbose == 2:
                print('First approximation is obtained')
            
            # Строим второе приближение на основе первого
            for start_w_new in optim_result:
                
                res = None
                if class_num == 1:
                    res = optimize_nelder_mead(self.compute_train_loss_class_1, x0=start_w_new, num_points=30,
                                               alpha=0.9, rude_param=0.05)
                    
                    # res = minimize(self.compute_train_loss_class_1, x0=start_w_new, method='Nelder-Mead',
                    #                options={'disp': False, 'adaptive': True, 'xatol': 1})
                elif class_num == 0:
                    res = optimize_nelder_mead(self.compute_train_loss_class_0, x0=start_w_new, num_points=30,
                                               alpha=0.9, rude_param=0.05)
                    
                    # res = minimize(self.compute_train_loss_class_0, x0=start_w_new, method='Nelder-Mead',
                    #                options={'disp': False, 'adaptive': True, 'xatol': 1})
                
                for k in range(2):
                    if class_num == 1:
                        # res = optimize_nelder_mead(self.compute_train_loss_class_1, x0=start_w_new, num_points=30,
                        #                                     alpha=0.9, rude_param=0.01)
                        
                        res = minimize(self.compute_train_loss_class_1, x0=res.x, method='Nelder-Mead',
                                       options={'disp': False, 'adaptive': True})
                    elif class_num == 0:
                        # res = optimize_nelder_mead(self.compute_train_loss_class_0, x0=start_w_new, num_points=30,
                        #                                     alpha=0.9, rude_param=0.01)
                        
                        res = minimize(self.compute_train_loss_class_0, x0=res.x, method='Nelder-Mead',
                                       options={'disp': False, 'adaptive': True})
                
                optim_result_more_precise += [[res.fun, res.x]]
            
            final_result = sorted(optim_result_more_precise, key=lambda elem: elem[0])
            hyperplane_coefficients = final_result[-1][1]
            min_loss_func = final_result[-1][0]
            
            if verbose == 2:
                print('The minimum of the loss function: {0}'.format(min_loss_func))
            if verbose == 2:
                print('Time taken for optimization: {0}'.format(time.time() - start_time))
        
        if optim_method == 'TNC':
            
            for i in range(cycle_range):
                
                start_w = np.array((np.random.rand(X_train.shape[1]) - 0.5) * c)
                
                if class_num == 1:
                    
                    res = minimize(self.compute_train_loss_class_1, x0=start_w, method=optim_method)
                
                elif class_num == 0:
                    
                    res = minimize(self.compute_train_loss_class_0, x0=start_w, method=optim_method)
                
                optim_result.append([res.fun, res.x])
            
            if optim_result == []:
                raise Exception('We\'ve not get any satisfying first approximation')
            
            optim_result.sort(key=lambda x: x[0])
            optim_result = [x[1] for x in optim_result][:int(0.1 * cycle_range)]
            if verbose == 2:
                print('First approximation is obtained')
            
            for start_w_new in optim_result:
                
                if class_num == 1:
                    res = minimize(self.compute_train_loss_class_1, x0=start_w_new, method=optim_method)
                elif class_num == 0:
                    res = minimize(self.compute_train_loss_class_0, x0=start_w_new, method=optim_method)
                
                for k in range(2):
                    if class_num == 1:
                        res = minimize(self.compute_train_loss_class_1, x0=res.x, method=optim_method)
                    elif class_num == 0:
                        res = minimize(self.compute_train_loss_class_0, x0=res.x, method=optim_method)
                
                optim_result_more_precise += [[res.fun, res.x]]
            
            optim_result_more_precise = pd.DataFrame(optim_result_more_precise)
            optim_result_more_precise = optim_result_more_precise.sort_values(0).head(1)
            hyperplane_coefficients = optim_result_more_precise[1].values[0]
            min_loss_func = optim_result_more_precise[0].values[0]
            
            if verbose == 2:
                print('The minimum of the loss function: {0}'.format(min_loss_func))
            if verbose == 2:
                print('Time taken for optimization: {0}'.format(time.time() - start_time))
        
        return hyperplane_coefficients
    
    def cutter(self, X: np.array, w: np.array) -> np.array:
        """
            Function that makes binary targets for rational numbers

            :param X: матрица признаков
            :param w: вектор оптимальных весов
            :returns: бинаризованные предсказания целевой переменной
        """
        
        linear = np.dot(X, w)
        return (np.sign(linear) > 0).astype(int)
    
    def fit(self, X: np.array, y: np.array, optim_method: str = 'Nelder-Mead', cycle_range: int = 100,
            disp: bool = False, adaptive: bool = True, maxiter: int = 10, xatol: float = 0.3, verbose: int = 0,
            stop_coeff: float = 1.7):
        """
            Fits the algorithm on the train sample

            :param X: матрица признаков, обучающая выборка
            :param y: вектор истинных значений целевой переменной обучающей выборки
            :param optim_method: метод оптимизации: ['Nelder-Mead', 'differential_evolution', 'BFGS', 'CG', 'SLSQP',
                                                    'COBYLA', 'TNC']
            Параметры оптимизации с помощью алгоритма Нелдера-Мида:
                :param disp: bool: печать сообщения о сходимости
                :param adaptive: bool: адаптация параметров алгоритма для размерности задачи (полезно при больших размерностях)
                :param maxiter: максимально допустимое количество итераций при оптимизации
                :param xatol: абсолютная ошибка на оптимальных точках между итерациями, приемлемая для сходимости
            :param verbose: подробный вывод описания обучения: [0, 1, 2]:
                0 - не печатать ничего
                1 - печатать общее время обучения
                2 - подробный вывод информации о процессе обучения
            :returns: -, но на выходе обученная моделька
        """
        
        start_of_fit_time = time.time()
        
        X_new = self.expand(X)
        start_shape = X_new.shape[0]

        self.feature_importance = np.zeros(X.shape[1])
        
        self.X_train = X_new
        self.y_train = y
        self.optim_params = {'cycle_range': cycle_range, 'disp': disp, 'adaptive': adaptive,
                             'maxiter': maxiter, 'xatol': xatol}
        
        hp_num = 1
        k = 10
        
        print(f'Optimizing method chosen is {optim_method}')
        
        # Проверка строим ли дерево дальше (эмпирика)
        while self.X_train.shape[0] > stop_coeff * self.N:
            
            if verbose == 2:
                print('Making hyperplane number {}'.format(hp_num))
                print('X_train.shape[0] = {}'.format(self.X_train.shape[0]))
            
            k = max(k, 2)
            
            # Строим очередную гиперплоскость
            while True:
                
                self.L = 2 ** k
                
                if verbose == 2:
                    print('k = {}'.format(k))
                    print('L = {}'.format(self.L))
                    print('Optimizing hyperplane for class 1')
                
                # Гиперплоскость для класса 1
                hp_weights_class_1 = self.make_hyperplane(class_num=1, X_train=self.X_train,
                                                          optim_method=optim_method,
                                                          c=0.1, cycle_range=self.optim_params['cycle_range'],
                                                          disp=self.optim_params['disp'],
                                                          adaptive=self.optim_params['adaptive'],
                                                          maxiter=self.optim_params['maxiter'],
                                                          xatol=self.optim_params['xatol'],
                                                          verbose=verbose)
                
                if verbose == 2:
                    print('Optimizing hyperplane for class 0')
                
                # Гиперплоскость для класса 0
                hp_weights_class_0 = self.make_hyperplane(class_num=0, X_train=self.X_train,
                                                          optim_method=optim_method,
                                                          c=0.1, cycle_range=self.optim_params['cycle_range'],
                                                          disp=self.optim_params['disp'],
                                                          adaptive=self.optim_params['adaptive'],
                                                          maxiter=self.optim_params['maxiter'],
                                                          xatol=self.optim_params['xatol'],
                                                          verbose=verbose)
                
                # Находим точки, которые отсекла каждая из гиперплоскостей
                cut_1 = self.cutter(self.X_train, hp_weights_class_1)
                nrof_cutted_1 = np.sum(cut_1)
                cut_0 = self.cutter(self.X_train, hp_weights_class_0)
                nrof_cutted_0 = np.sum(cut_0)
                X_1 = self.X_train[cut_1 == 1]
                y_1 = self.y_train[cut_1 == 1]
                y_1_rest = self.y_train[cut_1 == 0]  # оставшиеся после отсечения сэмплы
                X_0 = self.X_train[cut_0 == 1]
                y_0 = self.y_train[cut_0 == 1]
                y_0_rest = self.y_train[cut_0 == 0]  # оставшиеся после отсечения сэмплы
                
                if verbose == 2:
                    print('X_1.shape[0] = {}'.format(X_1.shape[0]))
                    print('X_0.shape[0] = {}'.format(X_0.shape[0]))
                
                if X_1.shape[0] < self.N and X_0.shape[0] < self.N:
                    if verbose == 2:
                        print('Cutted data shape is not enough\n')
                    k -= 1
                    continue
                else:
                    # Выбираем лучшую из гиперплоскостей по количеству отсеченных наблюдений
                    if X_1.shape[0] >= X_0.shape[0]:
                        proba = y_1.sum() / len(y_1)
                        proba_rest = y_1_rest.sum() / len(y_1_rest)
                        self.weights_hp[hp_num] = (1, hp_weights_class_1, proba, proba_rest)
                        
                        self.feature_importance += np.abs(hp_weights_class_1[:-1]) * nrof_cutted_1 / start_shape
                        
                        self.X_train = self.X_train[cut_1 == 0]
                        self.y_train = self.y_train[cut_1 == 0]
                    
                    elif X_1.shape[0] < X_0.shape[0]:
                        proba = y_0.sum() / len(y_0)
                        proba_rest = y_0_rest.sum() / len(y_0_rest)
                        self.weights_hp[hp_num] = (0, hp_weights_class_0, proba, proba_rest)

                        self.feature_importance += np.abs(hp_weights_class_0[:-1]) * nrof_cutted_0 / start_shape
                        
                        self.X_train = self.X_train[cut_0 == 0]
                        self.y_train = self.y_train[cut_0 == 0]
                    
                    hp_num += 1
                    if verbose == 2:
                        print()
                    break
        
        end_of_fit_time = time.time()
        
        if verbose == 2 or verbose == 1:
            print('Time taken to fit the model: {0}'.format(end_of_fit_time - start_of_fit_time))
        
        return self
    
    def predict_proba(self, X: np.array) -> np.array:
        """
            Makes predict_proba for the sample

            :param X: матрица признаков, обучающая выборка
            :returns: предсказания вероятностей отнесения к классу 1
        """
        
        start_of_predict_time = time.time()
        X_new = self.expand(X)
        test_ = X_new.copy()
        
        if self.weights_hp == {}:
            raise Exception('Model is not fitted')
        
        predictions_np = np.zeros((X.shape[0], 2))
        
        for hp_num in self.weights_hp.keys():
            
            class_num, weights, proba, proba_rest = self.weights_hp[hp_num]
            
            cut = self.cutter(test_, weights)
            test_ = test_[cut == 0]
            
            # Присваиваем вероятности только для отсеченных данной гиперплоскостью точек
            k = 0
            for i in range(predictions_np.shape[0]):
                if predictions_np[i, 1] == 0:
                    if cut[k] == 1:
                        predictions_np[i, 0] = proba
                        predictions_np[i, 1] = 1
                    k += 1
            
            if hp_num == list(self.weights_hp.keys())[-1]:
                predictions_np[(predictions_np[:, 1] == 0), 0] = proba_rest
                predictions_np[(predictions_np[:, 1] == 0), 1] = 1
        
        end_of_predict_time = time.time()
        print('Time taken to predict the targets: {0}'.format(end_of_predict_time - start_of_predict_time))
        
        return predictions_np[:, 0]
    
    def predict(self, X: np.array) -> np.array:
        """
            Makes predict for the sample

            :param X: матрица признаков, обучающая выборка
            :returns: предсказания классов
        """
        
        proba = self.predict_proba(X)
        return np.array(proba.astype(int))
    
    def get_reature_importances(self):
        return self.feature_importance
        
# df = pd.read_csv('../datasets/heart.csv')
# # print(data.head())
# # ones_col = np.ones(len(data)).reshape(-1, 1)
# # print(ones_col.reshape(-1, 1).shape)
# # print(data.shape)
# # print(np.hstack((data, ones_col)))
# X = df.drop(columns=['target']).values
# y = df['target'].values
#
# for i in range(X.shape[1]):
#     X[:,i]=(X[:,i]-X[:,i].min())/(X[:,i].max()-X[:,i].min())
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
