# Изучение технологий машинного обучения

### Задачи выполняемые с помощью машинного обучения
- Восстановление зависимоти (регрессия)
- Бинарная классификация
- Многоклассовая классификация
- Локализация (поиск объекта на изображении)
- Сегментация (отделение объекта от фона)
- Сжатие размерности
- SuperResolution (увеличение расширения)

### Подготовка данных к машинному обучению
Подготовка данных к машинному обучению происходит в основном с помощью библотеки pandas. В файле [PandasTutorial.ipynb](https://github.com/Yan-Minotskiy/mlstepik/blob/master/PandasTutorial.ipynb) приведены основные команды для обработки массивов данных.

### Визуализация данных
Основные формы визуализации данных в файле [Visualisation.ipynb](https://github.com/Yan-Minotskiy/mlstepik/blob/master/Visualisation.ipynb).
Вот мои примеры применения визуализации:

Визуализация решения задачи классификации [classificator.py](https://github.com/Yan-Minotskiy/mlstepik/blob/master/classificator.py "classificator.py")
![](https://github.com/Yan-Minotskiy/mlstepik/blob/master/img/classificator_result.png)

Значения функиии потерь и точности предсказания во время обучения в задаче [mnist_net.py](https://github.com/Yan-Minotskiy/mlstepik/blob/master/mnist_net.py)
![](https://github.com/Yan-Minotskiy/mlstepik/blob/master/img/loss_function_mnist_first_epoch.png)
![](https://github.com/Yan-Minotskiy/mlstepik/blob/master/img/mnist_accuracy.png)

Результат обучения сети для регрессии [regression.py](https://github.com/Yan-Minotskiy/mlstepik/blob/master/regression.py "regression.py")

![](https://github.com/Yan-Minotskiy/mlstepik/blob/master/img/regression_learning_outcome.png)

### Метрики

**Confusion matrix**

| predict\target  | Yt=1  | Yt=0  |
| :------------: | :------------: | :------------: |
| Yp = 1  | TP  | FP  |
| Yp = 0  | FN  |  TN |

**Accuracy** — доля правильных ответов алгоритма.

![](https://latex.codecogs.com/gif.latex?Accuracy%20%3D%20%5Cfrac%7BTP&plus;TN%7D%7BTP&plus;TN&plus;FP&plus;FN%7D)

**Precision** - доля объектов, названных классификатором положительными и при этом действительно являющимися положительными

![](https://latex.codecogs.com/gif.latex?Precision%3D%5Cfrac%7BTP%7D%7BTP&plus;FP%7D)

**Recall** - долю объектов положительного класса из всех объектов положительного класса, которую нашел алгоритм

![](https://latex.codecogs.com/gif.latex?Recall%3D%5Cfrac%7BTP%7D%7BTP&plus;FN%7D)

Значение F-меры в частном случае

![](https://latex.codecogs.com/gif.latex?F%20%3D%20%5Cfrac%7B2%20%5Ccdot%20Precision%20%5Ccdot%20Recall%7D%7BPrecision%20&plus;%20Recall%7D)

[Хорошая статья по этой теме](https://habr.com/ru/company/ods/blog/328372/)

### Функции активации
Функция активации (активационная функция, функция возбуждения) – функция, вычисляющая выходной сигнал искусственного нейрона.
#### Пороговая функция активации
![](https://latex.codecogs.com/gif.latex?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%200%20%26%20x%3E0%5C%5C%201%20%26%20x%5Cleq%200%20%5Cend%7Bmatrix%7D%5Cright.)
#### Сигмоида
![](https://latex.codecogs.com/gif.latex?%5Csigma%20%28x%29%5Cfrac%7B1%7D%7B1&plus;e%5E%5Cfrac%7B-x%7D%7Bt%7D%7D) , где t - коэффициент "плавности" сигмоиды
Если в сети много подряд идущих сигмоид происходит затхание градиента.
#### SoftMax
#### Гиперболический тангенс
#### ReLU
#### ELU
#### L-ReLU
#### SeLU

Сравнение эффективности функций активации про решении одной из задач курса "Нейронные сети и компьютерное зрение"
![](https://github.com/Yan-Minotskiy/mlstepik/blob/master/img/func_act_test.png)

### Функции потерь
Функция потерь — функция, которая в теории статистических решений характеризует потери при неправильном принятии решений на основе наблюдаемых данных.
#### Средний квадрат ошибки
![](https://latex.codecogs.com/gif.latex?MSE%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28%5Cbreve%7By%7D_%7Bi%7D-y_%7Bi%7D%29)

`torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')`

![](https://latex.codecogs.com/gif.latex?MSE%27%3D2%5Csigma%20%28%5Csigma%20-t%29%281-%5Csigma%20%29)

Проблема: паралич сигмоидной функции (функция потерь может быть равна 0 не в оптимальных случаях)
#### Бинарная кросс-энтропия
![](https://latex.codecogs.com/gif.latex?BCE%28p%2C%20t%29%20%3D%20-t%20%5Ccdot%20log%28p%29-%281-t%29log%281-p%29)

![](https://latex.codecogs.com/gif.latex?p%20%3D%20%5Csigma%20%28y%29)

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20BCE%7D%7B%5Cpartial%20y%7D%3D%5Csigma%20-%20t)

`torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')`
#### Кросс-энтропия
![](https://latex.codecogs.com/gif.latex?CE%28p%2C%20t%29%3D-%5Csum_%7Bc%3D1%7D%5E%7BN%7Dt_%7Bc%7Dlog%28p_%7Bc%7D%29)
если p=SM, то

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20CE%7D%7B%5Cpartial%20y_%7Bi%7D%7D%3D-t_%7Bi%7D&plus;p_%7Bi%7D)

`torch.nn.CrossEntropyLoss()`

[Документация по функциям потерь в PyTorch](https://pytorch.org/docs/stable/nn.html#loss-functions)
### Методы оптимизации
Методы, задача которых найти минимальное значение функции потерь.
#### Стахостический градиентный спуск
Особенности: 
 - находит только глобальный минимум
 -  долго сходится в ситуациях, когда функция потерь зависит гораздо больше от одних параметров, чем от других
 -  функция потерь должна быть дифференцируема
 -  производная по функции потерь не должна быть равна 0 (кроме случаев, когда мы нашли оптимальные значения)
 
![](https://latex.codecogs.com/gif.latex?w_%7Bi%7D%20%3D%20w_%7Bi%20-%201%7D-%5Calpha%20%5Cbigtriangledown%20f%28w_%7Bi%20-%201%7D%29)

#### Градиентный спуск с импульсом

![](https://latex.codecogs.com/gif.latex?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20w%5E%7Bt&plus;1%7D%3Dw%5E%7Bt%7D&plus;%5Calpha%20%5Cvartheta%20t%5C%5C%20%5Cvartheta%20%5E%7Bt&plus;1%7D%3D%5Cvartheta%20%5E%7Bt%7D%5Ccdot%20%5Cbeta%20-%5Cbigtriangledown%20f%20%5Cend%7Bmatrix%7D%5Cright.)
#### Экспоненциальное скользящее среднее (EMA)

![](https://latex.codecogs.com/gif.latex?EMA_%7B%5Cbeta%20%7D%28f%29%5E%7Bt%7D%3D%281-%5Cbeta%20%29f%5E%7Bt%7D&plus;pEMA_%7B%5Cbeta%20%7D%28f%29%5E%7Bt-1%7D)

#### Rprop
#### RMSprop
#### Adam
`torch.optim.Adam(<веса>, lr=<шаг оптимизатора>)`

####Планировщик
Механизм изменение шага обучения во время процесса обучения


### Свёрточные нейронные сети


## Список полезных источников

### Курсы
1. Курс лекций по машинному обучению К. В. Воронцова - http://www.machinelearning.ru/wiki/index.php?title=Машинное_обучение_%28курс_лекций%2C_К.В.Воронцов%29
2. Курс "Введение в Data Science и машинное обучение" на платформе Stepik - https://stepik.org/course/4852
3. Курс "Нейронные сети и компьютерное зрение" на платформе Stepik - https://stepik.org/course/50352

### Документация к библиотекам
1. Pandas - https://pandas.pydata.org/
2. PyTorch - https://pytorch.org/
3. Scikit learn - https://scikit-learn.org/stable/index.html
4. Matplotlib - https://matplotlib.org/
5. Saborn - https://seaborn.pydata.org/

### Платформа для практики в машинном обучении
Kaggle - https://www.kaggle.com/
