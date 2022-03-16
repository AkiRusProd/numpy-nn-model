# numpy-nn-model (in the pipeline)
Реализация собственной модели нейронной сети на numpy, в которой вы можете добавить слои, выбирать различные параметры, а также  её тренировать

### Некоторая информация и возможности:

### Встроенные функции активаций:
1) Sigmoid
2) Tanh
3) ReLU
4) Leaky ReLU

#### Встроенные функции ошибок:
1) MSE
2) binary crossentropy
3) categorical crossentropy
4) minimax loss `(Для GAN)`

#### Встроенные оптимизаторы:
1) BGD
2) Momentum
3) RMSProp
4) Adam
5) Nadam

#### Инициализация модели:
(Для инициализации модели необходимо импортировать из `nn_model` скрипта класс `Model`)
```python
model = Model()
```

#### Слои:
Полносвязный слой; пример добавления слоя:

```python
model.add_dense_layer(neurons_number = 128, activation_func = 'Sigmoid', bias = 0)
```
Сверточный слой; пример добавления слоя:
```python
model.add_conv2d_layer(kernels_number = 4, kernels_size = 2, input_size = 3, activation_func = 'Sigmoid', padding = 1, bias = 0)
```
Дропаут слой; пример добавления слоя:
```python
model.add_dropout_layer(rate = 0.1)
```
Пулинговый слой; пример добавления:
```python
model.add_pooling_layer(block_size = 2, pooling_type = 'MaxPooling')
```
#### Тренировка Модели и получение "loss" и  "accuracy" метрик:
```python
loss, acc = model.train(inputs, targets, epochs = 100, loss_function_name = 'MSE', optimizer_name = 'Nadam', batch_size = 1, alpha = 0.001)
```
#### Тестирование Модели и получение "accuracy" метрики:
```python
test_acc = model.test(inputs, targets)
```
#### Представление Модели:
```python
model.summary()
```
#### Загрузка и сохранение Модели:
```python
model.load('Model Path')
model.save('Model Path')
```
### Примеры использования Моделей с использованием датасета Mnist:
#### Классификатор цифр (will be uploaded later)
#### Автоэнкодер (will be uploaded later)
#### Аппроксиматор частей функций (will be uploaded later)
#### Генеративно-состязательная сеть (GAN)

Пример генерации цифры 3: 

![Alt Text](https://raw.githubusercontent.com/AkiRusProd/numpy-nn-model/master/examples/generated%20images/3%20training%20process.gif)
