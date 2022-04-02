# Numpy neural network Model (in the pipeline)
Реализация собственной модели нейронной сети на numpy, в которой вы можете добавить слои, выбирать различные параметры, а также  её тренировать

### Некоторая информация и возможности:

### Встроенные функции активаций:
1) Sigmoid
2) Tanh
3) ReLU
4) Leaky ReLU
5) ELU
6) GELU

#### Встроенные функции ошибок:
1) MSE
2) binary crossentropy
3) categorical crossentropy
4) minimax loss `(Для GAN)`

#### Встроенные оптимизаторы:
1) MBGD
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
model.add_conv2d_layer(kernels_number = 4, kernels_size = 2, input_size = 3, activation_func = 'Sigmoid', bias = 0)
```
Сверточный слой с добавленим рамки с нулями к входу; пример добавления слоя:
```python
model.add_conv2d_layer(kernels_number = 4, kernels_size = 2, input_size = 3, padding = 1, activation_func = 'Sigmoid', bias = 0)
```
Сверточный слой с транспонированием входа; пример добавления слоя:
```python
model.add_conv2d_layer(kernels_number = 4, kernels_size = 2, input_size = 3, transposing_stride = 2, activation_func = 'Sigmoid', bias = 0)
```

Сверточный слой с повышающей дискретизацией входа; пример добавления слоя:
```python
model.add_conv2d_layer(kernels_number = 4, kernels_size = 2, input_size = 3, upsampling_scale_factor = 2, activation_func = 'Sigmoid', bias = 0)
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
### Примеры этой Модели с использованием датасета Mnist:  
Скрипты:  
*[Непосредственно сама Модель](https://github.com/AkiRusProd/numpy-nn-model/blob/master/nn_model.py)*
#### Классификатор цифр

Скрипты:  
*[Классификатор](https://github.com/AkiRusProd/numpy-nn-model/blob/master/examples/classifier.py)*  

#### Аппроксиматор частей функций

Скрипты:  
*[Аппроксиматор](https://github.com/AkiRusProd/numpy-nn-model/blob/master/examples/approximator.py)*  

#### Обычный, а также Вариационный Автоэнкодер, устраняющие шумы (AE and VAE)

Скрипты:  
*[Обычный Автоэнкодер](https://github.com/AkiRusProd/numpy-nn-model/blob/master/examples/autoencoder.py)*  
*[Вариационный Автоэнкодер](https://github.com/AkiRusProd/numpy-nn-model/blob/master/examples/variational_autoencoder.py)*  
*[Модуль тренировки Вариационного Автоэнкодера](https://github.com/AkiRusProd/numpy-nn-model/blob/master/vae_trainer.py)*  

Пример зашумленных данных |  Пример данных с удаленным шумом
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/AkiRusProd/numpy-nn-model/master/examples/autoencoder%20images/ae%20noised%20set%20of%20images.jpeg)  |  ![](https://raw.githubusercontent.com/AkiRusProd/numpy-nn-model/master/examples/autoencoder%20images/ae%20denoised%20set%20of%20images.jpeg)

#### Генеративно-состязательная сеть (GAN)

Скрипты:  
*[GAN](https://github.com/AkiRusProd/numpy-nn-model/blob/master/examples/gan.py)*  
*[Модуль тренировки GAN](https://github.com/AkiRusProd/numpy-nn-model/blob/master/gan_trainer.py)*  

Пример генерации цифры 3   |  Пример интерполяции между изображениями
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/AkiRusProd/numpy-nn-model/master/examples/generated%20images/3%20training%20process.gif)  |  ![](https://raw.githubusercontent.com/AkiRusProd/numpy-nn-model/master/examples/generated%20images/images%20latent%20dim.gif)


### В планах:
1) Добавление возможности тренировать VAE (Добавлено)
2) Оптимизация и рефакторинг кода (Частично)
3) Написание версии с поддержкой numba для высокоскоростных вычислений (Частично)
