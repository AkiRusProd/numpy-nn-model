from nn_model import Model
from tqdm import tqdm
from PIL import Image
import numpy as np



training_data = open('dataset/mnist_train.csv','r').readlines()
test_data = open('dataset/mnist_test.csv','r').readlines()


def prepare_data(data, number_to_take = None):
    inputs = []

    for raw_line in tqdm(data, desc = 'preparing data'):

        line = raw_line.split(',')

        if number_to_take != None:
            if str(line[0]) == number_to_take:
                inputs.append(np.asfarray(line[1:]) / 255)
        else:
            inputs.append(np.asfarray(line[1:]) / 255)
        
    return inputs


inputs = np.asfarray(prepare_data(training_data, number_to_take = None))



def add_noise(data):
    noise_factor = 0.5

    noisy_data = data + noise_factor * np.random.normal(0, 1, (data.shape))

    return np.clip(noisy_data, 0, 1)

noisy_inputs = add_noise(inputs)


#Define Model
model = Model()

# model.add_input_layer(784, input_type = '1d')
# model.add_dense_layer(neurons_number = 256, activation_func = 'Sigmoid', bias = 0)
# model.add_dense_layer(neurons_number = 128, activation_func = 'Sigmoid', bias = 0)
# model.add_dense_layer(neurons_number = 64, activation_func = 'ReLU', bias = 0)
# model.add_dense_layer(neurons_number = 128, activation_func = 'Sigmoid', bias = 0)
# model.add_dense_layer(neurons_number = 256, activation_func = 'Sigmoid', bias = 0)
# model.add_dense_layer(neurons_number = 784, activation_func = 'Sigmoid', bias = 0)

# #Train model
# model.train(noisy_inputs, inputs, epochs = 20, loss_function_name = 'MSE', trained_model = 'encoder', batch_size = 65,  optimizer_name = 'Adam')

# model.save('models/Denoising Autoencoder')

model.load('models/Denoising Autoencoder')

x_num = 5
y_num = 5
margin = 15
image_size = 28



def get_images_set(images):
    '''Create set of images'''
    images_array = np.full((x_num * (margin + image_size), y_num * (margin + image_size)), 255, dtype=np.uint8)
    num = 0
    for i in range(y_num):
        for j in range(x_num):
            y = i*(margin + image_size)
            x = j*(margin + image_size)

            images_array[y:y+image_size,x:x+image_size] = images[num]
            num+=1

    images_array = images_array[: (y_num - 1) * (image_size + margin) + image_size, : (x_num - 1) * (image_size + margin) + image_size]

    return Image.fromarray(images_array).convert("L")


noised_images, denoised_images = [], []

for i in range(x_num * y_num):
    rand_i = np.random.randint(0, len(inputs))

    _, outputs = model.predict(noisy_inputs[rand_i])

    noised_images.append(np.reshape(noisy_inputs[rand_i], (image_size, image_size)) * 255)
    denoised_images.append(np.reshape(outputs, (image_size, image_size)) * 255)

get_images_set(noised_images).save(f'examples/autoencoder images/ae noised set of images.jpeg')
get_images_set(denoised_images).save(f'examples/autoencoder images/ae denoised set of images.jpeg')






# plot_2d_latent_space(15)

# def plot_2d_distribution(one_axis_samples_num = 15): NotImplementedYet

