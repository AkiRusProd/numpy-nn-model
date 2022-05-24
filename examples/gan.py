from nn_model import Model
from gan_trainer import GAN
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio

training_data = open('dataset/mnist_train.csv','r').readlines()
test_data = open('dataset/mnist_test.csv','r').readlines()


x_num = 5
y_num = 5
margin = 15
image_size = 28
channels = 1
noise_vector_size = 100



def prepare_data(data, number_to_generate = None):
    inputs = []

    for raw_line in tqdm(data, desc = 'preparing data'):

        line = raw_line.split(',')

        if number_to_generate != None:
            if str(line[0]) == number_to_generate:
                inputs.append(np.asfarray(line[1:]) / 127.5 - 1)
        else:
            inputs.append(np.asfarray(line[1:]) / 127.5 - 1)
        
    return inputs

inputs = prepare_data(training_data, number_to_generate = None)




#Define generator and discriminator models
generator = Model()
discriminator = Model()

generator.add_input_layer(inputs_number = noise_vector_size, input_type = '1d')
generator.add_dense_layer(neurons_number = 128, activation_func = 'Leaky ReLU', bias = 0)
generator.add_dense_layer(neurons_number = 512, activation_func = 'Leaky ReLU', bias = 0)
generator.add_dense_layer(neurons_number = image_size * image_size * channels, activation_func = 'Tanh', bias = 0)

discriminator.add_input_layer(inputs_number = image_size * image_size * channels, input_type = '1d')
discriminator.add_dense_layer(neurons_number = 128, activation_func = 'Leaky ReLU', bias = 0)
discriminator.add_dense_layer(neurons_number = 64, activation_func = 'Leaky ReLU', bias = 0)
discriminator.add_dense_layer(neurons_number = 2, activation_func = 'Sigmoid', bias = 0)

gan_model = GAN(generator, discriminator)

gan_model.predict_per_epoch(mode = True, outputs_per_epoch = x_num * y_num)

#Train model and get loss, accuracy metrics
gen_loss, discr_loss = gan_model.train(inputs, epochs = 15, loss_function_name = 'minimax crossentropy', optimizer_name = 'Nadam', batch_size = 64, alpha = 0.001, beta = 0.5)






def get_images_set(images):
    '''Create set of images'''
    images_array = np.full((x_num * (margin + image_size), y_num * (margin + image_size), channels), 255, dtype=np.uint8)
    num = 0
    for i in range(y_num):
        for j in range(x_num):
            y = i*(margin + image_size)
            x = j*(margin + image_size)

            images_array[y:y+image_size,x:x+image_size] = images[num]
            num+=1

    images_array = images_array[: (y_num - 1) * (image_size + margin) + image_size, : (x_num - 1) * (image_size + margin) + image_size]

    if channels == 1:
        return Image.fromarray(images_array.squeeze(axis = 2)).convert("L")
    else:
        return Image.fromarray(images_array)



def generate_images(noise_vectors):
    '''Generate list of images from noise'''
    generated_images = []
    for i in range(len(noise_vectors)):
        generated_images.append(np.reshape(gan_model.predict(noise_vectors[i]), (image_size, image_size, channels)))

    generated_images = np.asfarray(generated_images) * 127.5 + 127.5

    return generated_images



noise_vectors = np.random.normal(0, 1, (x_num * y_num, noise_vector_size))

generated_images = generate_images(noise_vectors)

images = get_images_set(generated_images)
images.save(f'examples/generated images/set of images.jpeg')




def create_gif(images):
    '''Create gif between sets of images'''
    set_of_images=[]

    for one_epoch_images in images:
        set_of_images.append(get_images_set(one_epoch_images))
    
    imageio.mimsave(f'examples/generated images/training process.gif', set_of_images)



def get_each_epoch_images():
    '''Get set of images from each epoсhs and put them into list'''
    images_per_epoch = gan_model.all_outputs_per_epochs#
    for i in range(len(images_per_epoch)):
        for j in range(len(images_per_epoch[i])):
            images_per_epoch[i][j] = np.reshape(images_per_epoch[i][j], (image_size, image_size, channels))
            
    images_per_epoch = np.asarray(images_per_epoch) * 127.5 + 127.5

    return images_per_epoch

images_per_epoch = get_each_epoch_images()

create_gif(images_per_epoch)




def create_vectors_interpolation():
    '''Create vectors create interpolation  in the latent space between two sets of noise vectors'''
    steps = 10
    interval = 15
    images=[]
    

    noise_vectors_1 = np.random.normal(0, 1, (x_num * y_num, noise_vector_size))

    for step in range(steps):
        noise_vectors_2 = np.random.normal(0, 1, (x_num * y_num, noise_vector_size))

        noise_vectors_interp = (np.linspace(noise_vectors_1, noise_vectors_2, interval))

        noise_vectors_1 = noise_vectors_2

        for vectors in noise_vectors_interp:  
            generated_images = generate_images(vectors)

            images.append(get_images_set(generated_images))
    
    imageio.mimsave(f'examples/generated images/images latent dim.gif', images)

create_vectors_interpolation()





gan_model.save(f'models/GAN')



'''Build D and G losses'''
plt.plot([i for i in range(len(discr_loss))], discr_loss)
plt.plot([i for i in range(len(gen_loss))], gen_loss)

plt.xlabel("steps")
plt.ylabel("loss")
plt.legend(['Discriminator', 'Generator'])
plt.show()

