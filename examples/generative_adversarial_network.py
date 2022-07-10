import sys
sys.path.append("../nnmodel")
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from PIL import Image

from nnmodel.layers import Dense, BatchNormalization, Dropout, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Activation, RepeatVector, \
TimeDistributed, RNN, LSTM, GRU, Bidirectional
from nnmodel import Model
from nnmodel.activations import LeakyReLU
from nnmodel.optimizers import SGD, Adam, Nadam
from nnmodel.modules import GAN


training_data = open('dataset/mnist_train.csv','r').readlines()
test_data = open('dataset/mnist_test.csv','r').readlines()


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



data = prepare_data(training_data)
# data = prepare_data(test_data, '3')


x_num = 5
y_num = 5
margin = 15
image_size = 28
channels = 1
noise_vector_size = 100


generator = Model()
generator.add(Dense(128, input_shape = (noise_vector_size), use_bias=False))
generator.add(Activation('leaky_relu'))
generator.add(Dense(512, use_bias=False))
# generator.add(Dropout(0.2))
generator.add(Activation('leaky_relu'))
generator.add(Dense(784, use_bias=False))
generator.add(Activation('tanh'))

discriminator = Model()
discriminator.add(Dense(128, input_shape = (784), use_bias=False))
discriminator.add(Activation('leaky_relu'))
discriminator.add(Dense(64, use_bias=False))
discriminator.add(Activation('leaky_relu'))
discriminator.add(Dense(2, use_bias=False))
discriminator.add(Activation('sigmoid'))


# discriminator.add(Reshape(shape = (28, 28)))
# discriminator.add(Bidirectional(RNN(32, input_shape=(28, 28), return_sequences=False, cycled_states = True)))
# discriminator.add(RepeatVector(28))
# discriminator.add(TimeDistributed(Dense(50)))
# discriminator.add(TimeDistributed(BatchNormalization()))
# discriminator.add(Bidirectional(RNN(16,  cycled_states = True)))
# discriminator.add(Dense(2, activation='sigmoid'))

gan = GAN(generator, discriminator)
gan.compile(optimizer = Nadam(alpha = 0.001, beta = 0.5), loss = 'minimax_crossentropy', each_epoch_predict={"mode": True, "num" : x_num * y_num})
G_loss, D_loss = gan.fit(data, epochs = 30, batch_size = 64, noise_vector_size = noise_vector_size)





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
    # generated_images = []
    # for i in range(len(noise_vectors)):
    generated_images = np.reshape(gan.predict(noise_vectors), (x_num * y_num, image_size, image_size, channels))

    generated_images = np.asfarray(generated_images) * 127.5 + 127.5

    return generated_images



noise_vectors = np.random.normal(0, 1, (x_num * y_num, noise_vector_size))

generated_images = generate_images(noise_vectors)

images = get_images_set(generated_images)
images.save(f'examples/generated images/set of images.jpeg')




def create_gif(images):
    '''Create gif between sets of images'''
    set_of_images=[]

    for i in range(len(images)):
        set_of_images.append(get_images_set(images[i]))
        
    imageio.mimsave(f'examples/generated images/training process.gif', set_of_images)



def get_each_epoch_images():
    '''Get set of images from each epoсhs and put them into list'''
    images_per_epoch = gan.get_each_epoch_predictions()

    reshaped_images_per_epoch = [[[] for j in range(len(images_per_epoch[i]))] for i in range(len(images_per_epoch))]
    for i in range(len(images_per_epoch)):
        for j in range(len(images_per_epoch[i])):
            reshaped_images_per_epoch[i][j] = np.reshape(images_per_epoch[i][j], (image_size, image_size, channels))
            
    reshaped_images_per_epoch = np.asarray(reshaped_images_per_epoch) * 127.5 + 127.5

    return reshaped_images_per_epoch

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





gan.save(f'saved models/GAN')



'''Build D and G losses'''
plt.plot([i for i in range(len(D_loss))], D_loss)
plt.plot([i for i in range(len(G_loss))], G_loss)

plt.xlabel("steps")
plt.ylabel("loss")
plt.legend(['Discriminator', 'Generator'])
plt.show()


