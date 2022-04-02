from nn_model import Model
from tqdm import tqdm
from PIL import ImageDraw, ImageOps
import PIL
import numpy as np
from tkinter import *



width, height = 168, 168


training_data = open('dataset/mnist_train.csv','r').readlines()
test_data = open('dataset/mnist_test.csv','r').readlines()


def prepare_data(data):
    inputs, targets = [], []

    for raw_line in tqdm(data, desc = 'preparing data'):

        line = raw_line.split(',')
    
        inputs.append(np.asfarray(line[1:])/255)
        targets.append(int(line[0]))

    return inputs, targets



training_inputs, training_targets = prepare_data(training_data)
test_inputs, test_targets = prepare_data(test_data)


#Define Model
model = Model()

model.add_input_layer(784, input_type = '1d')
model.add_dense_layer(neurons_number = 256, activation_func = 'Sigmoid', bias = 0)
model.add_dense_layer(neurons_number = 128, activation_func = 'Sigmoid', bias = 0)

#CNN model example
# model.add_conv2d_layer(kernels_number = 8, kernels_size = 5, input_size = 28, activation_func = 'Sigmoid', bias = 0)
# model.add_pooling_layer(2, 'MaxPooling')
# model.add_conv2d_layer(kernels_number = 32, kernels_size = 3, input_size = 12, activation_func = 'Sigmoid', bias = 0)
# model.add_pooling_layer(2, 'MaxPooling')

model.add_dense_layer(neurons_number = 10, activation_func = 'Sigmoid', bias = 0)

model.summary()

#Train and test Model
loss, acc = model.train(training_inputs, training_targets, epochs = 3, loss_function_name = 'MSE', optimizer_name = 'Nadam', batch_size = 100)
test_acc = model.test(test_inputs, test_targets)

#Plot loss and accuracy
# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(2)
# axs[0].plot(loss)
# axs[1].plot(acc)

# axs[0].set_ylabel('loss')
# axs[1].set_ylabel('accuracy')
# axs[1].set_xlabel('steps')
# plt.show()


# plt.plot(test_acc)
# plt.title('test accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('steps')
# plt.legend(['test'], loc='upper left')
# plt.show()

model.save('models/Simple Classifier')

model.load('models/Simple Classifier')

#Tkiner graphical interface methods
def recognize():
    inverted_img = ImageOps.invert(image)
    grayscaled_img = inverted_img.convert('L') 
    resized_img = grayscaled_img.resize((28,28), PIL.Image.ANTIALIAS)

    data = np.asarray(resized_img)
    inputs = np.reshape(data, (1,784))

    max_output_index, outputs = model.predict(inputs)

    probability = np.round(((np.max(outputs) / np.sum(outputs)) * 100), 2)

    print(f'{max_output_index} - {probability}%')

    lbl1['text'] = max_output_index
    lbl2['text'] = f'Probability: {probability}%'




def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill ="black", width = penSize_slider.get())
    draw.line([x1, y1, x2, y2], fill ="black", width = penSize_slider.get())



def clear():
    cv.delete("all")
    draw.rectangle((0, 0, 168, 168), fill=(255, 255, 255, 255))


    
root = Tk()
root.title("digits classifer") 


cv = Canvas(root, width=width, height=height, bg = 'white')
cv.pack()


image = PIL.Image.new("RGB", (width, height), color = 'white')
draw = ImageDraw.Draw(image)



cv.pack(side=RIGHT)
cv.bind("<B1-Motion>", paint)


button = Button(text = "Recognize", command = recognize, width = 20)
button2=Button(text = "Clear", command = clear, width = 20)
lbl0=Label(text = "Pen Size", font="Arial 10", width = 15)
lbl1=Label(text = " ", font = "Arial 30", fg = "red")
lbl2=Label(text = " ", font = "Arial 12", width = 15)

lbl0.pack()

penSize_slider = Scale(from_= 1, to = 10, orient = HORIZONTAL)
penSize_slider.pack()

button.pack()
button2.pack()

lbl1.pack()
lbl2.pack()

root.minsize(350, 200)
root.maxsize(350, 200)

root.mainloop()