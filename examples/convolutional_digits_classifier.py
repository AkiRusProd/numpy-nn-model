import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nnmodel.layers import Dense, BatchNormalization, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D, Activation
from nnmodel import Model
from nnmodel.activations import LeakyReLU, ReLU

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

model = Model()
model.add(Reshape(shape = (1, 28, 28)))
model.add(Conv2D(kernels_num = 8, kernel_shape = (5, 5), stride = 2, activation = "relu"))
model.add(MaxPooling2D())
model.add(Conv2D(kernels_num = 32, kernel_shape = (3, 3), activation = LeakyReLU(alpha=0.02)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout())
model.add(Dense(units_num = 10, activation = None))
model.add(Activation(activation = "softmax"))

model.compile(optimizer = "adam", loss = "mse")
model.fit(training_inputs,  training_targets, epochs = 3, batch_size = 100)
model.predict(test_inputs, test_targets)

model.save("saved models/convolutional_digits_classifier")
model.load("saved models/convolutional_digits_classifier")



from tkinter import *
from PIL import ImageDraw, ImageOps
import PIL


width, height = 168, 168



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