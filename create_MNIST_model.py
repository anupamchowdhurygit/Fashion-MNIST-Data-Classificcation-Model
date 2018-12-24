import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense

img_rows, img_cols = 28,28
num_classes = 10

def prep_data(raw):
    y = raw[:,0]
    out_y = keras.utils.to_categorical(y, num_classes)

    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows,img_cols,1)
    out_x = out_x/255
    return out_x, out_y

fashion_file = '/Users/anupamchowdhury/PycharmProjects/Fashion-MNIST-Data-Classificcation-Model/data/fashion-mnist_train.csv'
fashion_data = np.loadtxt(fashion_file,skiprows=1, delimiter=',')
x,y = prep_data(fashion_data)

fashion_model = Sequential()
fashion_model.add(Conv2D(12,activation='relu',kernel_size=3,input_shape=(img_rows,img_cols,1)))
fashion_model.add(Conv2D(20, activation='relu', kernel_size=3))
fashion_model.add(Conv2D(20, activation='relu', kernel_size=3))
fashion_model.add(Flatten())
fashion_model.add(Dense(100, activation='relu'))
fashion_model.add(Dense(10, activation='softmax'))

fashion_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

fashion_model.fit(x, y, batch_size=100, epochs=4, validation_split=0.2)
