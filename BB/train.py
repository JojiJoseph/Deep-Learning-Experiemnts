import tensorflow.keras as keras
from tensorflow.keras import metrics
import tensorflow.keras.layers as layers
import numpy as np


class Model(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c1 = layers.Conv2D(128, 5, activation="relu")
        self.c2 = layers.Conv2D(128, 5, activation="relu")
        self.c3 = layers.Conv2D(128, 5, activation="relu")
        self.flatten = layers.Flatten()
        self.l1 = layers.Dense(1024,"relu")
        self.l2 = layers.Dense(4,"sigmoid")
    def call(self,x):
        y = self.c1(x)
        y = self.c2(y)
        y = self.c3(y)
        y = self.flatten(y)
        y = self.l1(y)
        y = self.l2(y)
        return y

model = Model()

model.compile(optimizer="adam", loss="mae")

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

y_train = []
for xi in x_train:
    a = np.arange(0, 28)
    # Find width and cx
    img = xi.copy()
    img = np.sum(img, axis=0)
    width = np.sum(img>0)
    cx = np.sum((img>0)*a)/width

    # Find width and cx
    img = xi.copy()
    img = np.sum(img, axis=1)
    height = np.sum(img>0)
    cy = np.sum((img>0)*a)/height

    y_train.append([width/28, height/28,cx/28,cy/28])

y_train = np.array(y_train)
x_train = x_train.reshape((-1,28,28,1))/255.
x_test = x_test.reshape((-1,28,28,1))/255.
# x_train = x_train.reshape((-1,784))
# x_test = x_test.reshape((-1,784))
model.fit(x_train, y_train, epochs=10, batch_size=None)
import matplotlib.pyplot as plt
import cv2
for i, x in enumerate(x_test[:5]):
    # print(x.shape)
    w,h,cx, cy = model.predict(np.array([x]))[0]
    w = w*28
    h = h*28
    cx = cx*28
    cy = cy*28
    plt.subplot(1,5,i+1)
    x = np.uint8(x*255)
    # x = x.reshape((28,28,1))
    cv2.rectangle(x, (int(cx-width/2),int(cy-height/2)),(int(cx+width/2),int(cy+height/2)),(255,0,0),1)
    plt.imshow(x, cmap="gray")
plt.show()