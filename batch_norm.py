import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

class BatchNorm(keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def build(self, input_shape):
        # return super().build(input_shape)
        self.gamma = self.add_weight(shape=[input_shape[-1]], initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(shape=[input_shape[-1]], initializer=tf.initializers.zeros, trainable=True)
        self.moving_mean = self.add_weight(shape=[input_shape[-1]],initializer=tf.initializers.zeros, trainable=False)
        self.moving_var = self.add_weight(shape=[input_shape[-1]],initializer=tf.initializers.ones, trainable=False)
        super(BatchNorm, self).build(input_shape)
    def call(self, x, training):
        if training:
            mean = tf.reduce_mean(x, axis=0, keepdims=False)
            var = tf.reduce_mean((x - tf.stop_gradient(mean))**2, axis=0, keepdims=False)
            # print(mean.shape, var.shape)
            self.moving_mean.assign(self.moving_mean*0.9 + 0.1* tf.stop_gradient(mean))
            self.moving_var.assign((self.moving_var*0.9 + 0.1* tf.stop_gradient(var)))
            return self.gamma * (x-mean)/(tf.sqrt(var)+1e-9)+self.beta
        else:
            return self.gamma * (x-self.moving_mean)/(tf.sqrt(self.moving_var)+1e-9)+self.beta


model1 = keras.models.Sequential(
    [
        layers.Dense(1024, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="sigmoid"),
        
    ]
)

model1.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model2 = keras.models.Sequential(
    [
        layers.Dense(1024, activation="relu", use_bias=False),
        layers.BatchNormalization(momentum=0.9),
        layers.Dense(512, activation="relu", use_bias=False),
        layers.BatchNormalization(momentum=0.9),
        layers.Dense(256, activation="relu", use_bias=False),
        layers.BatchNormalization(momentum=0.9),
        layers.Dense(128, activation="relu", use_bias=False),
        layers.BatchNormalization(momentum=0.9),
        layers.Dense(10, activation="sigmoid"),
        
    ]
)

model3 = keras.models.Sequential(
    [
        layers.Dense(1024, activation="relu", use_bias=False),
        BatchNorm(),
        layers.Dense(512, activation="relu", use_bias=False),
        BatchNorm(),
        layers.Dense(256, activation="relu", use_bias=False),
        BatchNorm(),
        layers.Dense(128, activation="relu", use_bias=False),
        BatchNorm(),
        layers.Dense(10, activation="sigmoid"),
        
    ]
)

model2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model3.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


(x_train, y_train),(x_test, y_test)  = keras.datasets.cifar10.load_data()
print(x_train.shape)
x_train = x_train.reshape((-1,1024*3))/255.
x_test = x_test.reshape((-1,1024*3))/255.

hist3 = model3.fit(x_train, y_train, batch_size=100,validation_data=(x_test,y_test), epochs=100)
hist1 = model1.fit(x_train, y_train, batch_size=100,validation_data=(x_test,y_test), epochs=100)

hist2 = model2.fit(x_train, y_train, batch_size=100,validation_data=(x_test,y_test), epochs=100)



import matplotlib.pyplot as plt
plt.plot(hist1.history['accuracy'], label="model1")
plt.plot(hist2.history['accuracy'], label="model2")
plt.plot(hist3.history['accuracy'], label="model3")
plt.legend()
plt.show()
plt.plot(hist1.history['val_accuracy'], label="model1")
plt.plot(hist2.history['val_accuracy'], label="model2")
plt.plot(hist3.history['val_accuracy'], label="model3")
plt.legend()
plt.show()
# print(hist1,hist1.history, dir(hist1.history))