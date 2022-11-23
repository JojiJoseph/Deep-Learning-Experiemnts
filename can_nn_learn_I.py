import numpy as np
import tensorflow
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

model = keras.models.Sequential([layers.Dense(10, use_bias=False)])

model.compile(optimizer="adam", loss="mae")

x = np.random.uniform(-5,5, (10_000, 10))

model.fit(x, x, epochs=10)

print(model.layers[0].weights[0].shape)
print(np.round(model.layers[0].weights[0]*100)/100)

# Result - neural network can learn identity function