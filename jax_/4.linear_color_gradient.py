import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time as time
import jax

img = np.zeros((1024, 1024, 3)).astype(np.uint8)

t1 = time.time()
x1, y1 = 100, 100
x2, y2 = 1000, 1000

color1 = np.array([255, 255, 0])
color2 = np.array([0, 0, 255])

length = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
def get_color(x, y):
    alpha = ((x2-x1)*(x-x1) + (y2-y1)*(y-y1)) / length / length
    return color1 * alpha + color2 * (1-alpha)

for i in range(1024):
    for j in range(1024):
        img[i, j] = np.clip(np.int32(get_color(j, i)), 0, 255)

t2 = time.time()
print("Time taken: ", t2-t1)
plt.imshow(img)
plt.show()

# Vectorize version
t1 = time.time()
x1, y1 = 100, 100
x2, y2 = 1000, 1000

color1 = np.array([255, 255, 0])
color2 = np.array([0, 0, 255])

length = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))

def get_color_vec(x, y):
    alpha = ((x2-x1)*(x-x1) + (y2-y1)*(y-y1)) / length / length
    alpha  = alpha[..., None]
    return color1 * alpha + color2 * (1-alpha)

x = np.arange(1024)
y = np.arange(1024)
X, Y = np.meshgrid(x, y)
img = np.clip(np.int32(get_color_vec(X, Y)), 0, 255)
t2 = time.time()
print("Time taken: ", t2-t1)
plt.imshow(img)
plt.show()

# jnp version
t1 = time.time()
x1, y1 = 100, 100
x2, y2 = 1000, 1000

color1 = jnp.array([255, 255, 0])
color2 = jnp.array([0, 0, 255])

length = jnp.linalg.norm(jnp.array([x2, y2]) - jnp.array([x1, y1]))

@jax.jit
def get_color_vec(x, y):
    alpha = ((x2-x1)*(x-x1) + (y2-y1)*(y-y1)) / length / length
    alpha  = alpha[..., None]
    return color1 * alpha + color2 * (1-alpha)

x = jnp.arange(1024)
y = jnp.arange(1024)
X, Y = jnp.meshgrid(x, y)
img = jnp.clip(jnp.int32(get_color_vec(X, Y)), 0, 255)
t2 = time.time()
print("Time taken: ", t2-t1)
plt.imshow(img)
plt.show()

# 2nd call
t1 = time.time()
x = jnp.arange(1024)
y = jnp.arange(1024)
X, Y = jnp.meshgrid(x, y)
img = jnp.clip(jnp.int32(get_color_vec(X, Y)), 0, 255)
t2 = time.time()
print("Time taken: ", t2-t1)
plt.imshow(img)
plt.show()

# 3rd call
t1 = time.time()
x = jnp.arange(1024)
y = jnp.arange(1024)
X, Y = jnp.meshgrid(x, y)
img = jnp.clip(jnp.int32(get_color_vec(X, Y)), 0, 255)
t2 = time.time()
print("Time taken: ", t2-t1)
plt.imshow(img)
plt.title("3rd call with jax")
plt.show()

# # vmapped version
# get_color_vmapped = jax.vmap(get_color, in_axes=(0, 0))

# t1 = time.time()
# x = jnp.arange(1024)
# y = jnp.arange(1024)
# X, Y = jnp.meshgrid(x, y)
# img = jnp.clip(jnp.int32(get_color_vmapped(X, Y)), 0, 255)
# t2 = time.time()
# print("Time taken: ", t2-t1)
# plt.imshow(img)
# plt.show()
