import time
import jax
import matplotlib.pyplot as plt
import tqdm

import numpy as np
import jax.numpy as jnp

img = np.zeros((1024, 1024, 3)).astype(np.uint8)

t1 = time.time()

for i in tqdm.tqdm(range(1024)):
    for j in range(1024):
        a, b = 0, 0
        for it in range(10):
            a = a**2 - b**2 + i/100-512/100
            b = 2*a*b + j/100-512/100
            if a**2 + b**2 >= 4:
                break
        if a**2 + b**2 < 4:
            img[i, j] = [255, 255, 255]
t2  = time.time()

print("Time taken: ", t2-t1)
plt.imshow(img)
plt.show()
t1 = time.time()
img = np.ones((10240, 10240, 3)).astype(np.uint8) * 255

II, JJ = np.meshgrid(np.arange(10240), np.arange(10240))
img_a = np.zeros((10240, 10240)).astype(np.float32)
img_b = img_a.copy()
for it in tqdm.tqdm(range(20)):
    img_a = img_a**2 -img_b**2 + II/1000-512/100
    img_b = 2*img_a*img_b + JJ/1000-512/100
    mask = (img_a**2 + img_b**2 > 4) & (img[:,:,0] == 255)#& (np.prod(img[...,:] == [255, 255, 255],axis=-1))
    img[mask] = [(it+1)*12.5, 255-(it+1)*12.25, (it+1)*12.25]
t2 = time.time()
print("Time taken: ", t2-t1)
plt.imshow(img)
plt.show()


t1 = time.time()
img = np.ones((10240, 10240, 3)).astype(np.uint8) * 255

II, JJ = jnp.meshgrid(np.arange(10240), np.arange(10240))
img_a = jnp.zeros((10240, 10240)).astype(np.float32)
img_b = img_a.copy()


def iterate(a, b, II, JJ):
    a = a**2 -b**2 + II/1000-512/100
    b = 2*a*b + JJ/1000-512/100
    return a, b
iterate = jax.jit(iterate, device=jax.devices('gpu')[0])
for it in range(20):
    # img_a = img_a**2 -img_b**2 + II/100-512/100
    # img_b = 2*img_a*img_b + JJ/100-512/100
    img_a, img_b = iterate(img_a, img_b, II, JJ)
    mask = (img_a**2 + img_b**2 > 4) & (img[:,:,0] == 255)#& (np.prod(img[...,:] == [255, 255, 255],axis=-1))
    # img.at[mask].set([(it+1)*12.5, 255-(it+1)*12.25, (it+1)*12.25])
    img[mask] = [(it+1)*12.5, 255-(it+1)*12.25, (it+1)*12.25]

# mask = img_a**2 + img_b**2 < 4
# img[mask] = [255, 255, 255]
t2 = time.time()
print("Time taken: ", t2-t1)
plt.imshow(img)
plt.show()
print(jax.devices('gpu'))