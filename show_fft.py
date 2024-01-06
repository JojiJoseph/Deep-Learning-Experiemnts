from scipy.fft import fft, fft2, ifft2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

img = Image.open("lenna.png").convert("L")
# img = img + np.random.randn(*img.size) * 10
# plt.imshow(img, cmap="gray")
# plt.show()

for start in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    out = fft2(img)
    mag, phase = out.real, out.imag
    mag_sum = mag.sum()
    # mag_flatten = mag.flatten()
    # np.random.shuffle(mag_flatten)
    # mag = mag_flatten.reshape(mag.shape)
    mag[start:, start:] = 0
    phase[start:, start:] = 0
    mag = mag / mag.sum() * mag_sum
    print(mag_sum)
    print(mag.shape, mag.min(), mag.max())
    img2 = ifft2(mag + phase*1j)
    plt.subplot(1, 2, 1)
    plt.imshow(abs(img2.real), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(img-abs(img2.real), cmap="gray")
    plt.show()