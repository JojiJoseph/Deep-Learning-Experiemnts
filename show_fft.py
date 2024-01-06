from scipy.fft import fft, fft2, ifft2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

img = Image.open("lenna.png").convert("L")
img = img.resize((512, 512))
# img = img + np.random.randn(*img.size) * 10
# plt.imshow(img, cmap="gray")
# plt.show()

for start in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 257]:
    out = fft2(img)
    mag, phase = out.real, out.imag
    mag_sum = mag.sum()
    # mag_flatten = mag.flatten()
    # np.random.shuffle(mag_flatten)
    # mag = mag_flatten.reshape(mag.shape)
    temp = mag[start, start]
    temp_phase = phase[start, start]
    if start <= 512 - start:
        mag[start:512-start, :] = 0
        mag[:, start:512-start] = 0
        phase[start:512-start, :] = 0
        phase[:, start:512-start] = 0

    
    # phase[start:, start:] =, 0, 255 0
    mag[start, start] = temp
    phase[start, start] = temp_phase
    # mag[1:, 1:] = 0
    print(phase[0,0])
    
    # mag = mag / mag.sum() * mag_sum
    print(mag_sum)
    print(mag.shape, mag.min(), mag.max())
    img2 = ifft2(mag + phase*1j).real
    print("img means", np.mean(img), img2.mean(), np.max(img), img2.max(), np.min(img), img2.min())
    plt.subplot(1, 2, 1)
    plt.imshow(np.uint8(abs(img2)), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(img-img2).astype(np.uint8), cmap="gray")
    plt.show()