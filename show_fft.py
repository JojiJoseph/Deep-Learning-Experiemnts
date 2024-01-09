from scipy.fft import fft, fft2, ifft2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

img = Image.open("cat.jpg").convert("L")
img = img.resize((512, 512))
# img = img + np.random.randn(*img.size) * 10
# plt.imshow(img, cmap="gray")
# plt.show()

for start in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 257]:
    out = fft2(img)
    mag, phase = out.real, out.imag
    mag_copy = mag.copy()
    phase_copy = phase.copy()
    mag_sum = mag.sum()
    # mag_flatten = mag.flatten()
    # np.random.shuffle(mag_flatten)
    # mag = mag_flatten.reshape(mag.shape)
    temp = mag[start, start]
    temp_phase = phase[start, start]
    if 0 <= 512 - start < 512:
        temp2 = mag[512-start, 512-start]   
        temp_phase2 = phase[512-start, 512-start]
        mag[start:512-start, :] = 0
        mag[:, start:512-start] = 0
        phase[start:512-start, :] = 0
        phase[:, start:512-start] = 0

    mag_copy[:, :] = 0
    phase_copy[:, :] = 0
    mag_copy[start, start] = temp
    phase_copy[start, start] = temp_phase
    if 0 <= 512 - start < 512:
        mag_copy[512-start, 512-start] = temp2
        phase_copy[512-start, 512-start] = temp_phase2
        mag_copy[512-start, 512-start] = temp2
        phase_copy[512-start, 512-start] = temp_phase2

    
    # phase[start:, start:] =, 0, 255 0
    mag[start, start] = temp
    phase[start, start] = temp_phase
    # mag[1:, 1:] = 0
    print(phase[0,0])
    
    # mag = mag / mag.sum() * mag_sum
    print(mag_sum)
    print(mag.shape, mag.min(), mag.max())
    img2 = ifft2(mag + phase*1j).real
    img3 = ifft2(mag_copy + phase_copy*1j).real
    print("img means", np.mean(img), img2.mean(), np.max(img), img2.max(), np.min(img), img2.min())
    plt.subplot(1, 3, 1)
    plt.imshow(np.uint8(abs(img2)), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(img-img2).astype(np.uint8), cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(np.uint8(abs(img3)), cmap="gray")
    plt.show()