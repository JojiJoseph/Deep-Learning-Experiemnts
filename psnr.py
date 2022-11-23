from skimage.metrics import peak_signal_noise_ratio
import cv2
import numpy as np

def psnr(img_true, img_test):
    mse = ((img_true.astype(float)-img_test.astype(float))**2).mean()
    peak = 255
    return 20*np.log10(peak) - 10*np.log10(mse)

if __name__ == "__main__":
    # Download image of Lena Forsén from wiki and save it as lenna.png
    img_org = cv2.imread("lenna.png")
    img_down = cv2.resize(img_org, (128, 128))
    img_up = cv2.resize(img_down, (512, 512))
    custom_psnr = psnr(img_org, img_up)
    skimage_psnr = peak_signal_noise_ratio(img_org, img_up)

    print("PSNR by custom implementation: ", custom_psnr)
    print("PSNR by skimage: ", skimage_psnr)
    print("Diff: ", custom_psnr-skimage_psnr)
