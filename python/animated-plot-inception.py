import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (640, 480))

x = np.linspace(0, 20,100)
y = np.sin(x)

img = None

for i in tqdm(range(20)):
    fig = plt.figure()

    if i == 0:
        plt.plot(x, y)
    else:
        plt.imshow(img)

    fig.canvas.draw() # https://stackoverflow.com/a/35362787

    # https://stackoverflow.com/a/7821917
    img = fig.canvas.tostring_rgb()
    img = np.frombuffer(img, dtype="uint8")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    writer.write(img)
    plt.close()

writer.release()