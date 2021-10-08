# RANSAC Circle
import numpy as np
import cv2
import matplotlib.pyplot as plt

x0, y0 = 10, 10
r = 10

theta = np.linspace(0, 2*np.pi, 23)

x = x0 + r*np.cos(theta) + np.random.normal(0, 1, size=theta.shape)
y = y0 + r*np.sin(theta) + np.random.normal(0, 1, size=theta.shape)
# y[2] = 100

plt.scatter(x, y)


def ransac_circle(x, y, iter=1000, eps=0.5):
    best_x0, best_y0 = 0, 0
    best_n_out = np.inf
    best_r = 1
    for i in range(iter):
        idx, idx2, idx3 = np.random.randint(len(x), size=(3,))
        xa, ya = x[idx], y[idx]
        xb, yb = x[idx2], y[idx2]
        xc, yc = x[idx3], y[idx3]
        A = np.array([[2*(xa-xb), 2*(ya-yb)],
                      [2*(xa-xc), 2*(ya-yc)]])
        Y = np.array([(xa**2+ya**2-xb**2-yb**2),
                     (xa**2+ya**2-xc**2-yc**2)]).reshape((-1, 1))
        try:
            x0, y0 = np.linalg.inv(A) @ Y
        except np.linalg.LinAlgError:
            continue
        n_out = 0
        r = np.sqrt((xa-x0)**2 + (ya-y0)**2)
        for i in range(len(x)):
            if not (r - eps < np.sqrt((x[i]-x0)**2 + (y[i]-y0)**2) < eps + r):
                n_out += 1
        if n_out < best_n_out:
            best_n_out = n_out
            best_x0, best_y0 = x0, y0
            best_r = r
    print(best_n_out)
    return best_x0, best_y0, best_r


xn0, yn0, r = ransac_circle(x, y)
theta = np.linspace(0, 2*np.pi, 100)
x = xn0 + r*np.cos(theta)
y = yn0 + r*np.sin(theta)
plt.axis("equal")
plt.plot(x, y)
plt.scatter(xn0, yn0)
print(xn0, yn0, r)
plt.show()
