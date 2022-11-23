import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import shape

x = np.linspace(-10,10,10)
m = 2
b = 5

y = m*x + b

y_n = y + np.random.normal(scale=4,size=x.shape)
y_n[5] = 40
y_n[-1] = -50

median_best = np.inf
m_best = 0
b_best = 0
for iteration in range(100):
    m_sample = np.random.random()*5
    b_sample = np.random.random()*10
    y_pred = m_sample*x + b_sample
    median = np.median((y_pred - y_n)**2)
    if median < median_best:
        m_best = m_sample
        b_best = b_sample
        median_best = median
plt.scatter(x, y)
plt.scatter(x, y_n)

plt.plot(x, m_best*x + b_best)
plt.show()