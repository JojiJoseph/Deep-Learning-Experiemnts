import matplotlib.pyplot as plt
import numpy as np

data = np.random.random((100,1))

data[data > 0.5] = data[data > 0.5] ** 2
plt.boxplot(data)
plt.show()

import seaborn as sns

sns.boxplot(data=data)
plt.show()