import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

mean1 = [0, 0]
std1 = [[1, 0], [0, 2]]

mean2 = [10, 0]
std2 = [[2, 0], [0, 1]]

points1 = np.random.multivariate_normal(mean1, std1, 100)
points2 = np.random.multivariate_normal(mean2, std2, 100)

points = np.concatenate([points1, points2], axis=0)

mean1_hat = [0, 0]
std1_hat = [[1, 0], [0, 1]]
mean2_hat = [1,0]
std2_hat = [[1, 0], [0, 1]]

# E-step
# Create a prob distributions
for i in range(5):
    dist1 = scipy.stats.multivariate_normal(mean1_hat, std1_hat)
    dist2 = scipy.stats.multivariate_normal(mean2_hat, std2_hat)
    probs1 = dist1.pdf(points)
    probs2 = dist2.pdf(points)
    labels = probs1 > probs2
    mean1_hat = np.mean(points[labels], axis=0)
    mean2_hat = np.mean(points[~labels], axis=0)
    std1_hat = np.cov(points[labels].T)
    std2_hat = np.cov(points[~labels].T)

    points1_hat = points[labels]
    points2_hat = points[~labels]


    plt.scatter(points1_hat[:, 0], points1_hat[:, 1], c="r")
    plt.scatter(points2_hat[:, 0], points2_hat[:, 1], c="b")

    plt.scatter(mean1_hat[0], mean1_hat[1], c="r", marker="x")
    plt.scatter(mean2_hat[0], mean2_hat[1], c="b", marker="x")
    plt.show()
    print(mean1_hat, mean2_hat)