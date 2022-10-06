"""Program to check if the number of goals scored in a world cup match can be approximated using poisson distribution"""
from math import factorial
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("WorldCupMatches.csv") # Download from https://www.kaggle.com/datasets/abecklas/fifa-world-cup
df = df[:852] # The data is empty after this

df = df[['Home Team Goals','Away Team Goals']]
df['total'] = df['Home Team Goals'] + df['Away Team Goals']

print(f" Max: {df['total'].max()}, Min: {df['total'].min()}, Average: {df['total'].mean()}, Median: {df['total'].median()}, Variance: {df['total'].var()}")

lda = df['total'].mean()


max_goals = int(df['total'].max())

real_prob = [0] * (max_goals+1)
calc_prob = [0] * (max_goals+1)

for goals in range(max_goals+1):
    real_prob[goals] = (df['total'] == goals).mean()
    calc_prob[goals] = np.exp(-lda)*lda**goals/factorial(goals)

print(sum(real_prob), sum(calc_prob))

# Plot the probability distribution
plt.bar(range(max_goals+1),real_prob, alpha=0.5, label="Real prob")
plt.bar(range(max_goals+1),calc_prob, alpha=0.5, label="Calculated prob")
plt.legend()
plt.show()