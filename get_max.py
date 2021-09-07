"""
Neural network that outputs max of the given numbers.

It satisfies the following constraints

 - It should contain trainable paramters
 - Minimum number of parameters (1)
 - No search function to find max of the given numbers

This neural network implements following function

y = sum( softmax( p*x ) * x )

x is the given vector. p is a trainable parameter.

As p tends to infinity y tends to be the max of x.
"""
import torch
import torch.nn as nn

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.p = nn.Parameter(data=torch.zeros(1))
  def forward(self, x):
    scores = torch.softmax(self.p*x, axis=-1)
    y = torch.sum(scores*x, -1)
    return y

model = Model().to(device)

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.p = nn.Parameter(data=torch.zeros(1))
  def forward(self, x):
    scores = torch.softmax(self.p*x, axis=-1)
    y = torch.sum(scores*x, -1)
    return y

optim = torch.optim.Adam(model.parameters(), lr=10)

for epoch in tqdm(range(10_000)):
    x = torch.rand(1000, 10).to(device)*5
    y = model(x)
    loss = torch.mean((y-torch.max(x,dim=-1)[0])**2)
    optim.zero_grad()
    loss.backward()
    optim.step()

test_cases = torch.randint(-10,10,size=(100,10))

for t in test_cases:
    x = t[None, :].to(device)
    y = model(x)
    print(y.item(), torch.max(x).item())