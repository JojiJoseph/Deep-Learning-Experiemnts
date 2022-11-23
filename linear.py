import numpy as np

class Linear:
    def __init__(self, in_units=1,out_units=1) -> None:
        self.W = np.random.normal(size=(out_units, in_units))
    def forward(self, x):
        assert len(x.shape) == 2
        return (self.W @ x[:,:,None]).squeeze()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    lin = Linear(in_units=1,out_units=2)
    x = np.linspace(0,1,100).reshape((-1,1))
    lin.W[0,0] = 2
    lin.W[1,0] = 5
    y = lin.forward(x)
    print(y.shape)
    plt.plot(x, y)
    plt.show()