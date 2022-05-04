import numpy as np
import torch

np.random.seed(20211212)
class gmm():
    def __init__(self, n_class=5, x=10000):
        self.data = None
        self.targets = None
        self.n_c = n_class
        self.x = x
        self.init()

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)

    # def sample(self, mu, var, num):
    #     out = [torch.normal(mu, var) for _ in range(num*self.x)]
    #     return torch.stack(out)

    def init(self):
        # 每个类别随机大小
        n_mixture = self.n_c
        radius = 1
        std = 0.01
        thetas = np.linspace(0, 2 * (1 - 1 / n_mixture) * np.pi, n_mixture)
        xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
        data_size = self.x * n_mixture
        data = torch.zeros(data_size, 2)
        labels = torch.zeros(data_size)
        for i in range(data_size):
            coin = np.random.randint(0, n_mixture)
            data[i, :] = torch.normal(mean=torch.Tensor([xs[coin], ys[coin]]), std=std * torch.ones(1, 2))
            labels[i] = coin
        self.targets, indexes = torch.sort(labels)
        self.data = data[indexes]




