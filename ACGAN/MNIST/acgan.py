import PIL.Image as Image
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from torchvision import datasets, transforms
import copy
import threading
import time
from queue import Queue
from time import sleep
from random import Random
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets as torch_ds
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
from mnist_model import Discriminator, Generator
from torch.autograd import Variable
import pickle as pkl

Tensor = torch.cuda.FloatTensor if torch.cuda.device_count() else torch.FloatTensor


plt.ion()
lock = threading.Lock()
SimulationName = None
num_communication = 20000
workers = []
servers = []
cloud = None
cloud_epoch = 1
# server_epoch = 100
E = 5  # 表示不分享， 否则表示分享的间隔
num_workers = 10
num_servers = 5
num_class = 10        # 需要大于等于 num_worker
num_sample = 1000    # 每个类别的样本数量
iid = 0
rd = Random()
rd.seed(20211212)
torch.manual_seed(20211212)
torch.cuda.manual_seed(20211212)
datasets = []
test_set = []
input_size = None
batch_size = 100
frac_workers = 1     # 选择同步的工人的比例 （默认为全部）
epoch = 1            # 同步前的本地迭代次数

ims = 0
b1 = 0.5
b2 = 0.999
img_size = 28


dataset = None


def plot_2d():

    for item in tqdm(range(num_communication//500)):
        D = []
        for j in range(num_servers):
            X = servers[j].queen_gen_data.get()
            D.append(X)

        D = torch.cat(D)
        save_image(D[::D.shape[0] // 100], "./logger/" + SimulationName + "/%d.png" % item, nrow=10, normalize=True)




class Server(threading.Thread):

    def __init__(self, rank, dt=None, lr_g=0.0002, lr_d=0.0002):
        super(Server, self).__init__(name="Server"+str(rank+1))
        self.idx = rank
        self.generator = Queue(maxsize=20)
        self.discriminator = Queue(maxsize=20)
        self.cache = Queue(maxsize=5)
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.dataset = dt
        self.batch_size = batch_size
        self.epoch = epoch
        self.queen_g = Queue(maxsize=20)
        self.queen_d = Queue(maxsize=20)
        self.rd = Random()
        self.rd.seed(rank+100)
        self.client_list = []
        self.queen_gen_data = Queue(maxsize=50)
        self.beta = None
        self.Lambda = torch.tensor(0., requires_grad=False)
        self.opti_L = optim.SGD([self.Lambda], lr=0.1)
        self.data_len = None
        self.fixed_z = torch.randn(num_sample//num_servers, 100, device="cuda:0")

    def copy_parameters(self, net):
        parameters = {}
        for key, var in net.state_dict().items():
            if len(var.size()) != 0:
                parameters[key] = var.clone()
        return parameters

    def receive_parameter(self):
        p = self.discriminator.get()
        for _ in range(len(self.client_list) - 1):
            d = self.discriminator.get()
            for key in p:
                p[key] += d[key]
        for key in p:
            p[key] /= len(self.client_list)
        return p

    def run(self):
        print(self.name, ":starting service for", [item+1 for item in self.client_list])
        N = len(self.client_list)

        t = num_communication
        net_g = Generator(ims).cuda()
        opti_g = optim.Adam(net_g.parameters(), lr=self.lr_g, betas=(b1, b2))
        gen_data = []
        losses = []
        while t > 0:

            if t % 500 == 0:
                gen_data.append(self.plot_2d(net_g))

            losses.append(self.train(net_g, opti_g, N))
            t -= 1
        torch.save(net_g.state_dict(), "./logger/"+SimulationName + "/{}.pt".format(self.name))
        with open("./logger/" + SimulationName + "/config{}.pkl".format(self.name), 'wb') as f:  # 将数据写入pkl文件
             pkl.dump((self.client_list, [], gen_data, losses), f)
        print(self.name, "train over")

    def plot_2d(self, net):
        net.eval()
        with torch.no_grad():
            X = net(self.fixed_z).cpu()
        self.queen_gen_data.put(X)
        net.train()
        return X

    def train(self, net_g, opti, N):
        start = time.time()

        # 生成随机噪音，使用正态分布
        with torch.no_grad():
            z = torch.randn(self.batch_size, 100, device="cuda:0", requires_grad=True)
            Xd = net_g(z)

        z = torch.randn(self.batch_size, 100, device="cuda:0", requires_grad=True)
        Xg = net_g(z)

        # send
        for client in self.client_list:
            workers[client].queen_d.put((self.idx, Xd.clone()))
            workers[client].queen_g.put((self.idx, Xg.clone()))

        opti.zero_grad()
        loss = torch.zeros(N)

        for i in range(N):
            (idx, g_loss) = self.queen_g.get()
            # 更新个性层
            loss[self.client_list.index(idx)] = g_loss.clone()

        losses = loss.mean()
        losses.backward()
        opti.step()
        with lock:
            print(self.name, losses.item())
        end = time.time()
        return losses.item()



class Worker(threading.Thread):

    def __init__(self, rank, dataset, lr_g=0.0002, lr_d=0.0002):
        super(Worker, self).__init__(name="Worker" + str(rank+1))
        self.idx = rank
        self.generator = Queue(maxsize=20)
        self.discriminator = Queue(maxsize=20)
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = epoch
        self.queen_g = Queue(maxsize=20)
        self.queen_d = Queue(maxsize=20)
        self.para_d = Queue(maxsize=20)
        self.rd = Random()
        self.rd.seed(rank)
        self.server_list = []
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
        # self.test = self.dataset[self.rd.sample(range(len(self.dataset)), num_sample)]
        self.data = iter(self.dataloader)
        self.mse = torch.nn.L1Loss()
        self.A = None

    def copy_parameters(self, net):
        parameters = {}
        for key, var in net.state_dict().items():
            if len(var.size()) != 0:
                parameters[key] = var.clone()
        return parameters

    def receive_parameter(self):
        p = self.discriminator.get()
        for _ in range(len(self.server_list) - 1):
            d = self.discriminator.get()
            for key in p:
                p[key] += d[key]
        for key in p:
            p[key] /= len(self.server_list)
        return p

    def run(self):
        print(self.name, ":starting waiting for service from", [item+1 for item in self.server_list], "with dataset", len(self.dataset))
        N = len(self.server_list)

        t = num_communication
        net_d = Discriminator(ims).cuda()
        loss = nn.CrossEntropyLoss().cuda()
        opti_d = optim.Adam(net_d.parameters(), lr=self.lr_d, betas=(b1, b2))
        s = {}
        w = {}
        p = self.copy_parameters(net_d)
        for key in p:
            w[key] = 0
            s[key] = 0

        while t > 0:
            # if (num_communication - t) % E == 0 :
            #     p = self.copy_parameters(net_d)
            #     g = {}
            #     for key in p:
            #         g[key] = p[key] - w[key]
            #
            #     for key in w:
            #         w[key] += g[key]
            #
            #     idxs = []
            #     for s in self.server_list:
            #         idxs += servers[s].client_list
            #     for idx in idxs:
            #         if idx != self.idx:
            #             workers[idx].para_d.put(g)
            #     k = len(idxs)
            #     for idx in idxs:
            #         gk = self.para_d.get()
            #         for key in gk:
            #             s[key] += gk[key] / k
            #     for key in p:
            #         p[key] += (s[key] - w[key])
            #
            #     net_d.load_state_dict(p, strict=False)

            self.train(net_d, loss, opti_d, N)
            t -= 1

    def train(self, net_d, loss, opti_d, N):
        start = time.time()
        Xs = []
        for _ in self.server_list:
            Xs.append(self.queen_d.get())
        for idx, X in Xs:
            for i in range(epoch):
                try:
                    imgs = next(self.data)
                except StopIteration:
                    self.data = iter(self.dataloader)
                    imgs = next(self.data)
                # valid = Variable(Tensor(imgs.shape[0], 1).fill_(1), requires_grad=False)
                valid = Variable(torch.cuda.LongTensor(imgs.shape[0]).fill_(1), requires_grad=False)
                real_imgs = Variable(imgs.type(Tensor))
                opti_d.zero_grad()
                real_loss = loss(net_d(real_imgs), valid)

                # fake = Variable(Tensor(self.batch_size, 1).fill_(0), requires_grad=False)
                fake = Variable(torch.cuda.LongTensor(self.batch_size).fill_(0),
                                                requires_grad=False)
                fake_loss = loss(net_d(X), fake)
                D_loss = (real_loss + fake_loss)
                D_loss.backward()
                opti_d.step()
        # valid = Variable(Tensor(self.batch_size, 1).fill_(1), requires_grad=False)
        valid = Variable(torch.cuda.LongTensor(self.batch_size).fill_(1), requires_grad=False)
        for _ in self.server_list:
            (id, Xg) = self.queen_g.get()
            validaty = net_d(Xg)
            G_loss = loss(validaty, valid)
            servers[id].queen_g.put((self.idx, G_loss))

        end = time.time()

def del_tensor_ele(arr, index, l):
    arr1 = arr[0:index]
    arr2 = arr[index+l:]
    return torch.cat((arr1, arr2), dim=0)

def allocate_dataset(data, iid):
    labels = []
    global ims
    for i, d in enumerate(torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)):
        ims = d[0][0].shape
        data = d[0]
        labels = d[1]
    data_len = len(data)
    indexes = [x for x in range(0, data_len)]

    global test_set
    # test_set = copy.deepcopy(data)
    test_set = data[rd.sample(range(data_len), num_sample)]
    # iid 简单均分就完事了
    if iid == 0:
        sizes = [1.0 / num_workers for _ in range(num_workers)]
        rd.shuffle(indexes)  # 打乱下标
        for frac in sizes:
            part_len = int(frac * data_len)
            datasets.append(data[indexes[0:part_len]])
            indexes = indexes[part_len:]
    else:
        order = np.argsort(labels)
        data = data[order]
        labels = labels[order]
        # 生成和为 1 的随机比例
        se = rd.sample(range(1, num_workers ** 2), k=num_workers - 1)
        se.append(0)
        se.append(num_workers ** 2)
        se = sorted(se)
        sizes = [(se[i] - se[i - 1]) / (num_workers ** 2) for i in range(1, len(se))]

        if iid == 1:
            labels = labels.tolist()
            for i in range(num_workers):
                index_s = (i - 1 + num_class) % num_class
                index_e = (i + 2) % num_class
                s = labels.index(index_s)
                e = labels.index(index_e)
                l = int(sizes[i] * data_len)
                if s < e:
                    if l > (e - s):
                        l = e - s
                    datasets.append(data[rd.sample(range(s, e), l)])
                else:
                    if l > (e + data_len - s):
                        l = e + data_len - s
                    datasets.append(data[rd.sample(list(range(0, e)) + list(range(s, data_len)), l)])
        else:
            l = 1
            for i in range(num_workers):
                while labels[l] == labels[l - 1] and l < len(data) - 1:
                    l += 1
                datasets.append(data[:l])
                data = del_tensor_ele(data, 0, l)
                labels = del_tensor_ele(labels, 0, l)
                l = 1

if __name__ == "__main__":
    # sleep(60*60*2)
    for l in range(1):
        dataset_name = ""
        if l == 0:
            dataset_name = "MNIST"
            dataset = torch_ds.MNIST(root='./data/mnist', train=True, download=True,
                         transform=transforms.Compose(
                                              [transforms.Resize(img_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])
                                              ]))
        if l == 1:
            dataset_name = "Fashion-MNIST"
            dataset = torch_ds.FashionMNIST(root='./data', train=True, download=True,
                                     transform=transforms.Compose(
                                         [transforms.Resize(img_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5], [0.5])
                                          ]))
        for k in range(1, 2):
            workers.clear()
            servers.clear()
            datasets.clear()
            iid = k

            SimulationName = time.strftime("%Y-%m-%d %H-%M-%S",
                                           time.localtime()) + "-ACGAN-"+dataset_name + "-iid_%d" % iid + \
                             "-epoch_%d" % epoch + "_batch_size-%d"%batch_size
            if not os.path.isdir('./logger'):
                os.mkdir('./logger')
            if not os.path.isdir('./logger/' + SimulationName):
                os.mkdir('./logger/' + SimulationName)


            allocate_dataset(dataset, iid)

            for i in range(num_workers):
                workers.append(Worker(i, dataset=datasets[i]))
                save_image(datasets[i][rd.sample(range(len(datasets[i])), 100)],
                           "./logger/" + SimulationName + "/device_%d.png" % i, nrow=10,
                           normalize=True)

            for i in range(num_servers):
                servers.append(Server(i))

            # 获取每个设备处于overlap的概率
            worker = [id for id in range(num_workers)]
            for i in range(num_servers):
                alnwokers = worker[:num_workers // num_servers]
                worker = worker[num_workers // num_servers:]
                for j in alnwokers:
                    servers[i].client_list.append(j)
                    workers[j].server_list.append(i)

            print("Simulation", SimulationName, " is started!!!")
            # 启动所有程序
            for i in range(num_servers):
                servers[i].start()

            for i in range(num_workers):
                workers[i].start()

            painter = threading.Thread(plot_2d())
            painter.start()

            for i in range(num_servers):
                servers[i].join()

            for i in range(num_workers):
                workers[i].join()

            painter.join()

            print("Simulation", SimulationName, " is over!!!")
