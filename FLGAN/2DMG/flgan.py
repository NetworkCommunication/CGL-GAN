# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import copy
import os
import threading
import time
from queue import Queue
from random import Random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.stats import entropy
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import gmm
from model import Discriminator, Generator

Tensor = torch.cuda.FloatTensor if torch.cuda.device_count() else torch.FloatTensor

torch.cuda.set_device(0)
plt.ion()
lock = threading.Lock()

num_communication = 10000
workers = []
servers = []
cloud = None
cloud_epoch = 1

num_workers = 10
num_servers = 1
num_class = num_workers  # 需要大于等于 num_worker
num_sample = 1000  # 每个类别的样本数量
iid = 2
rd = Random()
rd.seed(20211212)
torch.manual_seed(20211212)
torch.cuda.manual_seed(20211212)
datasets = []
test_set = []
input_size = None
# service_size = [3, 3]  # 每个服务器的服务对象数, 有一个处于overlap位置
overlap = num_servers / num_workers  # 设备处于overlap区域的概率
batch_size = 100
frac_workers = 1  # 选择同步的工人的比例 （默认为全部）
epoch = 5  # 本地迭代次数
# overlap_worker = [[0], [0]]
ims = 0
b1 = 0.5
b2 = 0.999
num_plt = 100


def plot_2d():
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    df = pd.DataFrame()
    record = {}
    # sd = test_set[rd.sample([x for x in range(len(test_set))], num_sample)]
    sd = test_set[::test_set.shape[0] // num_sample]
    rx = sd[:, 0]
    ry = sd[:, 1]
    stepsize = 16
    count_r, _, _ = np.histogram2d(np.array(sd[:, 0]), np.array(sd[:, 1]), bins=stepsize, range=[[-1, 1], [-1, 1]])
    for item in tqdm(range(num_communication // num_plt)):
        plt.scatter(rx, ry, s=1, alpha=0.2, cmap="viridis")
        D = []
        for j in range(num_workers):
            X = workers[j].queen_gen_data.get()
            # D.append(X[rd.sample(range(X.shape[0]), num_sample // num_servers)])
            D.append(X[::X.shape[0] // (num_sample // num_workers)])
            gx = X[:, 0]
            gy = X[:, 1]
            plt.scatter(gx, gy, s=0.5, alpha=0.8, cmap="viridis")
        D = torch.cat(D, dim=0)

        count_g, _, _ = np.histogram2d(np.array(D[:, 0]), np.array(D[:, 1]), bins=stepsize, range=[[-1, 1], [-1, 1]])
        r_h_zero = []
        g_h_zero = []
        for i in range(len(count_r)):
            for j in range(len(count_r)):
                if count_r[i][j] != 0:
                    r_h_zero.append(count_r[i][j])
                    g_h_zero.append(count_g[i][j])
        r_h_zero = torch.tensor(r_h_zero)
        g_h_zero = torch.tensor(g_h_zero)
        kl_score = entropy(g_h_zero, r_h_zero)
        ds = g_h_zero.sum() / len(D)
        # cs = np.array(g_h_zero).nonzero()[0].size / r_h_zero.size
        record["Distribution Score"] = ds.item()
        # record["Mixture Guass Score"] = cs * ds
        record["KL Score"] = kl_score
        # plt.title("Round:{}".format(item + 1))
        plt.savefig("./logger/" + SimulationName + "/" + "%d.png" % (item + 1))
        plt.cla()
        df = df.append(record, ignore_index=True)
        df.to_excel("./logger/" + SimulationName + ".xlsx")
        with lock:
            print("ds", ds.item(), "kl", kl_score)

class Server(threading.Thread):

    def __init__(self, rank, dataset=None, lr_g=0.0002, lr_d=0.0002):
        super(Server, self).__init__(name="Server" + str(rank + 1))
        self.idx = rank
        self.generator = Queue(maxsize=20)
        self.discriminator = Queue(maxsize=20)
        self.cache = Queue(maxsize=5)
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = epoch
        self.queen_g = Queue(maxsize=20)
        self.queen_d = Queue(maxsize=20)
        self.rd = Random()
        self.rd.seed(rank + 100)
        self.client_list = []
        self.queen_gen_data = Queue(maxsize=50)

    def copy_parameters(self, net):
        parameters = {}
        for key, var in net.state_dict().items():
            parameters[key] = var.clone()
        return parameters

    def run(self):
        print(self.name, ":starting service for", [item + 1 for item in self.client_list])

        t = num_communication
        net_d = Discriminator().cuda()
        net_g = Generator(ims).cuda()
        p_g = self.copy_parameters(net_g)
        p_d = self.copy_parameters(net_d)
        while t > 0:

            for idx in self.client_list:
                workers[idx].queen_g.put(p_g)
                workers[idx].queen_d.put(p_d)
            p_d = {}
            p_g = {}
            for item in range(num_workers):
                paras = self.queen_d.get()
                for key in paras:
                    if key in p_d:
                        p_d[key] += paras[key] / len(self.client_list)
                    else:
                        p_d[key] = paras[key] / len(self.client_list)
                paras = self.queen_g.get()
                for key in paras:
                    if key in p_g:
                        p_g[key] += paras[key] / len(self.client_list)
                    else:
                        p_g[key] = paras[key] / len(self.client_list)
            t -= 1



class Worker(threading.Thread):

    def __init__(self, rank, dataset, lr_g=0.0002, lr_d=0.0002):
        super(Worker, self).__init__(name="Worker" + str(rank + 1))
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
        self.rd = Random()
        self.rd.seed(rank)
        self.server_list = []
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
        self.data = iter(self.dataloader)
        self.mse = torch.nn.L1Loss()
        self.A = None
        self.queen_gen_data = Queue(maxsize=50)
        self.data_len = None
        self.fixed_z = Variable(Tensor(np.random.normal(0, 1, (num_sample // num_servers, 100))))

    def copy_parameters(self, net):
        parameters = {}
        for key, var in net.state_dict().items():
            parameters[key] = var.clone()
        return parameters

    def run(self):
        print(self.name, ":starting waiting for service from", [item + 1 for item in self.server_list])

        t = num_communication
        net_g = Generator(ims).cuda()
        net_d = Discriminator().cuda()
        loss = nn.BCELoss().cuda()
        opti_g = optim.Adam(net_g.parameters(), lr=self.lr_g, betas=(b1, b2))
        opti_d = optim.Adam(net_d.parameters(), lr=self.lr_d, betas=(b1, b2))

        while t > 0:
            if t % num_plt == 0:
                self.plot_2d(net_g)
            p_g = self.queen_g.get()
            p_d = self.queen_d.get()
            net_d.load_state_dict(p_d, strict=False)
            net_g.load_state_dict(p_g, strict=False)
            self.train(net_d, net_g, loss, opti_g, opti_d)
            p_d = self.copy_parameters(net_d)
            p_g = self.copy_parameters(net_g)
            servers[0].queen_g.put(p_g)
            servers[0].queen_d.put(p_d)
            t -= 1

    def plot_2d(self, net):
        net.eval()
        with torch.no_grad():
            X = net(self.fixed_z).cpu()
        self.queen_gen_data.put(X)
        net.train()

    def train(self, net_d, net_g, loss, opti_g, opti_d):
        start = time.time()

        fake = Variable(Tensor(self.batch_size, 1).fill_(0), requires_grad=False)
        for i in range(epoch):
            try:
                imgs = next(self.data)
            except StopIteration:
                self.data = iter(self.dataloader)
                imgs = next(self.data)

            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1), requires_grad=False)
            z = Variable(Tensor(np.random.normal(0, 1, (self.batch_size, 100))))
            Xd = net_g(z)
            real_imgs = Variable(imgs.type(Tensor))

            opti_d.zero_grad()
            real_loss = loss(net_d(real_imgs), valid)
            fake_loss = loss(net_d(Xd), fake)
            D_loss = (real_loss + fake_loss)
            D_loss.backward()
            opti_d.step()

            valid = Variable(Tensor(self.batch_size, 1).fill_(1), requires_grad=False)
            opti_g.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (self.batch_size, 100))))
            Xg = net_g(z)
            g_loss = loss(net_d(Xg), valid)
            g_loss.backward()
            opti_g.step()

        end = time.time()


def del_tensor_ele(arr, index, l):
    arr1 = arr[0:index]
    arr2 = arr[index + l:]
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
    test_set = copy.deepcopy(data)
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
        se = rd.sample(range(1, num_workers * 2), k=num_workers - 1)
        se.append(0)
        se.append(num_workers * 2)
        se = sorted(se)
        sizes = [(se[i] - se[i - 1]) / (num_workers * 2) for i in range(1, len(se))]

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
    # sleep(14400)
    for k in range(2,3):
        workers.clear()
        servers.clear()
        datasets.clear()
        iid = k
        global SimulationName

        SimulationName = time.strftime("%Y-%m-%d %H-%M-%S",
                                       time.localtime()) + "-FLGAN-" + "iid_%d" % iid + "-data_class_%d" % num_class + "-epoch_%d" % epoch + "-cloud_%d" % cloud_epoch
        if not os.path.isdir('logger'):
            os.mkdir('logger')
        if not os.path.isdir('./logger/' + SimulationName):
            os.mkdir('./logger/' + SimulationName)
        allocate_dataset(gmm(num_class, num_sample), iid)

        for i in range(num_workers):
            workers.append(Worker(i, dataset=datasets[i]))
            plt.scatter(datasets[i][:, 0], datasets[i][:, 1], cmap="viridis")
            plt.title("device:%d" % i)
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            plt.savefig("./logger/" + SimulationName + "/Distribution_" + str(i) + ".png")
            plt.cla()


        worker = [id for id in range(num_workers)]

        for i in range(num_servers):
            servers.append(Server(i))
            servers[i].client_list = worker

        for i in range(num_workers):
            workers[i].server_list.append(0)

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
