from torchvision import transforms
import copy
import threading
import time
from queue import Queue
from random import Random
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets as torch_ds
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
from model.mnist_model import Discriminator, Generator
from torch.autograd import Variable
import pickle as pkl
from fedlab.utils.dataset.partition import MNISTPartitioner, CIFAR10Partitioner
from fedlab.utils.functional import partition_report
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
import numpy as np
Tensor = torch.cuda.FloatTensor if torch.cuda.device_count() else torch.FloatTensor
rd = Random()
seed = 20211212
rd.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

plt.ion()
lock = threading.Lock()
SimulationName = None
num_communication = 20000
workers = []
servers = []
cloud = None
cloud_epoch = 0
segema = 0
# server_epoch = 100
num_workers = 10
num_servers = 1
num_class = 10  # 需要大于等于 num_worker
num_sample = 1000  # 每个类别的样本数量
iid = 0
datasets = []
test_set = []
batch_size = 100
frac_workers = 1  # 选择同步的工人的比例 （默认为全部）
epoch = 1  # 同步前的本地迭代次数

b1 = 0.5
b2 = 0.999
img_size = 28
done = False
# dataset = None
dataset = torch_ds.MNIST(root="./data/", train=True, download=True)
ims = (1, img_size, img_size)

weight_avg = {"mean", "weight", "game-mean", "game-"}


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_2d():
    for item in tqdm(range(num_communication // 500)):
        D = []
        for j in range(num_servers):
            X = servers[j].queen_gen_data.get()
            D.append(X[::X.shape[0] // (num_sample // num_servers)])
            # D.append(X)
        D = torch.cat(D)
        save_image(D[::D.shape[0] // 100], "./logger/" + SimulationName + "/%d.png" % item, nrow=10, normalize=True)


class Cloud(threading.Thread):

    def __init__(self):
        super(Cloud, self).__init__(name="Cloud")
        self.cache = Queue()
        self.server_list = [x for x in range(num_servers)]
        self.A = []

    def getMaxCon(self, a, b):
        while b:
            a, b = b, a % b
        return a

    def run(self) -> None:
        weight = torch.zeros(num_servers)
        for id in range(num_servers):
            while servers[self.server_list[id]].data_len is None:
                pass
            weight[id] = servers[self.server_list[id]].data_len
        weight /= weight.sum()
        if cloud_epoch == 0:
            return
        for i in range(num_communication // cloud_epoch):

            p = [0 for _ in range(num_servers)]
            for item in range(num_servers):
                idx, paras = self.cache.get()
                p[idx] = paras
            para_sum = Aggregators.fedavg_aggregate(p, weights=weight)

            for idx in self.server_list:
                servers[idx].cache.put(copy.deepcopy(para_sum))


class Server(threading.Thread):

    def __init__(self, rank, dt=None, lr_g=0.0002, lr_d=0.0002):
        super(Server, self).__init__(name="Server" + str(rank + 1))
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
        self.rd.seed(rank + 100)
        self.client_list = []
        self.queen_gen_data = Queue(maxsize=50)
        self.beta = None
        self.Lambda = torch.tensor(0., requires_grad=True)
        self.opti_L = optim.SGD([self.Lambda], lr=0.1)
        self.data_len = None
        self.fixed_z = torch.randn(num_sample // num_servers, 100, device="cuda:0")

    def run(self):
        print(self.name, ":starting service for", [item + 1 for item in self.client_list])
        N = len(self.client_list)
        self.choose_flag = [0 for _ in range(N)]
        self.beta = torch.zeros(N)
        for c in range(N):
            self.beta[c] = len(workers[self.client_list[c]].dataset)
        self.data_len = self.beta.sum()
        self.beta /= self.data_len

        t = num_communication
        net_g = Generator(ims).cuda()
        # net_g.apply(weights_init)
        opti_g = optim.Adam(net_g.parameters(), lr=self.lr_g, betas=(b1, b2))

        lambda_list = []
        gen_data = []
        betas = []
        gammas = []
        while t > 0:

            if t % 500 == 0:
                gen_data.append(self.plot_2d(net_g))

            if cloud_epoch != 0 and t % cloud_epoch == 0:
                self_p = SerializationTool.serialize_model(net_g)
                parameters = copy.deepcopy(self_p)
                cloud.cache.put((self.idx, parameters))
                recv_p = self.cache.get()
                recv_p = segema * self_p + (1 - segema) * recv_p
                SerializationTool.deserialize_model(net_g, recv_p)

            fbeta, fgamma, fmax = self.train(net_g, opti_g, N)
            print(self.idx, num_communication - t, fmax, "lambda:", self.Lambda.item())

            if t % 500 == 0:
                betas.append(fbeta)
                gammas.append(fgamma)
                lambda_list.append(self.Lambda.item())

            if t % 5000 == 0:
                torch.save(net_g.state_dict(), "./logger/" + SimulationName + "/{}.pt".format(self.name, t))
                with open("./logger/" + SimulationName + "/config{}.pkl".format(self.name, t), 'wb') as f:  # 将数据写入pkl文件
                    pkl.dump((self.client_list, self.beta, lambda_list, [], gen_data, betas, gammas), f)
                    f.flush()
                    f.close()
                lambda_list.clear()
                gen_data.clear()
                betas.clear()
                gammas.clear()

            t -= 1

        torch.save(net_g.state_dict(), "./logger/" + SimulationName + "/{}.pt".format(self.name))
        with open("./logger/" + SimulationName + "/config{}.pkl".format(self.name), 'wb') as f:  # 将数据写入pkl文件
            pkl.dump((self.client_list, self.beta, lambda_list, [], gen_data, betas, gammas), f)
        exit()

    def plot_2d(self, net):
        net.eval()
        with torch.no_grad():
            X = net(self.fixed_z).cpu()
        self.queen_gen_data.put(X)
        net.train()
        return X.clone()

    def train(self, net_g, opti, N):
        start = time.time()

        # 生成随机噪音，使用正态分布
        with torch.no_grad():
            z = torch.randn(self.batch_size, 100, device="cuda:0", requires_grad=False)
            Xd = net_g(z)

        z = torch.randn(self.batch_size, 100, device="cuda:0", requires_grad=True)
        Xg = net_g(z)

        # send
        for client in self.client_list:
            workers[client].queen_d.put((self.idx, Xd.detach()))
            workers[client].queen_g.put((self.idx, Xg.clone()))

        opti.zero_grad()
        loss = torch.zeros(N)

        for i in range(N):
            (idx, g_loss) = self.queen_g.get()
            loss[self.client_list.index(idx)] = g_loss.clone()

        # 计算权重和并更新神经网络
        self.opti_L.zero_grad()

        # average when num_server = 1 this equal MDGAN
        # F_max = loss.sum()

        # mean weight
        gamma = F.softmax(self.Lambda.detach() * loss.detach(), dim=0)
        s = F.softmax(self.beta * gamma, dim=0)
        F_max = (s * loss).sum() - 0.001 * self.Lambda

        # exp weight
        # alpha = F.softmax(self.Lambda.detach() * loss.detach(), dim=0)
        # alpha = F.softmax(alpha * self.beta, dim = 0)
        # F_max = (alpha * loss).sum() - 0.001 * self.Lambda
        #

        # beta
        # F_max= (self.beta * loss).sum()

        # gamma
        # F_max = (gamma * loss).sum() - 0.001 * self.Lambda

        F_max.backward()
        self.opti_L.step()
        opti.step()
        end = time.time()
        return 0, 0, F_max.cpu().item()


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
        print(self.name, ":starting waiting for service from", [item + 1 for item in self.server_list])

        t = num_communication
        net_d = Discriminator(ims).cuda()
        # net_d.apply(weights_init)
        loss = nn.CrossEntropyLoss().cuda()
        opti_d = optim.Adam(net_d.parameters(), lr=self.lr_d, betas=(b1, b2))

        self.train(t, net_d, loss, opti_d)

    def train(self, t, net_d, loss, opti_d):
        while True:
            start = time.time()
            if servers[0].isAlive() is not True:
                return

            idx, X = self.queen_d.get()

            for i in range(epoch):
                try:
                    imgs, _ = next(self.data)
                except StopIteration:
                    self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
                    self.data = iter(self.dataloader)
                    imgs, _ = next(self.data)
                valid = Variable(torch.cuda.LongTensor(imgs.shape[0]).fill_(1.), requires_grad=False)
                real_imgs = Variable(imgs.type(Tensor))

                opti_d.zero_grad()
                real_loss = loss(net_d(real_imgs), valid)
                fake = Variable(torch.cuda.LongTensor(self.batch_size).fill_(0.),
                                requires_grad=False)
                fake_loss = loss(net_d(X), fake)
                D_loss = (real_loss + fake_loss) * 0.5
                D_loss.backward()
                opti_d.step()

            valid = Variable(torch.cuda.LongTensor(self.batch_size).fill_(1.), requires_grad=False)
            (idx, Xg) = self.queen_g.get()
            validaty = net_d(Xg)
            G_loss = loss(validaty, valid)
            servers[idx].queen_g.put((self.idx, G_loss))

            end = time.time()


def del_tensor_ele(arr, index, l):
    arr1 = arr[0:index]
    arr2 = arr[index + l:]
    return torch.cat((arr1, arr2), dim=0)


def allocate_dataset(data, iid):
    labels = list(data.targets)
    data_len = len(data.data)
    indexes = [x for x in range(0, data_len)]
    #
    global test_set
    # # test_set = copy.deepcopy(data)
    test_set = data.data[rd.sample(range(data_len), num_sample)]
    # iid 简单均分就完事了
    if iid == 0:
        sizes = [1.0 / num_workers for _ in range(num_workers)]
        rd.shuffle(indexes)  # 打乱下标
        for frac in sizes:
            part_len = int(frac * data_len)
            dd = copy.deepcopy(data)
            dd.data = data.data[indexes[0:part_len]]
            dd.targets = data.targets[indexes[0:part_len]]
            datasets.append(dd)
            indexes = indexes[part_len:]
    else:
        order = np.argsort(labels)
        data.data = data.data[order]
        data.targets = data.targets[order]
        labels = data.targets
        # 生成和为 1 的随机比例
        se = rd.sample(range(1, num_workers ** 2), k=num_workers - 1)
        se.append(0)
        se.append(num_workers ** 2)
        se = sorted(se)
        sizes = [(se[i] - se[i - 1]) / (num_workers ** 2) for i in range(1, len(se))]
        # [0.16, 0.24, 0.18, 0.03, 0.02, 0.01, 0.09, 0.22, 0.03, 0.02]
        if iid == 1:
            labels = labels.tolist()
            for i in range(num_workers):
                index_s = (i - 1 + num_class) % num_class
                index_e = (i + 2) % num_class
                s = labels.index(index_s)
                e = labels.index(index_e)
                l = int(sizes[i] * data_len)
                dd = copy.deepcopy(data)
                if s < e:
                    if l > (e - s):
                        l = e - s
                    choose_list = rd.sample(range(s, e), l)
                    dd.data = data.data[choose_list]
                    dd.targets = data.targets[choose_list]
                    datasets.append(dd)
                else:
                    if l > (e + data_len - s):
                        l = e + data_len - s
                    choose_list = rd.sample(list(range(0, e)) + list(range(s, data_len)), l)
                    dd.data = data.data[choose_list]
                    dd.targets = data.targets[choose_list]
                    datasets.append(dd)
        else:
            l = 1
            s = 0
            for i in range(num_workers):
                while l < data_len and labels[l] == labels[l - 1]:
                    l += 1
                dd = copy.deepcopy(data)
                choose_list = rd.sample(range(s, l), min(int(sizes[i] * data_len), l-s))
                dd.data = data.data[choose_list]
                dd.targets = data.targets[choose_list]
                datasets.append(dd)
                s = l % data_len
                l = s + 1

# def allocate_dataset(trainset, iid):
#     global test_set
#     test_set = trainset.data[rd.sample(range(len(trainset.data)), num_sample)]
#     # iid 简单均分就完事了
#     if iid == 0:
#         return CIFAR10Partitioner(trainset.targets,
#                                   num_workers,
#                                   partition="iid",
#                                   dir_alpha=0.3,
#                                   seed=seed)
#
#
#
#     elif iid == 1:
#         return CIFAR10Partitioner(trainset.targets,
#                                   num_workers,
#                                   balance=None,
#                                   partition="dirichlet",
#                                   dir_alpha=0.1,
#                                   seed=seed)
#
#     elif iid == 2:
#         return CIFAR10Partitioner(trainset.targets,
#                                   num_workers,
#                                   balance=False,
#                                   partition="shards",
#                                   num_shards=200,
#                                   dir_alpha=0.1,
#                                   seed=seed)
#     else:
#         raise ValueError('the value of iid only support 0 iid, 1 basic non-iid, 2 fully non-iid')


if __name__ == "__main__":
    # sleep(60*60*4)
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
                                           time.localtime()) + "-CAPsum-" + dataset_name + "-iid_%d" % iid + "-epoch_%d" % epoch \
                             + "-H-%d" % cloud_epoch + "_share-%.1f" % segema + "_batchsize%d" % batch_size
            if not os.path.isdir('./logger'):
                os.mkdir('./logger')
            if not os.path.isdir('./logger/' + SimulationName):
                os.mkdir('./logger/' + SimulationName)

            # datasets_parts = allocate_dataset(dataset, iid)
            allocate_dataset(dataset, iid)
            cloud = Cloud()

            # csv_file = './logger/' + SimulationName + "/mnist_iid_{}.csv".format(iid)
            # partition_report(dataset.targets, datasets_parts.client_dict, class_num=10, verbose=False,
            #                  file=csv_file)

            # for i in range(num_workers):
            #     dd = copy.deepcopy(dataset)
            #     dd.data = dataset.data[datasets_parts.client_dict[i]]
            #     dd.targets = dataset.targets[datasets_parts.client_dict[i]]
            #     workers.append(Worker(i, dataset=dd))

            for i in range(num_workers):
                workers.append(Worker(i, dataset=datasets[i]))

            for i in range(num_servers):
                servers.append(Server(i))

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

            cloud.start()
            painter = threading.Thread(plot_2d())
            painter.start()

            for i in range(num_servers):
                servers[i].join()

            for i in range(num_workers):
                workers[i].join()
            cloud.join()
            painter.join()

            print("Simulation", SimulationName, " is over!!!")
