import PIL.Image as Image
import fedlab.core.client
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from torchvision import datasets, transforms
import copy
import threading
import time
from queue import Queue
from time import sleep
from random import Random, random
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
# from mnist_model import Discriminator, Generator
from model.mnist_model import Discriminator, Generator
from torch.autograd import Variable
import pickle as pkl
from fedlab.utils.dataset.partition import MNISTPartitioner
from fedlab.utils.functional import partition_report, save_dict
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from scipy import stats

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
segema = 0
# server_epoch = 100
num_workers = 10
num_servers = 1
num_class = 10
num_sample = 1000  # 每个类别的样本数量
iid = 0
datasets = []
test_set = []
input_size = None
# service_size = [3, 3]  # 每个服务器的服务对象数, 有一个处于overlap位置
batch_size = 100
frac_workers = 0.2  # 选择同步的工人的比例 （默认为全部）
epoch = 1  # 同步前的本地迭代次数

b1 = 0.5
b2 = 0.999
img_size = 28
# ims = (1, 28, 28)
ims = (1, img_size, img_size)


def plot_2d():
    item = 0
    while True:
        D = []
        for j in range(num_servers):
            X = servers[j].queen_gen_data.get()
            D.append(X[::X.shape[0] // (num_sample // num_servers)])
            # D.append(X)
        D = torch.cat(D)
        if servers[0].isAlive() is not True:
            break
        item += 1
        save_image(D[::D.shape[0] // 100], "./logger/" + SimulationName + "/%d.png" % item, nrow=10, normalize=True)


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


class Server(threading.Thread):

    def __init__(self, rank, groups, dt=None, lr_g=0.0002, lr_d=0.0002):
        super(Server, self).__init__(name="Server" + str(rank + 1))
        self.idx = rank
        self.generator = Queue(maxsize=200)
        self.discriminator = Queue(maxsize=200)
        self.cache = Queue(maxsize=200)
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.dataset = dt
        self.batch_size = batch_size
        self.epoch = epoch
        self.queen_g = Queue(maxsize=200)
        self.queen_d = Queue(maxsize=200)
        self.rd = Random()
        self.rd.seed(rank + 100)
        self.client_list = []
        self.groups = groups
        self.queen_gen_data = Queue(maxsize=200)
        self.weight = None
        self.Lambda = torch.tensor(0., requires_grad=False)
        self.opti_L = optim.SGD([self.Lambda], lr=0.1)
        self.data_len = None
        self.fixed_z = torch.randn(num_sample // num_servers, 100, device="cuda:0")
        # self.fixed_z = torch.randn(num_sample // num_servers, 100, 1, 1).cuda()
        self.t = 0


    def run(self):
        print(self.name, ":starting service for", [item + 1 for item in self.client_list])
        # init_models
        net_d = Discriminator(ims).cuda()
        # net_d.apply(weights_init)

        net_g = Generator(ims).cuda()
        # net_g.apply(weights_init)
        # get the init_parameter
        p_g = SerializationTool.serialize_model(net_g)
        p_d = SerializationTool.serialize_model(net_d)
        gen_data = []
        self.t = 0
        # get each round clients index 0 ~ num_workers-1
        for group in self.groups:
            N = len(group)
            weight = torch.zeros(N)

            for idx in range(N):
                weight[idx] = workers[group[idx]].sk
            # fegan
            weight = torch.exp(weight)
            weight /= weight.sum()
            # flgan

            for idx in group:
                workers[idx].queen_g.put(copy.deepcopy(p_g))
                workers[idx].queen_d.put(copy.deepcopy(p_d))

            list_d = [0 for _ in range(N)]
            list_g = [0 for _ in range(N)]

            for item in range(N):
                idx, paras = self.queen_d.get()
                list_d[group.index(idx)] = paras

                idx, paras = self.queen_g.get()
                list_g[group.index(idx)] = paras

            p_g = Aggregators.fedavg_aggregate(list_g, weights=weight)
            p_d = Aggregators.fedavg_aggregate(list_d, weights=weight)

            print(self.t, group, weight)

            if self.t % 100 == 0:
                SerializationTool.deserialize_model(net_g, p_g)
                gen_data.append(self.plot_2d(net_g))

            # checkpoint
            if self.t % 5000 == 0:
                torch.save(net_g.state_dict(), "./logger/" + SimulationName + "/{}_{}.pt".format(self.name, self.t))
                with open("./logger/" + SimulationName + "/config{}.pkl".format(self.name), 'wb') as f:  # 将数据写入pkl文件
                    pkl.dump(gen_data, f)
                gen_data.clear()

            self.t += 1

        torch.save(net_g.state_dict(), "./logger/" + SimulationName + "/{}_{}.pt".format(self.name, self.t))
        exit()

    def plot_2d(self, net):
        net.eval()
        with torch.no_grad():
            X = net(self.fixed_z).cpu()
        self.queen_gen_data.put(X)
        net.train()
        return X


class Worker(threading.Thread):

    def __init__(self, rank, dataset, sk, lr_g=0.0002, lr_d=0.0002):
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
        self.sk = sk

    def run(self):
        # print(self.name, ":starting waiting for service from", [item+1 for item in self.server_list])

        net_g = Generator(ims).cuda()
        net_d = Discriminator(ims).cuda()

        loss = nn.BCELoss().cuda()
        opti_g = optim.Adam(net_g.parameters(), lr=self.lr_g, betas=(b1, b2))
        opti_d = optim.Adam(net_d.parameters(), lr=self.lr_d, betas=(b1, b2))
        gen_data = []
        t = 0
        while True:
            p_g = self.queen_g.get()
            p_d = self.queen_d.get()
            SerializationTool.deserialize_model(net_g, p_g)
            SerializationTool.deserialize_model(net_d, p_d)
            self.train(net_d, net_g, loss, opti_g, opti_d)
            p_d = SerializationTool.serialize_model(net_d)
            p_g = SerializationTool.serialize_model(net_g)
            if servers[0].isAlive() is not True:
                break
            servers[0].queen_g.put((self.idx, copy.deepcopy(p_g)))
            servers[0].queen_d.put((self.idx, copy.deepcopy(p_d)))
            t += 1

    def train(self, net_d, net_g, loss, opti_g, opti_d):
        # start = time.time()
        #
        # i = 0
        # while i < epoch:
        #     try:
        #         imgs, _ = next(self.data)
        #     except StopIteration:
        #         self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        #         self.data = iter(self.dataloader)
        #         imgs, _ = next(self.data)
        #     if imgs.size()[0] != self.batch_size:
        #         continue
        #
        #     fake = Variable(torch.cuda.FloatTensor(imgs.size()[0], 1).fill_(0), requires_grad=False)
        #     valid = Variable(torch.cuda.FloatTensor(imgs.size()[0], 1).fill_(1), requires_grad=False)
        #     real_imgs = Variable(imgs.type(Tensor))
        #
        #     opti_g.zero_grad()
        #     z = Variable(Tensor(np.random.normal(0, 1, (imgs.size()[0], 100))))
        #     Xg = net_g(z)
        #     d_g = net_d(Xg)
        #     g_loss = loss(d_g, valid)
        #     g_loss.backward()
        #     opti_g.step()
        #
        #     opti_d.zero_grad()
        #     real_loss = loss(net_d(real_imgs), valid)
        #     fake_loss = loss(net_d(Xg.detach()), fake)
        #     D_loss = (real_loss + fake_loss) * 0.5
        #     D_loss.backward()
        #     opti_d.step()
        #
        #     i += 1
        # end = time.time()

        start = time.time()
        fake = Variable(Tensor(self.batch_size, 1).fill_(0), requires_grad=False)

        for i in range(epoch):
            for i, (imgs, _)in enumerate(self.dataloader):
                z = Variable(Tensor(np.random.normal(0, 1, (self.batch_size, 100))))
                Xd = net_g(z)
                real_imgs = Variable(imgs.type(Tensor))

                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1), requires_grad=False)
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
    labels = list(data.targets)
    global ims
    ims = (1, 28, 28)
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


def init_groups(size, cls_freq_wrk):
    """
	Initialization of all distributed groups for the whole training process. We do this in advance so as not to hurt the performance of training.
	The server initializes the group and send it to all workers so that everybody can agree on the working group at some round.
	Args
		size		The total number of machines in the current setup
		cls_freq_wrk	The frequency of samples of each class at each worker. This is used when the "sample" option is chosen. Otherwise, random sampling is applied and this parameter is not used.
    """
    global all_groups
    global all_groups_np
    global choose_r
    all_groups = []
    all_groups_np = []
    choose_r = []
    done = False
    gp_size = max(1, int(frac_workers*(size)))
    #If opt.sample is set, use the smart sampling, i.e., based on frequency of samples of each class at each worker. Otherwise, use random sampling

    #2D array that records if class i exists at worker j or not
    wrk_cls = [[False for i in range(10)] for j in range(size)]
    cls_q = [Queue(maxsize=size) for _ in range(10)]
    for i,cls_list in enumerate(cls_freq_wrk):
        wrk_cls[i] = [True if freq != 0 else False for freq in cls_list]
    for worker,class_list in enumerate(reversed(wrk_cls)):
        for cls,exist in enumerate(class_list):
            if exist:
                cls_q[cls].put(size - worker-1)
    #This array counts the number of samples (per class) taken for training so far. The algorithm will try to make the numbers in this array as equal as possible
    taken_count = [0 for i in range(10)]
    while not done:
        visited = [False for i in range(size)]  # makes sure that we take any worker only once in the group
        g = []
        for _ in range(gp_size):
            # Choose class (that is minimum represnted so far)...using "taken_count" array
            cls = np.where(taken_count == np.amin(taken_count))[0][0]
            assert cls >= 0 and cls <= len(taken_count)
            # Choose a worker to represnt that class...using wrk_cls and visited array
            done_q = False
            count = 0
            while not done_q:
                wrkr = cls_q[cls].get()
                assert wrk_cls[wrkr][cls]
                if not visited[wrkr] and wrk_cls[wrkr][cls]:
                    # Update the state: taken_count and visited
                    g.append(wrkr)
                    taken_count += cls_freq_wrk[wrkr]
                    visited[wrkr] = True
                    done_q = True
                cls_q[cls].put(wrkr)
                count += 1
                if count == size:  # Such an optimal assignment does not exist
                    done_q = True
        choose_r0 = False
        if 0 in g:
            choose_r0 = True
        else:
            choose_r0 = False
        choose_r.append(choose_r0)

        # assert len(g) > 1, "Number of sampled nodes per FL round is too low; consider increasing the number of nodes in the deployment or the fraction of chosen ndoes per round"

        # try:
            # group = dist.new_group(g, timeout=datetime.timedelta(0, timeout))
        # except Exception as e:
            # done = True
        all_groups_np.append(g)
        # all_groups.append(group)
        if len(all_groups_np) >= 20000:
            done = True
    return all_groups_np

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
        for k in range(2, 3):
            workers.clear()
            servers.clear()
            datasets.clear()
            iid = k

            SimulationName = time.strftime("%Y-%m-%d %H-%M-%S",
                                           time.localtime()) + "-FeGAN-" + dataset_name + "-iid_%d" % iid + "-epoch_%d" % epoch \
                             + "_share-%.1f" % segema + "_batchsize%d" % batch_size
            if not os.path.isdir('./logger'):
                os.mkdir('./logger')
            if not os.path.isdir('./logger/' + SimulationName):
                os.mkdir('./logger/' + SimulationName)
            allocate_dataset(dataset, iid)
            # datasets_parts = allocate_dataset(dataset, iid)
            #
            # csv_file = './logger/' + SimulationName + "/mnist_iid_{}_dir_0.1_100clients.csv".format(iid)
            #
            # partition_report(dataset.targets, datasets_parts.client_dict, class_num=num_class, verbose=False,
            #                  file=csv_file)

            y = [0 for i in range(num_class)]
            for target in dataset.targets:
                y[target] += 1
            y = np.array(y)
            y = y / sum(y)
            xs = []

            # for i in range(num_workers):
            #     dd = copy.deepcopy(dataset)
            #     dd.data = dataset.data[datasets_parts.client_dict[i]]
            #     dd.targets = dataset.targets[datasets_parts.client_dict[i]]
            #     x = [0 for i in range(num_class)]
            #     for target in dd.targets:
            #         x[target] += 1
            #     x = np.array(x)
            #     x_norm = x / sum(x)
            #     sk = stats.entropy(x_norm, y) * (x_norm.sum() / y.sum())
            #     workers.append(Worker(i, dataset=dd, sk=sk))
            #     xs.append(x)
            # groups = init_groups(num_workers, xs)

            for i in range(num_workers):
                x = [0 for _ in range(num_class)]
                for target in datasets[i].targets:
                    x[target] += 1
                x = np.array(x)
                x_norm = x / sum(x)
                sk = stats.entropy(x_norm, y) * (x_norm.sum() / y.sum())
                workers.append(Worker(i, dataset=datasets[i], sk=sk))
                xs.append(x)
            groups = init_groups(num_workers, xs)

            for i in range(num_servers):
                servers.append(Server(i, groups))

            print(torch.tensor([w.sk for w in workers]).softmax(dim=-1))

            for i in range(num_workers):
                servers[0].client_list.append(i)
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
