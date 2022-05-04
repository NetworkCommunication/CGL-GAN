import os
import pickle as pkl
import threading
import time
from queue import Queue
from random import Random

import PIL.Image as Image
import numpy as np
import pandas as pd
import torch
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import datasets as torch_ds
from torchvision.utils import save_image
from tqdm import tqdm

from mnist_model import Discriminator, Generator

Tensor = torch.cuda.FloatTensor if torch.cuda.device_count() else torch.FloatTensor


plt.ion()
lock = threading.Lock()
SimulationName = None
num_communication = 20000
workers = []
servers = None

num_workers = 10
num_servers = 1
num_class = 10        # 需要大于等于 num_worker
num_sample = 1000    # 每个类别的样本数量
E = 1
iid = 0
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
frac_workers = 1     # 选择同步的工人的比例 （默认为全部）
epoch = 1            # 同步前的本地迭代次数
# overlap_worker = [[0], [0]]
ims = 0
b1 = 0.5
b2 = 0.999
img_size = 28


dataset = None


def plot_2d():
    df = pd.DataFrame()
    record = {}
    def interpolate(batch):
        arr = []
        for img in batch:
            if img.size()[0] == 1:
                img = torch.cat((img,) * 3, dim=0)
            pil_img = transforms.ToPILImage()(img)
            resized_img = pil_img.resize((299, 299), Image.BILINEAR)
            arr.append(transforms.ToTensor()(resized_img))
        return torch.stack(arr)

    def evaluation_step(engine, batch):
        with torch.no_grad():
            fake = interpolate(batch[0])
            real = interpolate(batch[1])
            return fake, real

    fid_metric = FID(device="cuda")
    is_metric = InceptionScore(device="cuda", output_transform=lambda x: x[0])
    evaluator = Engine(evaluation_step)
    fid_metric.attach(evaluator, "fid")
    is_metric.attach(evaluator, "is")

    for item in tqdm(range(num_communication//500)):
        D = []
        X = servers.queen_gen_data.get()
        D.append(X[::X.shape[0] // (num_sample // num_servers)])

        D = torch.cat(D)
        evaluator.run([[D[::D.shape[0] // 100], test_set[::len(test_set) // 100]]])
        # evaluator.run([[D, test_set]])
        metrics = evaluator.state.metrics
        fid_score = metrics['fid']
        is_score = metrics['is']
        record["FID"] = fid_score
        record["IS Score"] = is_score
        save_image(D[::D.shape[0] // 100], "./logger/" + SimulationName + "/%d.png" % item, nrow=10, normalize=True)
        df = df.append(record, ignore_index=True)
        df.to_excel("./logger/" + SimulationName + ".xlsx")
        with lock: print(f"*   FID : {fid_score:4f}", f"*    IS : {is_score:4f}")


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
        while t > 0:

            if t % 500 == 0:
                gen_data.append(self.plot_2d(net_g))
            # if t % E == 0:
            #     p_ds = []
            #     for idx in self.client_list:
            #         p_ds.append(self.queen_d.get())
            #     self.rd.shuffle(p_ds)
            #     for idx in self.client_list:
            #         workers[idx].para_d.put(p_ds[idx])
            self.train(net_g, opti_g, N)
            t -= 1

        torch.save(net_g.state_dict(), "./logger/"+SimulationName + "/{}.pt".format(self.name))
        with open("./logger/" + SimulationName + "/config{}.pkl".format(self.name), 'wb') as f:  # 将数据写入pkl文件
             pkl.dump((self.client_list, [workers[idx].test for idx in self.client_list], gen_data), f)

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
            workers[client].queen_d.put(Xd.clone())
            workers[client].queen_g.put(Xg.clone())

        opti.zero_grad()
        loss = torch.zeros(N)
        for i in range(N):
            g_loss = self.queen_g.get()
            # 更新个性层
            loss[i] = g_loss.clone()

        losses = loss.mean()
        losses.backward()
        opti.step()

        end = time.time()



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
        self.test = self.dataset[self.rd.sample(range(len(self.dataset)), num_sample)]
        self.data = iter(self.dataloader)

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
        print(self.name, ":starting waiting for service from", [item+1 for item in self.server_list])

        t = num_communication
        net_d = Discriminator(ims).cuda()
        loss = nn.BCELoss().cuda()
        opti_d = optim.Adam(net_d.parameters(), lr=self.lr_d, betas=(b1, b2))
        while t > 0:
            # if t % E == 0:
            #     p_d = self.copy_parameters(net_d)
            #     servers[0].queen_d.put(p_d)
            #     p_d = self.para_d.get()
            #     net_d.load_state_dict(p_d, strict=False)
            self.train(net_d, loss, opti_d)
            t -= 1

    def train(self, net_d, loss, opti_d):
        start = time.time()

        Xs = self.queen_d.get()
        for i in range(epoch):
            try:
                imgs = next(self.data)
            except StopIteration:
                self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
                self.data = iter(self.dataloader)
                imgs = next(self.data)

            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1), requires_grad=False)
            real_imgs = Variable(imgs.type(Tensor))

            opti_d.zero_grad()
            real_loss = loss(net_d(real_imgs), valid)

            fake = Variable(Tensor(self.batch_size, 1).fill_(0), requires_grad=False)
            fake_loss = loss(net_d(Xs), fake)
            D_loss = (real_loss + fake_loss)
            D_loss.backward()
            opti_d.step()

        valid = Variable(Tensor(self.batch_size, 1).fill_(1), requires_grad=False)

        Xg = self.queen_g.get()
        validaty = net_d(Xg)
        G_loss = loss(validaty, valid)
        servers.queen_g.put(G_loss)

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
    # sleep(14400)
    for l in range(2):
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
        for k in range(3):
            workers.clear()
            datasets.clear()
            iid = k

            SimulationName = time.strftime("%Y-%m-%d %H-%M-%S",
                                           time.localtime()) + "-MDGAN-"+dataset_name + "-iid_%d" % iid + "-epoch_%d" % epoch
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

            servers = Server(0)

            # 获取每个设备处于overlap的概率
            worker = [id for id in range(num_workers)]

            servers.client_list = worker

            for i in range(num_workers):
                workers[i].server_list.append(0)

            print("Simulation", SimulationName, " is started!!!")
            # 启动所有程序

            servers.start()

            for i in range(num_workers):
                workers[i].start()


            painter = threading.Thread(plot_2d())
            painter.start()
            servers.join()

            for i in range(num_workers):
                workers[i].join()

            painter.join()

            print("Simulation", SimulationName, " is over!!!")
