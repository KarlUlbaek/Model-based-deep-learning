import os

import torch
from torch.utils.data import DataLoader
from math import ceil


class custom_data_loader():
    def __init__(self, split_name="train", batch_size=8, device='cpu', data_set=None, name="noname"):

        if split_name not in ["train", "test", "val"]:
            print("pick er proper split you donkey among:", ["train", "test", "val"])
            raise NameError

        self.split_name = split_name
        self.d = device
        self.data_set = data_set
        self.name = name + "_" + split_name

        if not self.split_name == "train":
            self.batch_size = 100000
        else:
            self.batch_size = batch_size

        try:
            self.load_data()
            print("{} was loaded".format(self.name))
            self.n = self.x.shape[0]
            self.num_batches = int(ceil(self.n/self.batch_size))
            self.random_idx = self.create_random_idx_list()

        except FileNotFoundError:
            if data_set is not None:
                print("{} was not found. creating data".format(self.name))
                self.save_tensor_data()
                print("{} created and loaded".format(self.name))
                self.n = self.x.shape[0]
                self.num_batches = int(ceil(self.n / self.batch_size))
                self.random_idx = self.create_random_idx_list()

            else:
                print("no data loaded, set .x .y .n and set_batch_size() manually")
                self.n = None
                self.num_batches = None
                self.random_idx = None

        # if self.d == 'cpu' and torch.cuda.is_available():
        #     print("running on CPU but cuda is available.")

    def create_random_idx_list(self):
        perm = torch.randperm(self.n-1)
        return [perm[i*self.batch_size:(i+1)*self.batch_size] for i in range(self.num_batches)]

    # def create_random_idx_list(self):
    #     perm = torch.randperm(self.n - 1)
    #     split_idx = [i * self.batch_size for i in range(self.num_batches)]
    #     split_idx.append(self.n)
    #
    #     random_idx = [perm[split_idx[i]:split_idx[i + 1]]
    #                        for i in range(self.num_batches)]
    #     return random_idx

    def save_tensor_data(self):
        batch_size = 10000000
        loader = DataLoader(self.data_set, batch_size=batch_size, shuffle=False)

        x, y = next(iter(loader))

        torch.save(x, self.name+"_x_.pt")
        torch.save(y, self.name+"_y_.pt")

        self.x = x.to(self.d)
        self.y = y.to(self.d)

        self.n = self.x.shape[0]

    def load_data(self):

        self.x = torch.load(self.name+"_x_.pt").to(self.d)
        self.y = torch.load(self.name+"_y_.pt").to(self.d)

        self.n = self.x.shape[0]


    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.num_batches = int((self.n/self.batch_size))
        self.random_idx = self.create_random_idx_list()

    def __getitem__(self, item):
        if self.batch_size == 1: #unsqueeze i.e. keep batch dimension
            return self.x[:self.batch_size][None,:], self.y[:self.batch_size][None,:]
        else:
            return self.x[:self.batch_size], self.y[:self.batch_size]

    def __iter__(self):
        self.i = -1
        return self

    def __next__(self):
        self.i += 1
        if self.i < self.num_batches:
            if self.batch_size == 1: #unsqueeze i.e. keep batch dimension
                return self.x[self.random_idx[self.i]][None,:], self.y[self.random_idx[self.i]][None,:]
            else:
                return self.x[self.random_idx[self.i]], self.y[self.random_idx[self.i]]

        else:
            self.random_idx = self.create_random_idx_list()
            raise StopIteration

    def __len__(self):
        return self.n








