import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
#Example based on https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

TENSOR_SIZE = 512*1024

class SingleLayer(nn.Module):
    def __init__(self):
        super(SingleLayer, self).__init__()
        self.fc = nn.Linear(TENSOR_SIZE, 1)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        out = self.fc(x)
        return out

class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, gpu, start, end):
         super(MyIterableDataset).__init__()
         assert end > start, "this example code only works with end >= start"
         self.start = start
         self.end = end
         self.gpu = gpu

    def __iter__(self):
         worker_info = torch.utils.data.get_worker_info()
         if worker_info is None:  # single-process data loading, return the full iterator
             self.iter_start = self.start
             self.iter_end = self.end
             self.per_worker = self.end - self.start
         else:  # in a worker process
             # split workload
             self.per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             self.iter_start = self.start + worker_id * per_worker
             self.iter_end = min(iter_start + per_worker, self.end)
            #  t = torch.ones([TENSOR_SIZE], dtype=torch.float32)
            #  t.data.fill_(self.gpu)
         return iter([torch.ones([TENSOR_SIZE], dtype=torch.float32).cuda(non_blocking=True).data.fill_(self.gpu) for i in range(self.iter_start, self.iter_end)])

    def __len__(self):
        if not hasattr(self, 'per_worker'):
            self.__iter__()
        
        return self.per_worker

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    orig_model = SingleLayer()
    torch.cuda.set_device(gpu)
    orig_model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.L1Loss().cuda(gpu)
    optimizer = torch.optim.SGD(orig_model.parameters(), 1)
   
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(orig_model, device_ids=[gpu])
    params = [m for m in model.parameters()]
    # Data loading code
    train_dataset = MyIterableDataset(start = 0, end = 1000, gpu = gpu)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=False
                                            #    sampler=train_sampler
                                               )

    start = datetime.now()
    alpha = torch.ones([1], dtype=torch.float32).cuda(non_blocking=True)
    total_step = len(train_loader)/batch_size
    labels = torch.ones([100, 1], dtype=torch.long).cuda(non_blocking=True) #labels.cuda(non_blocking=True)

    for epoch in range(1):
        for i, images in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            
            #print("optimizer.step()")
            #print(params[0].grad)
            if False:
                loss.backward()
                # weight = torch.ones([TENSOR_SIZE], dtype=torch.float32).cuda(non_blocking=True)
                # grad = torch.ones([TENSOR_SIZE], dtype=torch.float32).cuda(non_blocking=True)
                torch.distributed.sgd_update(params[0], params[0].grad, alpha)
                # print("grad", params[0].grad)
                # print("weight", orig_model.fc.weight)
            else:
                loss.backward()
                optimizer.step()
                # print("grad", params[0].grad)
                # print("weight", params[0])
            # torch.distributed.all_reduce(torch.zeros([TENSOR_SIZE], dtype=torch.float32).cuda(non_blocking=True))
           # print(model.grad)
            if (i + 1) % 1 == 0:
                print('GPU {} Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(gpu, epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    main()