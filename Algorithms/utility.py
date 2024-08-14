import torch
import numpy as np
from torchvision import datasets, transforms


def data_next(dataloader, trainloader): # 한 round에 여러 device들이 학습되므로 dataloader가 차례차례 돌아가도록 하는 코드
    try: # 정상적인 경우
        data = dataloader.__next__()
    except: # dataloader가 다 끝났으면 다시 trainloader를 넣어줌
        dataloader = enumerate(trainloader)
        data = dataloader.__next__()
    return data, dataloader


def loader(args):

    if args.dataset == 'MNIST':
        transforms_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        traindata = datasets.MNIST(root='./data', train=True, download=True, transform=transforms_train)
        testdata = datasets.MNIST(root='./data', train=False, download=True, transform=transforms_test)
        test_loader = torch.utils.data.DataLoader(testdata, batch_size=args.test_batch_size, shuffle=False)
                
        target_labels = torch.stack([traindata.targets.clone().detach() == i for i in range(10)]) # MNIST

        index_per_label = []
        for i in range(10):
            index_per_label.append(torch.where(target_labels[i])[0])
            index_per_label[i] = index_per_label[i][torch.randperm(len(index_per_label[i]))]

        num_layer = int(args.num_workers / (10 / args.num_label_per_worker))

        traindata_split = []
        for i in range(num_layer):
            worker_label = np.random.choice(np.linspace(0, 9, num=10), size=10, replace=False)
            # worker_label = np.array([0, 2, 4, 6, 8, 1, 3, 5, 7, 9])
            # worker_label = np.linspace(0, 9, num=10)
            label_per_worker = np.split(worker_label, int(10 / args.num_label_per_worker))
            index_per_worker = ()
            for j in range(int(10 / args.num_label_per_worker)):
                tmp = torch.cat([index_per_label[int(label_per_worker[j][k])] for k in range(len(label_per_worker[j]))], 0)
                index_per_worker += (tmp[torch.randperm(len(tmp))],)
                traindata_split.append(torch.utils.data.Subset(traindata, index_per_worker[j]))

        
    elif args.dataset == 'CIFAR10':
        transforms_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        traindata = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)
        testdata = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)
        test_loader = torch.utils.data.DataLoader(testdata, batch_size=args.test_batch_size, shuffle=False)

        target_labels = torch.stack([torch.tensor(traindata.targets).clone().detach() == i for i in range(10)]) # CIFAR10

        index_per_label = []
        for i in range(10):
            index_per_label.append(torch.where(target_labels[i])[0])
            index_per_label[i] = index_per_label[i][torch.randperm(len(index_per_label[i]))]

        num_layer = int(args.num_workers / (10 / args.num_label_per_worker))

        traindata_split = []
        for i in range(num_layer):
            worker_label = np.random.choice(np.linspace(0, 9, num=10), size=10, replace=False)
            # worker_label = np.array([0, 2, 4, 6, 8, 1, 3, 5, 7, 9])
            # worker_label = np.linspace(0, 9, num=10)
            label_per_worker = np.split(worker_label, int(10 / args.num_label_per_worker))
            index_per_worker = ()
            for j in range(int(10 / args.num_label_per_worker)):
                tmp = torch.cat([index_per_label[int(label_per_worker[j][k])] for k in range(len(label_per_worker[j]))], 0)
                index_per_worker += (tmp[torch.randperm(len(tmp))],)
                traindata_split.append(torch.utils.data.Subset(traindata, index_per_worker[j]))



    elif args.dataset == 'CIFAR100':
        transforms_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomRotation(15),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
        ])
        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5088964127604166, 0.48739301317401956, 0.44194221124387256], 
                                 std=[0.2682515741720801, 0.2573637364478126, 0.2770957707973042]),
        ])

        traindata = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms_train)
        testdata = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms_test)
        test_loader = torch.utils.data.DataLoader(testdata, batch_size=args.test_batch_size, shuffle=False)

        target_labels = torch.stack([torch.tensor(traindata.targets).clone().detach() == i for i in range(100)]) # CIFAR10

        index_per_label = []
        for i in range(100):
            index_per_label.append(torch.where(target_labels[i])[0])
            index_per_label[i] = index_per_label[i][torch.randperm(len(index_per_label[i]))]

        num_layer = int(args.num_workers / (100 / args.num_label_per_worker))

        traindata_split = []
        for i in range(num_layer):
            worker_label = np.random.choice(np.linspace(0, 99, num=100), size=100, replace=False)
            label_per_worker = np.split(worker_label, int(100 / args.num_label_per_worker))
            index_per_worker = ()
            for j in range(int(100 / args.num_label_per_worker)):
                tmp = torch.cat([index_per_label[int(label_per_worker[j][k])] for k in range(len(label_per_worker[j]))], 0)
                index_per_worker += (tmp[torch.randperm(len(tmp))],)
                traindata_split.append(torch.utils.data.Subset(traindata, index_per_worker[j]))


    train_loader = []
    for i in range(args.num_workers):
        train_loader.append(torch.utils.data.DataLoader(traindata_split[i], batch_size=int(args.train_batch_size[i]), shuffle=True))          

    # for i, v in enumerate(train_loader[0]):
    #     print(len(v))
    #     break

    dataloader = []
    for i in range(args.num_workers):
        dataloader += [enumerate(train_loader[i])] 

    
    return dataloader, train_loader, test_loader

