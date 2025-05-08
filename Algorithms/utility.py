import torch
import numpy as np
from torchvision import datasets, transforms
import scipy
import os
import random
import copy


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


    elif args.dataset == 'ImageNet':

        def list_chunk(lst, n):
            quo = len(lst) // n
            rem = len(lst) % n

            num_elems = [quo] * n
            for i in range(rem):
                num_elems[i] += 1

            num_start = [0]
            for i in range(n-1):
                num_start.append(num_start[i] + num_elems[i])

            return [lst[num_start[i]: num_start[i]+num_elems[i]] for i in range(n)]

        traindata_split = [copy.deepcopy([]) for i in range(args.num_workers)]
        train_folder_dir = './data/ImageNet/train'
        folders_name = os.listdir(train_folder_dir)
        for idx, name in enumerate(folders_name[:100]):
            image_dir = os.path.join(train_folder_dir, name)
            file_name = os.listdir(image_dir)
            file_tuple = list(zip(file_name, [idx] * len(file_name)))
            random.shuffle(file_tuple)
            split_files = list_chunk(file_tuple, args.num_workers)
            for j in range(args.num_workers):
                traindata_split[j] += split_files[j]

        
        testdata = []       
        test_folder_dir = './data/ImageNet/val'
        folders_name = os.listdir(test_folder_dir)
        for idx, name in enumerate(folders_name[:100]):
            image_dir = os.path.join(test_folder_dir, name)
            file_name = os.listdir(image_dir)
            file_tuple = list(zip(file_name, [idx] * len(file_name)))
            testdata += file_tuple 


        # traindata = datasets.ImageNet(root='./data/ImageNet', split='train', transform=transform_train)
        # testdata = datasets.ImageNet(root='./data/ImageNet', split='val', transform=transform_test)

        test_loader = torch.utils.data.DataLoader(testdata, batch_size=args.test_batch_size, shuffle=False, pin_memory=False)
 
    train_loader = []

    for i in range(args.num_workers):
        train_loader.append(torch.utils.data.DataLoader(traindata_split[i], batch_size=int(args.train_batch_size[i]), shuffle=True, pin_memory=False))

    # for idx, data in enumerate(train_loader[0]):
    #     print(idx, data)
    #     asdfadsfdsf


    dataloader = []
    for i in range(args.num_workers):
        dataloader += [enumerate(train_loader[i])] 

    
    return dataloader, train_loader, test_loader



def loader_text(args):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder

    # 1. 데이터 준비
    class NewsGroupDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, vocab):
            self.texts = []
            self.labels = []
            self.vocab = vocab

            # 빈 텍스트 필터링 (데이터셋 내에 비어 있는 (공백) 데이터가 있나봄)
            for text, label in zip(texts, labels):
                if text.strip():    # 빈 텍스트가 아닌 경우에만 추가
                    self.texts.append(text)
                    self.labels.append(label)

        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            tokenized = [self.vocab.get(word, self.vocab['<unk>']) for word in text.split()]
            return torch.tensor(tokenized, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        
    # 데이터를 가져오고 전처리
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    texts, labels = newsgroups.data, newsgroups.target

    # 데이터셋 분할
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 단어 빈도를 기반으로 어휘 새성
    vectorizer = CountVectorizer(max_features=20000, stop_words='english')
    vectorizer.fit(train_texts)
    vocab = {word: idx + 1 for idx, word in enumerate(vectorizer.get_feature_names_out())}
    vocab['<unk>'] = 0

    # 데이터셋 및 데이터로더 생성
    train_dataset = NewsGroupDataset(train_texts, train_labels, vocab)
    test_dataset = NewsGroupDataset(test_texts, test_labels, vocab)

    traindata_split = [copy.deepcopy([]) for i in range(args.num_workers)]

    idx_data_per_label = [copy.deepcopy([]) for _ in range(len(newsgroups.target_names))]

    for idx, (_, label) in enumerate(train_dataset):
        idx_data_per_label[int(label)].append(idx)

    def list_chunk(lst, n):
        quo = len(lst) // n
        rem = len(lst) % n

        num_elems = [quo] * n
        for i in range(rem):
            num_elems[i] += 1

        num_start = [0]
        for i in range(n-1):
            num_start.append(num_start[i] + num_elems[i])

        return [lst[num_start[i]: num_start[i]+num_elems[i]] for i in range(n)]

    for i in range(len(idx_data_per_label)):
        random.shuffle(idx_data_per_label[i])
        split_data = list_chunk(idx_data_per_label[i], args.num_workers)

        for j in range(args.num_workers):
            traindata_split[j] += [train_dataset[idx] for idx in split_data[j]]   

    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: collate_batch(x, vocab))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=lambda x: clip_sequences(x, vocab))

    def collate_batch(batch, vocab):
        texts, labels = zip(*batch)
        lengths = [len(text) for text in texts]
        padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=vocab['<unk>'])
        return padded_texts, torch.tensor(labels), torch.tensor(lengths)     
    
    def clip_sequences(batch, vocab, max_len=512):
        texts, labels = zip(*batch)
        texts = [text[:max_len] for text in texts]
        lengths = [len(text) for text in texts]
        texts = torch.nn.utils.rnn.pad_sequence([torch.tensor(text) for text in texts], batch_first=True, padding_value=vocab['<unk>'])
        return texts, torch.tensor(labels), torch.tensor(lengths)
    


    train_loader = []
    for i in range(args.num_workers):
        # train_loader.append(torch.utils.data.DataLoader(traindata_split[i], batch_size=int(args.train_batch_size[i]), \
        #                                                 shuffle=True, collate_fn=lambda x: collate_batch(x, vocab)))
        train_loader.append(torch.utils.data.DataLoader(traindata_split[i], batch_size=int(args.train_batch_size[i]), \
                                                        shuffle=True, collate_fn=lambda x: clip_sequences(x, vocab)))

    # for idx, data in enumerate(train_loader[0]):
    #     print(idx, data)
    #     asdfadsfdsf

    dataloader = []
    for i in range(args.num_workers):
        dataloader += [enumerate(train_loader[i])] 

    
    return dataloader, train_loader, test_loader, vocab



def loader_text_AGNEWS(args):
    from torchtext.datasets import AG_NEWS
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    from torch.nn.utils.rnn import pad_sequence


    def yield_tokens(data_iter, tokenizer):
        for _, text in data_iter:
            yield tokenizer(text)

    tokenizer = get_tokenizer("basic_english")
    train_iter, test_iter = AG_NEWS(split=("train", "test"))
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def text_pipeline(x):
        return vocab(tokenizer(x))
    
    def label_pipeline(x):
        return int(x) - 1
    
    def collate_batch(batch):
        # text_list, label_list = [], []
        text_list, label_list, offsets = [], [], [0]

        for label, text in batch:
            # text_list.append(torch.tensor(text_pipeline(text), dtype=torch.long))
            # label_list.append(torch.tensor(label_pipeline(label), dtype=torch.long))

            label_list.append(torch.tensor(label_pipeline(label), dtype=torch.long))
            processed_text = torch.tensor(text_pipeline(text), dtype=torch.long)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))

        # text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
        # label_list = torch.tensor(label_list, dtype=torch.long)

        label_list = torch.tensor(label_list, dtype=torch.long)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)

        # return text_list, label_list
        return text_list, label_list, offsets
    
    train_dataset, test_dataset = list(train_iter), list(test_iter)

    traindata_split = [copy.deepcopy([]) for _ in range(args.num_workers)]
    idx_data_per_label = [copy.deepcopy([]) for _ in range(4)]

    for idx, (label, _) in enumerate(train_dataset):
        idx_data_per_label[label_pipeline(label)].append(idx)

    def list_chunk(lst, n):
        quo = len(lst) // n
        rem = len(lst) % n

        num_elems = [quo] * n
        for i in range(rem):
            num_elems[i] += 1

        num_start = [0]
        for i in range(n-1):
            num_start.append(num_start[i] + num_elems[i])

        return [lst[num_start[i]: num_start[i]+num_elems[i]] for i in range(n)]

    for i in range(len(idx_data_per_label)):
        random.shuffle(idx_data_per_label[i])
        split_data = list_chunk(idx_data_per_label[i], args.num_workers)

        for j in range(args.num_workers):
            traindata_split[j] += [train_dataset[idx] for idx in split_data[j]]   

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_batch)
    
    train_loader = []
    for i in range(args.num_workers):
        train_loader.append(torch.utils.data.DataLoader(traindata_split[i], batch_size=int(args.train_batch_size[i]), \
                                                        shuffle=True, collate_fn=collate_batch))

    # for idx, data in enumerate(train_loader[0]):
    #     print(idx, data)
    #     asdfadsfdsf

    dataloader = []
    for i in range(args.num_workers):
        dataloader += [enumerate(train_loader[i])] 

    
    return dataloader, train_loader, test_loader, vocab
