import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

from Algorithms.utility import data_next, loader
import Algorithms.model as model

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

criterion = torch.nn.CrossEntropyLoss()
eps = 10 ** -5


def initial(args):
    # Neural network
    if args.dataset == 'MNIST':
        global_model = model.Net()
    elif args.dataset == 'CIFAR10':
        global_model = model.ResNet56()
    else:
        raise NotImplementedError('Invalid: neural network')
    
    client_models = [copy.deepcopy(global_model) for _ in range(args.num_workers)]

    global_opt = optim.SGD(global_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    client_opts = [optim.SGD(tmp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
                   for tmp_model in client_models]
    
    # Zeros of neural network model
    tmp = []
    for i_v, v in enumerate(global_model.parameters()):
        tmp.append(torch.zeros_like(v))

    # Top-K error accumulation
    accum_grad = []
    for i in range(args.num_workers):
        accum_grad.append(copy.deepcopy(tmp))

    return global_model, client_models, global_opt, client_opts, tmp, accum_grad


def transition_unit(global_model, client_models, device, num_workers):
    global_model = global_model.to(device)
    for i in range(num_workers):
            client_models[i] = client_models[i].to(device)

    return global_model, client_models
        


def train_clients(args, client_models, client_opts, dataloader, train_loader):
    # No optim.step()
    for i in range(args.num_workers):
        client_models[i].train()
        (idx, (data, target)), dataloader[i] = data_next(dataloader[i], train_loader[i])
        data, target = data.to(device), target.to(device)
        client_opts[i].zero_grad()
        output = client_models[i](data)

        train_loss = criterion(output, target)
        train_loss.backward()
    
    return client_models, client_opts, dataloader, train_loader



def sparsification(args, gradient, accum_grad):
    if args.spar_method == 'top':
        accum_grad *= args.accum_weight # no accumulation -> 0
        accum_grad += gradient
        flatten = torch.reshape(accum_grad, (-1,))
        num_gradient = torch.numel(flatten)
        num_select = int(np.ceil(num_gradient * args.sparsity))
        _, index_select = torch.topk(abs(flatten), num_select)

        # threshold = abs(flatten[index_select[-1]])

        spar_gradient = torch.zeros(flatten.shape)
        spar_gradient[index_select] = flatten[index_select]
        flatten[index_select] = 0
        accum_grad = torch.reshape(flatten, gradient.shape)

    else:
        flatten = torch.reshape(gradient, (-1,))
        num_gradient = torch.numel(flatten)
        num_select = int(np.ceil(num_gradient * args.sparsity))
        index_select = np.random.choice(num_gradient, size=num_select, replace=False)
        spar_gradient = torch.zeros(flatten.shape)
        spar_gradient[index_select] = flatten[index_select]

    mod_gradient = torch.reshape(spar_gradient, gradient.shape)
    return mod_gradient, accum_grad



def pre_processing(args, client_models, i, est_grad, accum_grad, num_workers_sent):
    for i_v, v in enumerate(client_models[i].parameters()):
        if args.learning_method == 'SGD':
            est_grad[i_v] += v.grad
        elif args.learning_method == 'TopK':
            v.grad, accum_grad[i][i_v] = sparsification(args, v.grad, accum_grad[i][i_v])
            num_workers_sent[i_v] += (v.grad != 0).float()
            est_grad[i_v] += v.grad

    return est_grad, accum_grad, num_workers_sent


def attack(args, client_model):
    for i_v, v in enumerate(client_model.parameters()):
        if args.attack_method == 'det':
            v.grad = -v.grad
        elif args.attack_method == 'sto':
            v.grad = (2 * (torch.rand(v.size()) < 0.5).int() - 1) * v.grad
        elif args.attack_method == 'gauss':
            v.grad = torch.randn_like(v.grad) * 1
        else:
            continue
    return client_model


def grad_processing(args, client_models, tmp, accum_grad):
    eps = 1e-5
    est_grad = copy.deepcopy(tmp)

    num_workers_sent = []
    for i_v in range(len(tmp)):
        num_workers_sent.append(torch.zeros_like(tmp[i_v]))
    
    for i in range(args.num_workers):
        # Attack
        if sum(i == args.attacked_workers) == 1:
            client_models[i] = attack(args, client_models[i])
        
        # Gradient processing
        est_grad, accum_grad, num_workers_sent = pre_processing(args, client_models, i, est_grad, accum_grad, num_workers_sent)

    # Averaging
    for i_v in range(len(est_grad)):
        if args.learning_method == 'SGD':
            est_grad[i_v] = est_grad[i_v] / args.num_workers
        elif args.learning_method == 'TopK':
            est_grad[i_v] = est_grad[i_v] / (num_workers_sent[i_v] + eps)
    
    return est_grad, accum_grad


def train_global(global_model, global_opt, tmp, est_grad):
    global_model.train()
            
    for i_v, v in enumerate(global_model.parameters()):
        v.grad = torch.zeros_like(tmp[i_v])
        v.grad = est_grad[i_v]

    global_model = global_model.to(device)

    global_opt.step()

    return global_model, global_opt


def distribute(global_model, client_models, args):
    ttmp = []
    for i_v, v in enumerate(global_model.parameters()):
        ttmp.append(v)

    for i in range(args.num_workers):
        for i_v, v in enumerate(client_models[i].parameters()):
            v.data = ttmp[i_v]

    return client_models
    

def test_model(global_model, test_loader, accuracy, args, r):
    global_model.eval()
    test_loss = 0
    correct = 0

    iter_num = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            iter_num += 1

    test_loss /= iter_num
    accuracy[int((r + 1) / args.test_round)] += correct / (args.num_it * len(test_loader.dataset))
    print('# of workers : ', args.num_workers, ', Attacked worker : ', args.attacked_workers, ' ------------------------- \n')
    print(r + 1, '-th round test loss', test_loss)
    print(r + 1, '-th round test accuracy', correct / len(test_loader.dataset))
    print('\n')

    return accuracy, test_loss



def DSGD(args):
    # randomseed = np.linspace(0, int(20 * (args.num_it - 1)), num=args.num_it)
    randomseed = np.random.randint(1000, size=args.num_it)

    accuracy = torch.zeros(int(args.num_round / args.test_round) + 1).to(device)

    for it in range(args.num_it):
        torch.manual_seed(randomseed[it])

        dataloader, train_loader, test_loader = loader(args)

        global_model, client_models, global_opt, client_opts, tmp, accum_grad = initial(args)

        global_model, client_models = transition_unit(global_model, client_models, device, args.num_workers)

        for r in range(args.num_round):
            global_model, client_models = transition_unit(global_model, client_models, device, args.num_workers)

            client_models, client_opts, dataloader, train_loader = train_clients(args, client_models, client_opts, dataloader, train_loader)

            global_model, client_models = transition_unit(global_model, client_models, torch.device('cpu'), args.num_workers)

            est_grad, accum_grad = grad_processing(args, client_models, tmp, accum_grad)

            global_model, global_opt = train_global(global_model, global_opt, tmp, est_grad)

            client_models = distribute(global_model, client_models, args)

            if r % args.test_round == 0:
                accuracy, test_loss = test_model(global_model, test_loader, accuracy, args, r)

    return accuracy, test_loss



            




            



            


            
                

        




