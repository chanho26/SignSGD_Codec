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
    elif args.dataset == 'CIFAR100':
        global_model = model.ResNet56_CIFAR100()
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

    
    count_error = []
    total_count = []
    for i_v in range(len(tmp)):
        shape_layer = tmp[i_v].shape + (args.num_workers,)

        # Count the number of errors 
        count_error.append(torch.zeros(shape_layer))

        # Total counts that each worker sends the gradient coordinate
        total_count.append(torch.zeros(shape_layer))

    return global_model, client_models, global_opt, client_opts, tmp, count_error, total_count


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

        # print('Output :', output)
        # print('Target :', target)
        
        train_loss = criterion(output, target)
        train_loss.backward()
    
    return client_models, client_opts, dataloader, train_loader


def calc_weight(args, r, count_error, total_count):
    weight = []
    if r == 0:
        for i_v in range(len(count_error)):
            weight.append(torch.ones_like(count_error[i_v]))

    else:
        for i_v in range(len(count_error)):
            prob_right = count_error[i_v] / (total_count[i_v] + eps)

            tmp_weight = torch.log((prob_right + eps) / (1 - prob_right + eps))
            tmp_weight = torch.sign(tmp_weight) * torch.minimum(torch.abs(tmp_weight), args.max_LLR * torch.ones_like(tmp_weight))

            weight.append(tmp_weight)
    
    return weight


def pre_processing(args, r, client_models, i, est_grad, weight):
    for i_v, v in enumerate(client_models[i].parameters()):
        if args.learning_method == 'MV':
            est_grad[i_v] += torch.sign(v.grad)
        elif args.learning_method == 'FV':
            if r < args.T_in:
                est_grad[i_v] += torch.sign(v.grad)
            else:
                est_grad[i_v] += torch.sign(v.grad) * weight[i_v][..., i]
        else:
            raise NotImplementedError('Invalid input argument: learning_method')
    return est_grad


def update_count(args, client_models, count_error, total_count, est_grad):
    for i in range(args.num_workers):
        for i_v, v in enumerate(client_models[i].parameters()):
            count_error[i_v][..., i] *= args.weight_exp
            total_count[i_v][..., i] *= args.weight_exp

            total_count[i_v][..., i] += 1
            if args.learning_method == 'FV':
                count_error[i_v][..., i] += (est_grad[i_v] == torch.sign(v.grad))
    return count_error, total_count


# def attack(args, client_model):
#     for i_v, v in enumerate(client_model.parameters()):
#         if args.attack_method == 'det':
#             v.grad = -v.grad
#         elif args.attack_method == 'sto':
#             v.grad = (2 * (torch.rand(v.size()) < 0.5).int() - 1) * v.grad
#         else:
#             continue
#     return client_model


def grad_processing(args, client_models, tmp, r, weight, count_error, total_count):
    est_grad = copy.deepcopy(tmp)

    for i in range(args.num_workers):
        # # Attack
        # if sum(i == args.attacked_workers) == 1:
        #     client_models[i] = attack(args, client_models[i])
        
        # Gradient processing
        est_grad = pre_processing(args, r, client_models, i, est_grad, weight)

    # Majority vote (MV)
    for i_v in range(len(est_grad)):
        est_grad[i_v] = torch.sign(est_grad[i_v]) # est_grad[i_v] / args.num_workers

    # Count error
    count_error, total_count = update_count(args, client_models, count_error, total_count, est_grad)
    
    return est_grad, count_error, total_count


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
    

def test_model(global_model, test_loader, accuracy, args, r, train_batch_size):
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
    print('# of workers : ', args.num_workers, ', batch mode : ', train_batch_size, ' ------------------------- \n')
    print(r + 1, '-th round test loss', test_loss)
    print(r + 1, '-th round test accuracy', correct / len(test_loader.dataset))
    print('\n')

    return accuracy, test_loss



def signSGD_FV(args, train_batch_size):
    # randomseed = np.linspace(0, int(20 * (args.num_it - 1)), num=args.num_it)
    randomseed = np.random.randint(1000, size=args.num_it)

    accuracy = torch.zeros(int(args.num_round / args.test_round) + 1).to(device)

    for it in range(args.num_it):
        torch.manual_seed(randomseed[it])

        dataloader, train_loader, test_loader = loader(args)

        global_model, client_models, global_opt, client_opts, tmp, count_error, total_count = initial(args)

        global_model, client_models = transition_unit(global_model, client_models, device, args.num_workers)

        for r in range(args.num_round):
            global_model, client_models = transition_unit(global_model, client_models, device, args.num_workers)

            client_models, client_opts, dataloader, train_loader = train_clients(args, client_models, client_opts, dataloader, train_loader)

            global_model, client_models = transition_unit(global_model, client_models, torch.device('cpu'), args.num_workers)

            weight = calc_weight(args, r, count_error, total_count)

            est_grad, count_error, total_count = grad_processing(args, client_models, tmp, r, weight, count_error, total_count)

            global_model, global_opt = train_global(global_model, global_opt, tmp, est_grad)

            client_models = distribute(global_model, client_models, args)

            if r % args.test_round == 0:
                accuracy, test_loss = test_model(global_model, test_loader, accuracy, args, r, train_batch_size)

    return accuracy, test_loss
