import numpy as np
import torch
import torch.optim as optim
import time
import os
# import torchvision.models as torchmodel
from torchvision import transforms
from PIL import Image


from Algorithms.utility import data_next, loader, loader_text_AGNEWS
import Algorithms.model as model
import Algorithms.model_big as model_big


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

criterion = torch.nn.CrossEntropyLoss()
eps = 10 ** -5


def initial(args, device, vocab=None):
    # Neural network
    if args.dataset == 'MNIST':
        global_model = model.Net()

    elif args.dataset == 'CIFAR10':
        global_model = model.ResNet56()

    elif args.dataset == 'CIFAR100':
        global_model = model.ResNet56_CIFAR100()

    elif args.dataset == 'ImageNet':
        global_model = model_big.ResNet18(num_classes=100)
        # global_model = model.ResNet56_ImageNet()
        # global_model = model.ResNet110_ImageNet()

    elif args.dataset == '20Newsgroups':
        embed_dim = 128
        num_heads = 8
        num_layers = 4
        num_classes = 20
        
        if vocab:
            vocab_size = len(vocab)
            global_model = model_big.TransformerClassifier(vocab_size, embed_dim, num_heads, num_classes, num_layers)
        
        # global_model = model.TextClassifier_20News(vocab_size, embed_dim, num_classes)

    elif args.dataset == 'AGNews':
        vocab_size = len(vocab)
        embed_dim = 64
        hidden_dim = 128
        num_classes = 4
        num_heads = 8
        num_layers = 4
        global_model = model.TextClassificationModel(vocab_size, embed_dim, num_classes)
        # global_model = model.TextClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
        # global_model = model_big.TransformerClassifier(vocab_size, embed_dim, num_heads, num_classes, num_layers)

    else:
        raise NotImplementedError('Invalid: neural network')
    
    global_model = global_model.to(device)

    for i_v, v in enumerate(global_model.named_parameters()):
        print(i_v, v[0], v[1].shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    num = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print('Model parameters: ', num)
    

    global_opt = optim.SGD(global_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)


    stack_grad = []

    for _, v in enumerate(global_model.parameters()):
        shape_grad = v.shape + (args.num_workers,)
        stack_grad.append(torch.zeros(shape_grad).to(device))
    if torch.cuda.is_available():
        torch.cuda.synchronize()


    if args.learning_method == 'FD':
        count_error = []
        for _, v in enumerate(global_model.parameters()):
            shape_weight = v.shape + (args.num_workers,)
            count_error.append(torch.zeros(shape_weight, dtype=torch.int16).to(device))

    elif args.learning_method == 'MV':
        count_error = None

    else:
        raise NotImplementedError('Invalid input argument: learning_method')
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return global_model, global_opt, stack_grad, count_error
        


def train_clients(args, global_model, stack_grad, dataloader, train_loader, device):
    # No optim.step()
    total_loss = 0       

    for i in range(args.num_workers):
        global_model.train()

        for _, v in enumerate(global_model.parameters()):
            v.grad = None 

        if args.dataset == 'ImageNet':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            (_, (name, target)), dataloader[i] = data_next(dataloader[i], train_loader[i])
            train_folder_dir = './data/ImageNet/train'
            folders_name = os.listdir(train_folder_dir)
            batch_train_folder_dir = [train_folder_dir] * len(folders_name)
            batch_folders_name = [folders_name[target[f].item()] for f in range(len(name))]
            batch_file_name = [os.path.join(batch_train_folder_dir[f], batch_folders_name[f], name[f]) for f in range(len(name))]

            images = [transform_train(Image.open(img_path).convert("RGB")) for img_path in batch_file_name]
            time.sleep(0.1)
            data = torch.stack(images)

            data, target = data.to(device), target.to(device)
            output = global_model(data)
            train_loss = criterion(output, target)

        elif args.dataset == '20Newsgroups' or args.dataset == 'AGNews':
            (_, (texts, labels, lengths)), dataloader[i] = data_next(dataloader[i], train_loader[i])
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            time.sleep(0.1)
            output = global_model(texts, lengths)
            train_loss = criterion(output, labels)

        else:
            (_, (data, target)), dataloader[i] = data_next(dataloader[i], train_loader[i])
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            train_loss = criterion(output, target)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        train_loss.backward()

        total_loss += (train_loss / args.num_workers).item()

        for i_v, v in enumerate(global_model.parameters()):
            stack_grad[i_v][..., i] = torch.sign(v.grad).detach()

            if i in args.attacked_workers:
                stack_grad[i_v][..., i] = attack(args, stack_grad[i_v][..., i])

            v.grad = None
    
    return stack_grad, dataloader, train_loader, total_loss



def attack(args, gradient):
    if args.attack_method == 'det':
        mod_gradient = -gradient
    elif args.attack_method == 'sto':
        mod_gradient = (2 * (torch.rand(gradient.size()) < 0.5).int() - 1) * gradient
    elif args.attack_method == 'gauss':
        mod_gradient = torch.randn_like(gradient) * 1
    elif args.attack_method == 'lie':
        mod_gradient += torch.randn_like(gradient) * 0.01
    else:
        raise NotImplementedError('Invalid input argument: attack_method')

    return mod_gradient



def train_global(args, global_model, global_opt, stack_grad, r, count_error, device):
    global_model.train()
    global_opt.zero_grad()

    for i_v, v in enumerate(global_model.parameters()):
        v.grad = None
        
        if args.learning_method == 'MV':
            v.grad = torch.sign(torch.sum(stack_grad[i_v], dim=-1)).detach()

        elif args.learning_method == 'FD':
            if r == 0:
                wt = torch.ones_like(count_error[i_v])

            elif r < args.T_in:
                total_comp = sum(p.numel() for p in global_model.parameters() if p.requires_grad)

                if i_v == 0:
                    total_error = torch.zeros(args.num_workers).to(device)
                    for i in range(args.num_workers):
                        for j in range(len(count_error)):
                            total_error[i] += torch.sum(count_error[j][..., i])

                wt = torch.log(total_error / (total_comp * r - total_error))
                wt = torch.sign(wt) * torch.minimum(torch.abs(wt), args.num_workers * torch.ones_like(wt))

            else:
                wt = torch.log(count_error[i_v] / (r - count_error[i_v]))
                wt = torch.sign(wt) * torch.minimum(torch.abs(wt), args.num_workers * torch.ones_like(wt))


            if (args.dataset == '20Newsgroups' or args.dataset == 'AGNews') and (i_v == 0):
                # Weights are not applied to the embedding layers
                    v.grad = torch.sign(torch.sum(stack_grad[i_v], dim=-1)).detach()

            else:
                # Weights are not applied to the last layer
                if (i_v == len(count_error)-1) or (i_v == len(count_error)-2):
                    v.grad = torch.sign(torch.sum(stack_grad[i_v], dim=-1)).detach()

                else:
                    v.grad = torch.sign(torch.sum(stack_grad[i_v] * wt, dim=-1)).detach()

            count_error[i_v] += (v.grad.detach().unsqueeze(-1) * torch.ones(v.shape + (args.num_workers,)).to(device) == stack_grad[i_v]).int().detach()

        else:
            raise NotImplementedError('Invalid input argument: learning_method')

    global_opt.step()

    return global_model, global_opt, count_error
    

def test_model(global_model, test_loader, accuracy, test_loss, args, r, device):
    global_model.eval()
    test_lss = 0
    correct = 0

    iter_num = 0

    with torch.no_grad():
        if args.dataset == '20Newsgroups' or args.dataset == 'AGNews':
            for texts, labels, lengths in test_loader:
                texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
                time.sleep(0.1)
                output = global_model(texts, lengths)
                test_lss += criterion(output, labels).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                iter_num += 1
        
        else:
            for data, target in test_loader:
                if args.dataset == 'ImageNet':
                    transform_test = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

                    test_folder_dir = './data/ImageNet/val'
                    folders_name = os.listdir(test_folder_dir)
                    batch_test_folder_dir = [test_folder_dir] * len(folders_name)
                    batch_folders_name = [folders_name[target[f].item()] for f in range(len(data))]
                    batch_file_name = [os.path.join(batch_test_folder_dir[f], batch_folders_name[f], data[f]) for f in range(len(data))]

                    images = [transform_test(Image.open(img_path).convert("RGB")) for img_path in batch_file_name]
                    time.sleep(0.1)
                    data = torch.stack(images)

                data, target = data.to(device), target.to(device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                output = global_model(data)
                test_lss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                iter_num += 1

    test_lss /= iter_num
    accuracy[int((r + 1) / args.test_round)] += correct / (10 * len(test_loader.dataset))
    test_loss[int(r / args.test_round)] += test_lss / 10
    print(r + 1, '-th round test loss', test_lss)
    print(r + 1, '-th round test accuracy', correct / len(test_loader.dataset))
    print('\n')

    return accuracy, test_loss



def signSGD_FD(args, train_batch_size):
    # randomseed = np.linspace(0, int(20 * (args.num_it - 1)), num=args.num_it)
    randomseed = np.random.randint(1000, size=args.num_it)

    accuracy = torch.zeros(int(args.num_round / args.test_round) + 1).to(device)
    train_loss = torch.zeros(int(args.num_round / args.test_round) + 1).to(device)
    test_loss = torch.zeros(int(args.num_round / args.test_round) + 1).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    num_stop = 0

#################################################################################################################
    # If you need to load any pre-trained model (training is forced to stop):
    # tmp = torch.load('./Results/num_workers_15/train_batch_size_1/AGNews_FV_T_30_0.001.pth', weights_only=False)

    # accuracy = tmp['acc']
    # train_loss = tmp['train_loss']
    # test_loss = tmp['test_loss']

    # num_stop = (torch.count_nonzero(accuracy) - 1) * args.test_round + 1
#################################################################################################################

    for it in range(args.num_it):
        print('Iteration: ', int(it+1))
        torch.manual_seed(randomseed[it])

        if args.dataset == '20Newsgroups':
            dataloader, train_loader, test_loader, vocab = loader_text(args)
            global_model, global_opt, stack_grad, count_error = initial(args, device, vocab)

        elif args.dataset == 'AGNews':
            dataloader, train_loader, test_loader, vocab = loader_text_AGNEWS(args)
            global_model, global_opt, stack_grad, count_error = initial(args, device, vocab)

        else:
            dataloader, train_loader, test_loader = loader(args)
            global_model, global_opt, stack_grad, count_error = initial(args, device)

#################################################################################################################
        # If you need to load any pre-trained model (training is forced to stop):
        # global_model.load_state_dict(tmp['params'])
        # count_error = tmp['count_error']
#################################################################################################################

        for r in range(int(num_stop), args.num_round, 1):
            if r % 10 == 0:
                print('Training round: ', r)

            stack_grad, dataloader, train_loader, train_lss = train_clients(args, global_model, stack_grad, dataloader, train_loader, device)

            global_model, global_opt, count_error = train_global(args, global_model, global_opt, stack_grad, r, count_error, device)

            if r % args.test_round == 0:
                train_loss[int(r / args.test_round)] += train_lss / 10
                print('# of workers : ', args.num_workers, ', batch mode : ', train_batch_size, ' ------------------------- \n')
                print(r + 1, '-th round train loss', train_lss)
                accuracy, test_loss = test_model(global_model, test_loader, accuracy, test_loss, args, r, device)

                results = {'args': args,
                           'acc': accuracy, 
                           'train_loss': train_loss, 
                           'test_loss': test_loss, 
                           'params': global_model.state_dict(), 
                           'count_error': count_error
                           }
                torch.save(results, './Results/num_workers_'+str(args.num_workers)+'/train_batch_size_'+str(train_batch_size)
                           +'/'+str(args.dataset)+'_'+str(args.learning_method)+'_T_'+str(args.T_in)+'_'+str(args.lr)+'.pth')

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return accuracy, train_loss, test_loss
