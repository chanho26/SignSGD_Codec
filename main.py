import torch
import numpy as np
import argparse
import os

from Algorithms.train_signSGD_FV import signSGD_FV
from Algorithms.train_signSGD_FD import signSGD_FD
from Algorithms.train_SGD import DSGD
from Algorithms.train_S3GD import S3GD_FV


parser = argparse.ArgumentParser(description='SignSGD-FV and other distributed learning algorithms')

parser.add_argument('--test_batch_size', type=int, default=64, help='Mini-batch size in inference(test)')
parser.add_argument('--num_it', type=int, default=3, help='Number of iterations')
parser.add_argument('--lr', type=float, default=10**-3, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
parser.add_argument('--momentum', type=float, default=0, help='Momentum')
parser.add_argument('--max_LLR', type=float, default=10, help='Weight clipping (for stable learning)')
parser.add_argument('--num_workers', type=int, default=15, help='Number of workers')
parser.add_argument('--num_label_per_worker', type=int, default=10, help='Number of labels per each worker (Non-IID dataset)')
parser.add_argument('--train_batch_size', type=int, default=1, choices=[1, 2, 3, 4, 5], help='Mini-batch size of each worker (batch mode)')
parser.add_argument('--weight_exp', type=float, default=1, choices=[0.631, 0.794, 0.925, 0.977, 0.992, 1], help='Attenuation level for the past error samples')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset')
parser.add_argument('--num_round', type=int, default=3001, help='Total number of training iterations + 1')
parser.add_argument('--test_round', type=int, default=100, help='Inference(test) cycle')
parser.add_argument('--learning_method', type=str, default='MV', choices=['MV', 'FV', 'FD', 'SGD', 'TopK'], help='Learning method')
parser.add_argument('--T_in', type=int, default='100', help='Initial phase duration')
parser.add_argument('--attacked_workers', type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help='Indexes of attacked workers')
parser.add_argument('--attack_method', type=str, default='det', choices=['det', 'sto', 'gauss', 'lie'], help='Attack method')
parser.add_argument('--sparsity', type=float, default='1', help='Sparsity level')
parser.add_argument('--spar_method', type=str, default='top', choices=['top', 'rand'], help='Sparsification method')
parser.add_argument('--accum_weight', type=float, default='1', help='Error accumulation factor')
parser.add_argument('--attack_prob', type=float, default='1', help='Stochastic sign flip probability')
parser.add_argument('--avg_batch_size', type=int, default=64)
parser.add_argument('--small_batch_size', type=int, default=4)

args = parser.parse_args()

# (Setting) Number of attacked workers
attacked_workers = args.attacked_workers

if args.attacked_workers == 0:
    args.attacked_workers = np.array([])

elif args.attacked_workers == 1:
    args.attacked_workers = np.array([0, 1, 2])

elif args.attacked_workers == 2:
    args.attacked_workers = np.array([0, 1, 2, 3, 4, 5])

elif args.attacked_workers == 3:
    args.attacked_workers = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

# Customize
elif args.attacked_workers == 4:
    args.attacked_workers = np.array([0, 1])

else:
    raise NotImplementedError('Invalid input argument: attacked_workers')


# (Setting) Batch mode
train_batch_size = args.train_batch_size
batch = np.ones(args.num_workers, dtype=int)

# batch mode 1
if args.train_batch_size == 1:
    num_large = args.num_workers

# batch mode 2
elif args.train_batch_size == 2:
    num_large = int(len(batch) * 1 / 2)

# batch mode 3
elif args.train_batch_size == 3:
    num_large = int(len(batch) / 5)

# batch mode 4
elif args.train_batch_size == 4:
    num_large = 1 

else:
    raise NotImplementedError('Invalid input argument: train_batch_size')

avg_batch_size = args.avg_batch_size
small_batch_size = args.small_batch_size

for i in range(len(batch)):
    if i < num_large:
        batch[i] = int((avg_batch_size * args.num_workers - small_batch_size * (args.num_workers - num_large)) / num_large)
    else:
        batch[i] = small_batch_size

args.train_batch_size = batch

print('Batch size: ', batch)




if args.sparsity == 1:
    if args.learning_method == 'FD':
        accuracy, test_loss = signSGD_FD(args, train_batch_size) 

    elif args.learning_method == 'SGD':
        accuracy, test_loss = DSGD(args)

    elif args.learning_method == 'FV' or 'MV':
        accuracy, test_loss = signSGD_FV(args, train_batch_size)

    else:
        raise NotImplementedError('Invalid input argument: learning_method')
    
else:
    if args.learning_method == 'TopK':
        accuracy, test_loss = DSGD(args)

    elif args.learning_method == 'FD' or 'FV' or 'MV':
        accuracy, test_loss = S3GD_FV(args)

    else:
        raise NotImplementedError('Invalid input argument: learning_method')


######################################### DSGD, S3GD_FV 수정 필요 #########################################

# results = {'args': args,
#            'acc': accuracy,
#            'train_loss': train_loss,
#            'test_loss': test_loss
#            }

# # Save results
# torch.save(results, os.getcwd()+'/Results/num_workers_'+str(args.num_workers)
#            +'/train_batch_size_'+str(train_batch_size)+'/'+args.dataset+'_'+args.learning_method+'.pth')
