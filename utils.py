import numpy as np
import torch
import torch.nn as nn
import os
import scipy.sparse as sp
import time
import torch.nn.functional as F
import random

def init_random_seed(SEED=2021):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(SEED)
init_random_seed()

"""
def count_arr(predictions, nclass,grads = False):
    if not grads:
        predictions.detach()
    predictions = F.one_hot(predictions,num_classes = nclass)
    return predictions
"""

def count_arr(predictions, nclass):
    nodes_n=predictions.shape[0]
    counts = np.zeros((nodes_n,nclass), dtype=int)
    for n,idx in enumerate(predictions):
        counts[n,idx] += 1
    return counts

def train_smoothing_model(model,dataset,optimizer,args):
    endure_count = 0
    best_acc_val = 0
    adj, features, labels, idx_train, idx_val, idx_test = dataset
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model.forward_perturb(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        if epoch %10==0:
            model.eval()
            output = model.forward_perturb(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s\n'.format(time.time() - t))
            if  acc_val > best_acc_val:
                best_acc_val = acc_val
                endure_count = 0
                torch.save(model, args.model_dir)
                print('model saved to:',args.model_dir)
            else:
                endure_count += 1
            if endure_count > args.patience:
                print('early stop at epoch:',epoch,'\n')
                break

def load_data(path):
    graph = np.load(path)
    A = sp.csr_matrix((np.ones(graph['A'].shape[1]).astype(int), graph['A']))
    data = (np.ones(graph['X'].shape[1]), graph['X'])
    X = sp.csr_matrix(data, dtype=np.float32).todense()
    y = graph['y']
    n, d = X.shape
    nc = y.max() + 1
    return A, X, y, n, d, nc

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def split(labels, n_per_class=20, seed=0):
    """
    Randomly split the training data.

    Parameters
    ----------
    labels: array-like [n_nodes]
        The class labels
    n_per_class : int
        Number of samples per class
    seed: int
        Seed

    Returns
    -------
    split_train: array-like [n_per_class * nc]
        The indices of the training nodes
    split_val: array-like [n_per_class * nc]
        The indices of the validation nodes
    split_test array-like [n_nodes - 2*n_per_class * nc]
        The indices of the test nodes
    """
    np.random.seed(seed)
    nc = labels.max() + 1
    split_train, split_val = [], []
    for l in range(nc):
        np.random.seed(seed+l)
        perm = np.random.RandomState(seed=seed).permutation((labels == l).nonzero()[0])
        split_train.append(perm[:n_per_class])
        split_val.append(perm[n_per_class:2 * n_per_class])


    split_train = np.random.RandomState(seed=seed).permutation(np.concatenate(split_train))
    split_val = np.random.RandomState(seed=seed).permutation(np.concatenate(split_val))

    assert split_train.shape[0] == split_val.shape[0] == n_per_class * nc

    split_test = np.setdiff1d(np.arange(len(labels)), np.concatenate((split_train, split_val)))
    print("Number of samples per class:", n_per_class)
    print("Training-validation-testing Size:", len(split_train),len(split_val),len(split_test))
    return split_train, split_val, split_test

def normalize(adj):
    degree = torch.sum(adj,dim=0)
    D_half_norm = torch.pow(degree, -0.5)
    D_half_norm = torch.nan_to_num(D_half_norm, nan=0.0, posinf=0.0, neginf=0.0)
    D_half_norm = torch.diag(D_half_norm)
    DAD = torch.mm(torch.mm(D_half_norm,adj), D_half_norm)
    return DAD


def listSubset(A_list,index_list):
    '''take out the elements of a list (A_list) by a index list (index_list)'''
    return [A_list[i] for i in index_list]

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


