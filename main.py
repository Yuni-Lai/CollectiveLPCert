# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:38:45 2023

@author: MSI-NB
"""
import argparse
import warnings
import os
import pickle
import pandas as pd
from tqdm import trange
import numpy as np
import time
import pprint
import random
from statsmodels.stats.proportion import proportion_confint
from utils import load_data,split,init_random_seed
from optimize import *

#-------------Config----------------------------
parser = argparse.ArgumentParser(description='collective certify GNN node injecttion')
parser.add_argument('-seed', type=int, default=2021)
parser.add_argument('-n_per_class', type=int, default=50, help='sample numebr per class')
parser.add_argument('-model', type=str, default='GCN',choices=['GCN', 'GAT'], help='GNN model')
# parser.add_argument('-n_hidden', type=int, default=64, help='size of hidden layer')
# parser.add_argument('-drop', type=float, default=0.5, help='dropout rate')
parser.add_argument('-certify_mode', type=str, default='evasion',
                    choices=['evasion', 'poisoning'], help="evasion only for gray box collective certify")
parser.add_argument('-conf_alpha', type=float, default=0.01, help='confident alpha for statistic testing')
parser.add_argument('-p_e', type=float, default=0.9, help='probability of deleting edges')
parser.add_argument('-p_n', type=float, default=0.8, help='probability of deleting nodes')
parser.add_argument('-n_smoothing', type=int, default=100000, help='number of smoothing samples evalute')
parser.add_argument('-tau', type=int, default=6, help='number of injection edges')
parser.add_argument('-optimization', type=str, default='LP2',
                    choices=['LP1','LP2'],help="Collective-LP1 or LP2")
parser.add_argument('-rho_range', type=list, default=[20,50,80,100,120,140,160], help='number of injection nodes (rho)')
parser.add_argument('-small_rho', action='store_true', default=False, help='test with small rho')
parser.add_argument('-test_subset', action='store_true', default=True, help='verify sampled subset of nodes')
parser.add_argument('-test_num', type=int, default=100, help='number of testing nodes')
parser.add_argument('-force_cert', type=bool, default=True, help='force computing the certificate')
parser.add_argument('-dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'])
parser.add_argument('-output_dir', type=str, help='model outputs directory')

args = parser.parse_args()
init_random_seed(args.seed)

if args.test_subset==False:
    args.test_num="All"

# Smoothing samples config
sample_config = {'p_e': args.p_e, 'p_n': args.p_n}
args.input_dir = f'./results_{args.dataset}_{args.model}/{args.certify_mode}_mode_include/{sample_config["p_e"]}_{sample_config["p_n"]}_{args.n_smoothing}/'

# Dataset
if args.dataset == "cora":
    args.data_dir = "./Data/cora_ml.npz"
    # args.tau = 6
elif args.dataset == "citeseer":
    args.data_dir = "./Data/citeseer.npz"
    # args.tau = 4
elif args.dataset == "pubmed":
    args.data_dir = "./Data/pubmed.npz"
    # args.tau = 5

if args.small_rho:
    args.rho_range=[2,4,6,8,10,12]
    args.output_dir = f'./results_{args.dataset}_{args.model}/{args.certify_mode}_mode_include/{sample_config["p_e"]}_{sample_config["p_n"]}_{args.n_smoothing}/{args.optimization}_{args.test_num}_smallrho/'
else:
    args.output_dir = f'./results_{args.dataset}_{args.model}/{args.certify_mode}_mode_include/{sample_config["p_e"]}_{sample_config["p_n"]}_{args.n_smoothing}/{args.optimization}_{args.test_num}/'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

pprint.pprint(vars(args), width=1)
#-----------------------------------------

''' Load Smoothing result '''
f1=open(f'{args.input_dir}smoothing_result.pkl','rb')
top2, count1, count2=pickle.load(f1) # the top-2 prediction counts of randomied smoothing
mat_pABar = np.zeros((top2.shape[0],1))
mat_pBBar = np.zeros((top2.shape[0],1))
# calculate the confidences
for i in range(top2.shape[0]):
    mat_pABar[i,0] = proportion_confint(count1[i], args.n_smoothing, alpha=2 * args.conf_alpha/7, method="beta")[0]
    mat_pBBar[i,0] = proportion_confint(count2[i], args.n_smoothing, alpha=2 * args.conf_alpha/7, method="beta")[1]
f1.close()
print('shape of pa,pb,top2',mat_pABar.shape,mat_pBBar.shape,top2.shape)
print('\n')

'''Load dataset'''
adj_origin, features, labels, n, d, nc = load_data(args.data_dir)
# fix the dataset split, make sure that the split is the same
idx_train, idx_val, idx_test = split(labels=labels, n_per_class=args.n_per_class, seed=2020)
adj = adj_origin.toarray()
# preprocessing the adj: remove self loop and check symmetry
for i in range(adj.shape[0]):
    adj[i][i] = 0
assert (adj ==  adj.T).all()

if args.test_subset: # Sample testing nodes that correctly classified
    random.seed(args.seed)
    index_ = np.where(np.array(top2[idx_test,0]) == labels[idx_test])[0]
    index_correct = list(idx_test[index_])
    index_include = random.sample(index_correct, args.test_num)
    test_label = labels[index_include]
    correct_num = (np.array(top2[index_include,0]) == test_label).sum()
else: #verify all testing nodes that correctly classified
    test_label = labels[idx_test]
    index_include = list(np.where(np.array(top2[idx_test, 0]) == test_label)[0])
    correct_num = (np.array(top2[idx_test, 0]) == test_label).sum()
    args.test_num = len(idx_test)

correct_ratio = correct_num / len(test_label)
print('test node correct num, ratio:',correct_num, ', ',correct_ratio,'\n')

'''Collective certificates '''
rho_list = args.rho_range
certi_acc_list = [correct_ratio]
opt_M_list=[0.0]
time_list=[0.0]
for rho in rho_list:
    print('### The current rho is:',rho)
    if certi_acc_list[-1]==0.0:
        certi_acc_list.append(0.0)
        opt_M_list.append(args.test_num)
        time_list.append('NA')
        continue
    if not os.path.exists(f'{args.output_dir}/optimization_result_tau{args.tau}_rho{rho}_{args.seed}.pkl') or args.force_cert:
        if args.optimization == "LP1":
            M, A1, A2, dep_time, opt_time = LP1(rho, args.tau, mat_pABar, mat_pBBar, index_include, adj, args)
        elif args.optimization == "LP2":
            M, A1, A2, dep_time, opt_time = LP2(rho, args.tau, mat_pABar, mat_pBBar, index_include, adj, args)
        elif args.optimization == "BLP_cvxpy":
            M, A1, A2, dep_time, opt_time = BLP_cvxpy(rho, args.tau, mat_pABar, mat_pBBar, index_include, adj, args)
        if args.optimization == "QP_gurobi":
            M, A1, A2, dep_time, opt_time = QP_gurobi(rho, args.tau, mat_pABar, mat_pBBar, index_include, adj, args)
        elif args.optimization == "LP_gurobi":
            M, A1, A2, dep_time, opt_time = LP_gurobi(rho, args.tau, mat_pABar, mat_pBBar, index_include, adj, args)
        elif args.optimization == "LP_gurobi2":
            M, A1, A2, dep_time, opt_time = LP_gurobi2(rho, args.tau, mat_pABar, mat_pBBar, index_include, adj, args)

        f = open(f'{args.output_dir}/optimization_result_tau{args.tau}_rho{rho}_{args.seed}.pkl', 'wb')
        pickle.dump([M, A1, A2, dep_time, opt_time], f)
        f.close()
    else:
        f = open(f'{args.output_dir}/optimization_result_tau{args.tau}_rho{rho}_{args.seed}.pkl', 'rb')
        M, A1, A2, dep_time, opt_time = pickle.load(f)
        f.close()

    certi_acc_list.append((correct_num - np.ceil(M)) / args.test_num)
    opt_M_list.append(M)
    time_list.append(dep_time+opt_time)
    print("Optimal M:",M)
    print("\nThe certified nodes number:", (correct_num - np.ceil(M)))

'''Record the results'''
df = {'rho': [0]+rho_list, 'certified_ratio': certi_acc_list, 'opt_M': opt_M_list,'time': time_list}
f = open(f'{args.output_dir}/collective_certify_result_tau{args.tau}_{args.seed}.pkl', 'wb')
pickle.dump(df, f)
f.close()

print(f'Save result to {args.output_dir}/collective_certify_result_tau{args.tau}.pkl')
print(f'{args.dataset}_{args.optimization}_{sample_config["p_e"]}_{sample_config["p_n"]}_{args.n_smoothing}_{args.seed}')
print(pd.DataFrame(df).to_string())

if os.path.exists(f'./results_{args.dataset}_{args.model}/result_summary.txt') == False:
    with open(f'./results_{args.dataset}_{args.model}/result_summary.txt', 'w') as f:
        f.write('result record\n')
        f.write(f'{args.dataset}_{args.optimization}_{sample_config["p_e"]}_{sample_config["p_n"]}_{args.n_smoothing}_{args.seed}\n')
        f.write(pd.DataFrame(df).to_string())
        f.write('\n')
else:
    with open(f'./results_{args.dataset}_{args.model}/result_summary.txt', 'a') as f:
        f.write(f'{args.dataset}_{args.optimization}_{sample_config["p_e"]}_{sample_config["p_n"]}_{args.n_smoothing}_{args.seed}\n')
        f.write(pd.DataFrame(df).to_string())
        f.write('\n')

print("program completes")