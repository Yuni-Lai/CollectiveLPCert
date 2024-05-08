import gurobipy as gp
from gurobipy import GRB
import os
import pickle
from tqdm import trange
import numpy as np
import time
import cvxpy as cp
import pandas as pd
import scipy.sparse as sp
from statsmodels.stats.proportion import proportion_confint



def LP1(rho, tau, mat_pABar, mat_pBBar, index_include, adj, args):
    ### parameter setting
    n = mat_pABar.shape[0]
    n1 = len(index_include)
    p_e = args.p_e
    p_n = args.p_n
    pe_bar = 1 - p_e
    pn_bar = 1 - p_n
    log_term = np.log(1 - (mat_pABar - mat_pBBar) / 2)
    p1 = (np.log(1 - pe_bar * pn_bar))
    p2 = (np.log(1 - (pe_bar * pn_bar) ** 2))
    constraints = []

    start_time = time.time()
    ### creating model
    m = cp.Variable(shape=(n, 1), name='m')
    A = cp.Variable(shape=(rho, n), name='A')  # A is A_1
    # B = cp.Variable(shape=(rho, rho), name='B') #B is A_2
    B = cp.Variable(shape=(rho, rho), name='B', symmetric=True)

    # all-ones vectors
    e_rho = np.ones((rho, 1))
    e_n = np.ones((n, 1))
    e_index = np.zeros(n)
    # e_index = np.full((n, ), False)
    for v in index_include:
        e_index[v] = 1

    objective = cp.Maximize(e_index.T @ m)
    e_index = (e_index == 1)
    # add constraints
    constraints.append(m >= 0)
    constraints.append(m <= 1)
    constraints.append(A >= 0)
    constraints.append(A <= 1)
    constraints.append(B >= 0)
    constraints.append(B <= 1)
    # constraints.append(B.T == B)
    constraints.append(A @ e_n + B @ e_rho <= tau)
    # constraints.append(A @ e_n + B @ e_rho == tau)
    # A_squre_(n+rho):,1:n: = A @adj + B@A
    # add auxillary variables Q to linearise the products in B@A
    # Q_v are equivalent to the elements: B_ij * A_jv, for i, j in 1,..., rho
    for v in index_include:
        globals()[f'Q_{v}'] = cp.Variable(shape=(rho, rho), name=f'Q_{v}')
        constraints.append(eval(f'Q_{v}') <= e_rho @ cp.reshape(A[:, v], (1, rho)))
        constraints.append(eval(f'Q_{v}') <= B)
        constraints.append(e_rho @ cp.reshape(A[:, v], (1, rho)) + B - eval(f'Q_{v}') <= 1)
        constraints.append(eval(f'Q_{v}') >= 0)
        constraints.append(eval(f'Q_{v}') <= 1)

        # constraints.append(log_term[v]*m[v] - p1 * A[:,v].T@e_rho - p2 * adj[v,:] @ A.T @ e_rho - p2 * e_rho.T @ eval(f'Q_{v}') @ e_rho >= 0)

    constraints.append((cp.multiply(log_term, m) - p1 * A.T @ e_rho - p2 * adj.T @ A.T @ e_rho)[e_index] - cp.vstack(
        [p2 * e_rho.T @ eval(f'Q_{v}') @ e_rho for v in index_include]) >= 0)

    end_time = time.time()
    deployment_time = end_time - start_time
    print('Problem deployment time:', deployment_time)

    # Define and solve the CVXPY problem.
    prob = cp.Problem(objective, constraints)

    start_time = time.time()
    prob.solve(solver=cp.MOSEK, verbose=True)
    # save_file = args.output_dir+f'/CVX_problem_tau{tau}_rho{rho}.mps'
    print("status:", prob.status)

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution A,B is")
    print(A.value, B.value)
    print("A.max():", A.value.max())
    print("(A==1).sum():", (A.value == 1).sum())

    end_time = time.time()
    optimization_time = end_time - start_time
    print('Problem optimization time:', optimization_time)
    return prob.value, A.value, B.value, deployment_time, optimization_time



def LP2(rho, tau, mat_pABar, mat_pBBar, index_include, adj, args):
    # reduce the complexity to o(|T|*rho).
    # substitute A2@e_rho by vector variable Z.
    ### parameter setting
    n = mat_pABar.shape[0]
    t = len(index_include)
    p_e = args.p_e
    p_n = args.p_n
    pe_bar = 1 - p_e
    pn_bar = 1 - p_n
    log_term = np.log(1 - (mat_pABar - mat_pBBar) / 2)
    p1 = (np.log(1 - pe_bar * pn_bar))
    p2 = (np.log(1 - (pe_bar * pn_bar) ** 2))
    constraints = []

    start_time = time.time()
    ### creating model
    m = cp.Variable(shape=(n, 1), name='m')
    A = cp.Variable(shape=(rho, n), name='A')  # A is A_1
    Z = cp.Variable(shape=(rho, 1), name='Z') # Z=A2@e_rho

    # all-ones vectors
    e_rho = np.ones((rho, 1))
    e_n = np.ones((n, 1))
    e_index = np.zeros(n)
    # e_index = np.full((n, ), False)
    for v in index_include:
        e_index[v] = 1

    objective = cp.Maximize(e_index.T @ m)
    e_index = (e_index == 1)
    # add constraints
    constraints.append(m >= 0)
    constraints.append(m <= 1)
    constraints.append(A >= 0)
    constraints.append(A <= 1)
    constraints.append(Z >= 0)
    constraints.append(Z <= min(tau, rho))
    constraints.append(A @ e_n + Z <= tau)
    # constraints.append(A @ e_n + Z == tau)
    # Q are equivalent to the elements: A_ij z_j, for i in 1,..., n; j in 1,..., rho
    Q= cp.Variable(shape=(n, rho), name='Q')
    constraints.append(Q<= min(tau, rho) * A.T)
    constraints.append(Q <= e_n @ Z.T)
    constraints.append(min(tau, rho)*A.T + e_n @ Z.T - Q <= min(tau, rho))
    constraints.append(Q >= 0)
    constraints.append(Q <= min(tau, rho))

    constraints.append((cp.multiply(log_term, m) - p1 * A.T @ e_rho - p2 * adj.T @ A.T @ e_rho - p2 * Q@e_rho)[e_index] >= 0)

    end_time = time.time()
    deployment_time = end_time - start_time
    print('Problem deployment time:', deployment_time)

    # Define and solve the CVXPY problem.
    prob = cp.Problem(objective, constraints)

    start_time = time.time()
    prob.solve(solver=cp.MOSEK, verbose=True)
    # save_file = args.output_dir+f'/CVX_problem_tau{tau}_rho{rho}.mps'
    print("status:", prob.status)

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution A,Z is")
    print(A.value, Z.value)
    print("A.max():", A.value.max())
    print("(A==1).sum():", (A.value == 1).sum())

    end_time = time.time()
    optimization_time = end_time - start_time
    print('Problem optimization time:', optimization_time)
    return prob.value, A.value, Z.value, deployment_time, optimization_time


def LP_gurobi(rho, tau, mat_pABar, mat_pBBar, index_include, adj, args):
    ## LP programming using gurobi.
    # It has exactly the same result as LP_cvxpy. But it is much slower in deployment.
    start_time = time.time()
    # parameter setting
    n = mat_pABar.shape[0]
    mat_pABar = mat_pABar.reshape(-1)
    mat_pBBar = mat_pBBar.reshape(-1)
    p_e = args.p_e
    p_n = args.p_n
    pe_bar = 1 - p_e
    pn_bar = 1 - p_n
    p1 = (np.log(1 - pe_bar * pn_bar))
    p2 = (np.log(1 - (pe_bar * pn_bar) ** 2))
    log_term = np.log(1 - (mat_pABar - mat_pBBar) / 2).tolist()

    # creating model
    options = {
        "WLSACCESSID": "56ce9c35-f970-4979-8f4a-49e846c9c2a4",
        "WLSSECRET": "9c3b8586-14f6-487f-b54f-01710c59f067",
        "LICENSEID": 2422867,
    }
    env = gp.Env(params=options)
    model = gp.Model(env=env)

    # creating variables
    m = model.addMVar((n), vtype=GRB.CONTINUOUS, lb=0, ub=1, name='m')
    M = model.addVar()
    A = model.addMVar((rho, n), vtype=GRB.CONTINUOUS, lb=0, ub=1, name='A')
    B = model.addMVar((rho, rho), vtype=GRB.CONTINUOUS, lb=0, ub=1, name='B')

    # define objective function
    print("\nsetting up objective function\n")
    model.setObjective(M, gp.GRB.MAXIMIZE)

    # A_squre_(n+rho):,1:n: = A @adj + B@A
    # add auxillary variables Q to linearise the products in B@A
    # Q are the elements: B_ij * A_jk
    Q = dict()
    for i in range(rho):
        for j in range(rho):
            ij = f'{i},{j}'
            for v in index_include:
                jv = f'{j},{v}'
                Q[f'{ij},{jv}'] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'A_{ij}-B_{jv}')

    # update environment settings
    model.update()

    print('settinng up linearizing constraints\n')
    # for i in range(rho):
    #     for j in range(rho):
    #         ij=f'{i},{j}'
    #         for k in index_include:
    #             jk=f'{j},{k}'
    #             model.addConstr(Q[f'{ij},{jk}'] <= B[i][j])
    #             model.addConstr(Q[f'{ij},{jk}'] <= A[j][k])
    #             model.addConstr(A[j][k]+B[i][j]-Q[f'{ij},{jk}']<=1)

    model.addConstrs(Q[keys] <= B[int(keys.split(',')[0])][int(keys.split(',')[1])] for keys in list(Q.keys()))
    model.addConstrs(Q[keys] <= A[int(keys.split(',')[2])][int(keys.split(',')[3])] for keys in list(Q.keys()))
    model.addConstrs(A[int(keys.split(',')[2])][int(keys.split(',')[3])]
                     + B[int(keys.split(',')[0])][int(keys.split(',')[1])]
                     - Q[keys] <= 1 for keys in list(Q.keys()))

    # objective function:L1 norm
    model.addConstr(M == gp.quicksum(m[v] for v in index_include))

    print('settinng up constraint 1\n')
    for v in index_include:
        # A_square_1 == (A1 @ adj + A2 @ A1)
        # matrix multiplication of  A2 @ A1 by elements
        # ！！！A1 @ adj ？？？
        model.addConstr(log_term[v] * m[v] - (A[:, v].sum() * p1 + (adj[v, :] @ A.T).sum() * p2 +
                                              gp.quicksum(
                                                  gp.quicksum(Q[f'{i},{j},{j},{v}'] for j in range(rho)) for i in
                                                  range(rho)) * p2)
                        >= 0)

    print('settinng up constraint 2\n')
    for i in range(rho):
        model.addConstr(A[i, :].sum() + B[i, :].sum() <= tau)

    print('setting up synmetric constraints 3')
    model.addConstr(B == B.T)

    end_time = time.time()
    deployment_time = end_time - start_time
    print('Problem deployment time:', deployment_time)

    # optimizing----------
    start_time = time.time()
    print('optimizing the M')
    model.optimize()
    print("A solution A,B is")
    print(A.x, B.x)
    print("A.x.max():", A.x.max())
    print("(A.x==1).sum():", (A.x == 1).sum())
    end_time = time.time()
    optimization_time = end_time - start_time
    print('Problem optimization time:', optimization_time)
    return M.x, A.x, B.x, deployment_time, optimization_time



def BLP_cvxpy(rho, tau, mat_pABar, mat_pBBar, index_include, adj, args):
    ### parameter setting
    n = mat_pABar.shape[0]
    n1 = len(index_include)
    p_e = args.p_e
    p_n = args.p_n
    pe_bar = 1 - p_e
    pn_bar = 1 - p_n
    log_term = np.log(1 - (mat_pABar - mat_pBBar) / 2)
    p1 = (np.log(1 - pe_bar * pn_bar))
    p2 = (np.log(1 - (pe_bar * pn_bar) ** 2))
    constraints = []

    start_time = time.time()
    ### creating model
    m = cp.Variable(shape=(n, 1), name='m', boolean=True)
    A = cp.Variable(shape=(rho, n), name='A', boolean=True)
    B = cp.Variable(shape=(rho, rho), name='B', boolean=True)
    # B = cp.Variable(shape=(rho, rho), name='B', boolean=True, symmetric=True)

    # all-ones vectors
    e_rho = np.ones((rho, 1))
    e_n = np.ones((n, 1))
    e_index = np.zeros(n)
    # e_index = np.full((n, ), False)
    for v in index_include:
        e_index[v] = 1
    objective = cp.Maximize(e_index.T @ m)
    e_index = (e_index == 1)
    # add constraints
    constraints.append(B.T == B)
    constraints.append(A @ e_n + B @ e_rho <= tau)
    # A_squre_(n+rho):,1:n: = A @adj + B@A
    # add auxillary variables Q to linearise the products in B@A
    # Q_v are equivalent to the elements: B_ij * A_jv, for i, j in 1,..., rho
    for v in index_include:
        globals()[f'Q_{v}'] = cp.Variable(shape=(rho, rho), name=f'Q_{v}', boolean=True)
        constraints.append(eval(f'Q_{v}') <= e_rho @ cp.reshape(A[:, v], (1, rho)))
        constraints.append(eval(f'Q_{v}') <= B)
        constraints.append(e_rho @ cp.reshape(A[:, v], (1, rho)) + B - eval(f'Q_{v}') <= 1)

    constraints.append((cp.multiply(log_term, m) - p1 * A.T @ e_rho - p2 * adj.T @ A.T @ e_rho)[e_index] - cp.vstack(
        [p2 * e_rho.T @ eval(f'Q_{v}') @ e_rho for v in index_include]) >= 0)

    end_time = time.time()
    deployment_time = end_time - start_time
    print('Problem deployment time:', deployment_time)

    # Define and solve the CVXPY problem.
    prob = cp.Problem(objective, constraints)

    start_time = time.time()
    prob.solve(
        solver=cp.MOSEK,
        verbose=True,
        save_file=args.output_dir + f'/CVX_problem_tau{tau}_rho{rho}.mps'
    )
    print("status:", prob.status)
    print("objective:", prob.value)

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution A,B is")
    print(A.value, B.value)
    print("A.max():", A.value.max())
    print("(A==1).sum():", (A.value == 1).sum())

    end_time = time.time()
    optimization_time = end_time - start_time
    print('Problem optimization time:', optimization_time)
    return prob.value, A.value, B.value, deployment_time, optimization_time


def QP_gurobi(rho, tau, mat_pABar, mat_pBBar, index_include, adj, args):
    start_time = time.time()
    n = mat_pABar.shape[0]

    # creating model
    options = {
        "WLSACCESSID": "56ce9c35-f970-4979-8f4a-49e846c9c2a4",
        "WLSSECRET": "9c3b8586-14f6-487f-b54f-01710c59f067",
        "LICENSEID": 2422867,
    }
    env = gp.Env(params=options)
    model = gp.Model(env=env)
    model.params.NonConvex = 2

    # creating variables
    m = model.addMVar((n), vtype=GRB.CONTINUOUS, lb=0, ub=1, name='m')
    A1 = model.addMVar((rho, n), vtype=GRB.CONTINUOUS, lb=0, ub=1, name='A1')
    A2 = model.addMVar((rho, rho), vtype=GRB.CONTINUOUS, lb=0, ub=1, name='A2')
    A_square_1 = model.addMVar((rho, n), vtype=GRB.CONTINUOUS, lb=0, ub=1, name='AS')
    M = model.addVar()

    # parameter setting

    mat_pABar = mat_pABar.reshape(-1)
    mat_pBBar = mat_pBBar.reshape(-1)
    log_term = np.log(1 - (mat_pABar - mat_pBBar) / 2).tolist()
    p_e = args.p_e
    p_n = args.p_n
    pe_bar = 1 - p_e
    pn_bar = 1 - p_n
    p1 = (np.log(1 - pe_bar * pn_bar))
    p2 = (np.log(1 - (pe_bar * pn_bar) ** 2))

    # update environment settings
    model.update()

    # define objective function
    print("\nsetting up objective function\n")
    model.setObjective(M, gp.GRB.MAXIMIZE)

    # define constraints

    # objective function:L1 norm
    model.addConstr(M == gp.quicksum(m[v] for v in index_include))

    print('settinng up constraint 1\n')
    model.addConstr(A_square_1 == (A1 @ adj + A2 @ A1))

    print('settinng up constraint 2\n')
    for v in index_include:
        model.addConstr(log_term[v] * m[v] - (A1[:, v].sum() * p1 + A_square_1[:, v].sum() * p2) >= 0)

    print('settinng up constraint 3\n')
    for i in range(rho):
        model.addConstr(A1[i, :].sum() + A2[i, :].sum() <= tau)

    print('setting up synmetric constraints 4')
    model.addConstr(A2 == A2.T)
    end_time = time.time()
    deployment_time = end_time - start_time
    print('Problem deployment time:', deployment_time)

    # optimizing-------------
    start_time = time.time()
    print('optimizing the M')
    model.optimize()
    print("A1.x.max():", A1.x.max())
    print("(A1.x==1).sum():", (A1.x == 1).sum())
    end_time = time.time()
    optimization_time = end_time - start_time
    print('Problem optimization time:', optimization_time)
    return M.x, A1.x, A2.x, deployment_time, optimization_time


def LP_gurobi2(rho, tau, mat_pABar, mat_pBBar, index_include, adj, args):
    # define LP as min c'x, s.t. Ax=b (Ax<=b), l<=x<=u.
    # It is much faster than LP_gurobi. However, it has some unknown error with different results.
    # vectorize the matrix A_1, A_2, Q column-first to the variable x
    # x = [A_1(11), A_1(21), ... ,A_1(rho,n), A_2(11), A_2(21), ... ,A_2(rho,rho), Q_1(11), Q_1(21), ... ,Q_1(rho,rho), ... ,Q_t(11), Q_t(21), ... ,Q_t(rho,rho), m(1), m(2), ... ,m(t)]

    # A is a matrix of size (nConstraints) * (n*rho + rho^2 + t*rho^2 + t)
    # constraint 1: t in total
    # constraint 2: rho in total
    # constraint 3: t*rho^2 in total
    # constraint 4: t*rho^2 in total
    # constraint 5: t*rho^2 in total
    # constraint 6: rho*(rho-1)/2 in total
    # nConstraints = t + rho + 3 * t * rho ** 2 + rho * (rho - 1) / 2

    # define the objective function c = t'm
    start_time = time.time()
    t = len(index_include)
    n = mat_pABar.shape[0]
    nCols = n * rho + rho ** 2 + t * rho ** 2 + t

    p_e = args.p_e
    p_n = args.p_n
    pe_bar = 1 - p_e
    pn_bar = 1 - p_n
    log_term = np.log(1 - (mat_pABar - mat_pBBar) / 2)
    # reshape log_term to a vector
    log_term = log_term.reshape(-1)

    p1 = (np.log(1 - pe_bar * pn_bar))
    p2 = (np.log(1 - (pe_bar * pn_bar) ** 2))

    # transform adj to a sparse matrix
    adj = sp.csr_matrix(adj, dtype=np.float32)

    # get the rowStarts and colIndices of the sparse matrix
    rowStarts = adj.indptr
    colIndices = adj.indices

    nConstraints = t + rho + 3 * t * rho ** 2 + rho * (rho - 1) / 2
    # define the constraint 1 by triples (row, col, val)

    for i in range(t):
        rowStart = rowStarts[index_include[i]]
        rowEnd = rowStarts[index_include[i] + 1]
        row = np.ones(rho * (rowEnd - rowStart + 1) + rho ** 2 + 1, dtype=int) * 0
        col = np.arange(index_include[i]*rho, index_include[i]*rho+rho, dtype=int)
        val = np.ones(rho) * p1
        for j in range(rowStart, rowEnd):
            col = np.concatenate((col, np.arange(colIndices[j]*rho, colIndices[j]*rho+rho)), axis=0)
            val = np.concatenate((val, np.ones(rho) * p2), axis=0)
        col = np.concatenate((col, np.arange(n*rho+rho**2+i*rho**2, n*rho+rho**2+(i+1)*rho**2)), axis=0)
        val = np.concatenate((val, np.ones(rho**2) * p2), axis=0)
        col = np.concatenate((col, [n*rho+rho**2+t*rho**2+i]), axis=0)
        val = np.concatenate((val, [-log_term[index_include[i]]]), axis=0)

        # create sparse matrix by row, col, val
        if i == 0:
            constraint_matrix = sp.csr_matrix((val, (row, col)), shape=(1, nCols), dtype=np.float32)
        else:
            constraint_matrix = sp.vstack([constraint_matrix, sp.csr_matrix((val, (row, col)), shape=(1, nCols), dtype=np.float32)])

    # constraint 2
    for i in range(rho):
        row = np.ones(n + rho, dtype=int) * 0
        col = np.arange(i, n*rho, rho, dtype=int)
        col = np.concatenate((col, np.arange(rho*i, rho*i+rho, dtype=int)), axis=0)
        val = np.ones(n + rho)
        # create sparse matrix by row, col, val
        constraint_matrix = sp.vstack([constraint_matrix, sp.csr_matrix((val, (row, col)), shape=(1, nCols), dtype=np.float32)])

    # constraint 3
    for i in range(t):
        row = np.arange(0, rho**2, dtype=int)
        col = np.arange(n * rho + rho ** 2 + i*rho**2, n * rho + rho ** 2 + (i+1)*rho**2, dtype=int)
        val = np.ones(rho**2)
        # can be optimized
        row = np.concatenate((row, np.arange(0, rho**2, dtype=int)), axis=0)
        for j in range(rho):
            col = np.concatenate((col, np.ones(rho)*rho*index_include[i]+j), axis=0)
        val = np.concatenate((val, np.ones(rho**2)*(-1)), axis=0)
        constraint_matrix = sp.vstack(
            [constraint_matrix, sp.csr_matrix((val, (row, col)), shape=(rho**2, nCols), dtype=np.float32)])

    # constraint 4
    # col = np.arange(n * rho + rho ** 2 + i * rho ** 2, n * rho + rho ** 2 + (i + 1) * rho ** 2, dtype=int)
    for i in range(t):
        row = np.arange(0, rho**2, dtype=int)
        col = np.arange(n * rho + rho ** 2 + i*rho**2, n * rho + rho ** 2 + (i+1)*rho**2, dtype=int)
        val = np.ones(rho**2)
        # can be optimized
        row = np.concatenate((row, np.arange(0, rho**2, dtype=int)), axis=0)
        col = np.concatenate((col, np.arange(n*rho, n*rho+rho**2, dtype=int)), axis=0)
        val = np.concatenate((val, np.ones(rho**2)*(-1)), axis=0)
        constraint_matrix = sp.vstack(
            [constraint_matrix, sp.csr_matrix((val, (row, col)), shape=(rho**2, nCols), dtype=np.float32)])

    # constraint 5
    for i in range(t):
        row = np.arange(0, rho**2, dtype=int)
        col = np.arange(n * rho + rho ** 2 + i*rho**2, n * rho + rho ** 2 + (i+1)*rho**2, dtype=int)
        val = np.ones(rho**2)*(-1)
        # can be optimized
        row = np.concatenate((row, np.arange(0, rho ** 2, dtype=int)), axis=0)
        for j in range(rho):
            col = np.concatenate((col, np.ones(rho) * rho * index_include[i] + j), axis=0)
        val = np.concatenate((val, np.ones(rho ** 2)), axis=0)
        row = np.concatenate((row, np.arange(0, rho ** 2, dtype=int)), axis=0)
        col = np.concatenate((col, np.arange(n * rho, n * rho + rho ** 2, dtype=int)), axis=0)
        val = np.concatenate((val, np.ones(rho ** 2)), axis=0)
        constraint_matrix = sp.vstack(
            [constraint_matrix, sp.csr_matrix((val, (row, col)), shape=(rho**2, nCols), dtype=np.float32)])


    # define the rhs vector b
    b = np.zeros(t)
    b = np.concatenate((b, np.ones(rho)*tau), axis=0)
    b = np.concatenate((b, np.zeros(t*rho**2)), axis=0)
    b = np.concatenate((b, np.zeros(t*rho**2)), axis=0)
    b = np.concatenate((b, np.ones(t*rho**2)), axis=0)

    # define the obj vector c
    c = np.zeros(n*rho+rho**2+t*rho**2)
    c = np.concatenate((c, np.ones(t)), axis=0)

    # define the sense vector, all constraints are <=
    # sense = ['<'] * int(nConstraints - num)
    # sense = np.concatenate((sense, ['='] * num), axis=0)

    # define the lower bound vector
    lb = np.zeros(nCols)
    # define the upper bound vector
    ub = np.ones(nCols)

    # define the variable type vector
    vtype = ['C'] * nCols

    # creating model
    options = {
        "WLSACCESSID": "56ce9c35-f970-4979-8f4a-49e846c9c2a4",
        "WLSSECRET": "9c3b8586-14f6-487f-b54f-01710c59f067",
        "LICENSEID": 2422867,
    }
    env = gp.Env(params=options)
    # model = gp.Model(env=env)
    # apply the gurobi solver, use the above A b c sense lb ub vtype
    model = gp.Model('LP',env=env)
    # model.Params.OutputFlag = 0
    x = model.addMVar(shape=nCols, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name='x')
    model.setObjective(c @ x, GRB.MAXIMIZE)
    model.addConstr(constraint_matrix @ x <= b, name='constraint')

    # constraint 6
    num = int(rho * (rho - 1) / 2)
    row = np.arange(0, num, dtype=int)
    col = np.arange(n*rho+1, n*rho+rho, dtype=int)
    for i in range(1, rho-1):
        col = np.concatenate((col, np.arange(n*rho+i*rho+i+1, n*rho+i*rho+rho)), axis=0)
    val = np.ones(num)

    row = np.concatenate((row, np.arange(0, num, dtype=int)), axis=0)
    for i in range(rho-1):
        col = np.concatenate((col, np.arange(n*rho+(i+1)*rho+i, n*rho+rho**2, rho)), axis=0)
    val = np.concatenate((val, np.ones(num)*(-1)), axis=0)
    constraint_matrix6 = sp.csr_matrix((val, (row, col)), shape=(num, nCols), dtype=np.float32)
    b = np.zeros(int(rho * (rho - 1) / 2))

    # gurobi add constraint 6 as equality
    model.addConstr(constraint_matrix6 @ x == b, name='constraint6')

    end_time = time.time()
    deployment_time = end_time - start_time
    print('Problem deployment time:', deployment_time)

    model.optimize()
    print('Obj: %g' % model.objVal)

    end_time = time.time()
    optimization_time = end_time - start_time
    print('Problem optimization time:', optimization_time)

    #    # constraint 4o 2n
    return (c*x.x).sum(), x.x, None, deployment_time, optimization_time


def LP_ATA(rho, tau, mat_pABar, mat_pBBar, index_include, adj, args):
    # remove A2@e_rho by tau-A1@e_n.
    # it has even higher complexity: o(n*|T|*rho)
    ### parameter setting
    n = mat_pABar.shape[0]
    n1 = len(index_include)
    p_e = args.p_e
    p_n = args.p_n
    pe_bar = 1 - p_e
    pn_bar = 1 - p_n
    log_term = np.log(1 - (mat_pABar - mat_pBBar) / 2)
    p1 = (np.log(1 - pe_bar * pn_bar))
    p2 = (np.log(1 - (pe_bar * pn_bar) ** 2))
    constraints = []

    start_time = time.time()
    ### creating model
    m = cp.Variable(shape=(n, 1), name='m')
    A = cp.Variable(shape=(rho, n), name='A')  # A is A_1

    # all-ones vectors
    e_rho = np.ones((rho, 1))
    e_n = np.ones((n, 1))
    e_index = np.zeros(n)
    for v in index_include:
        e_index[v] = 1

    objective = cp.Maximize(e_index.T @ m)
    e_index = (e_index == 1)
    # add constraints
    constraints.append(m >= 0)
    constraints.append(m <= 1)
    constraints.append(A >= 0)
    constraints.append(A <= 1)
    constraints.append(A @ e_n <= tau)
    constraints.append(A @ e_n >= tau - rho)

    # add auxillary variables Q to linearise the products in A.T@A
    # Q_v are equivalent to the elements: A_ji * A_vj, for i in 1,..., n, for i in 1,...,rho
    for v in index_include:
        globals()[f'Q_{v}'] = cp.Variable(shape=(n, rho), name=f'Q_{v}')
        constraints.append(eval(f'Q_{v}') <= e_n @ cp.reshape(A[:, v], (1, rho)))
        constraints.append(eval(f'Q_{v}') <= A.T)
        constraints.append(e_n @ cp.reshape(A[:, v], (1, rho)) + A.T - eval(f'Q_{v}') <= 1)
        constraints.append(eval(f'Q_{v}') >= 0)
        constraints.append(eval(f'Q_{v}') <= 1)

    constraints.append(
        (cp.multiply(log_term, m) - p1 * A.T @ e_rho - p2 * adj.T @ A.T @ e_rho - p2 * tau * A.T @ e_rho)[
            e_index] + cp.vstack([p2 * e_n.T @ eval(f'Q_{v}') @ e_rho for v in index_include]) >= 0)

    end_time = time.time()
    deployment_time = end_time - start_time
    print('Problem deployment time:', deployment_time)

    # Define and solve the CVXPY problem.
    prob = cp.Problem(objective, constraints)

    start_time = time.time()
    prob.solve(solver=cp.MOSEK, verbose=True)
    print("status:", prob.status)

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution A,B is")
    print(A.value)
    print("A.max():", A.value.max())
    print("(A==1).sum():", (A.value == 1).sum())

    end_time = time.time()
    optimization_time = end_time - start_time
    print('Problem optimization time:', optimization_time)
    return prob.value, A.value, None, deployment_time, optimization_time

