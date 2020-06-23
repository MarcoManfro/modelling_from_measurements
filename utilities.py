import numpy as np
from scipy import io, integrate
import os
import sympy as sy
from sympy.parsing.sympy_parser import parse_expr
from sklearn import linear_model

from pygmo import *
# from exercise1 import fitness_fun


def DMD(X, X1, r):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Ur = np.matrix(U[:,:r])
    Sr = np.matrix(np.diag(S[:r]))
    Vtr = np.matrix(Vt[:r,:])
    
    Atilde = np.linalg.solve(Sr.H, (Ur.H @ X1 @ Vtr.H).H).H
    check = np.allclose(Atilde, (Ur.H @ X1 @ Vtr.H @ np.linalg.inv(Sr)))
    if check == False:
        print('Computation of A tilde is not correct')

    Lambda, W = np.linalg.eig(Atilde)
    Lambda = np.diag(Lambda)

    Phi = X1 @ np.linalg.solve(Sr.H,Vtr).H @ W
    alpha1 = Sr @ Vtr[:,0]
    b = np.linalg.solve(W @ Lambda,alpha1)

    return Phi, Lambda, b

def forecastDMD(Phi, Lambda, b, time, how = 'M'):
    dt = time[1] - time[0]
    
    if how == 'V':
        omega = np.log(np.reshape(np.diag(Lambda), (-1,1)))/dt

        for k in range(len(time)):
            if k == 0:
                u_modes = np.r_['c',(np.array(b)*np.exp(omega*time[k]))]
            else:
                u_modes = np.hstack((u_modes, (np.array(b)*np.exp(omega*time[k]))))

        Omega = np.diag(omega)
        u_dmd = np.array(Phi @ u_modes)

    elif how == 'M':
        Omega = np.log(Lambda)/dt

        for k in range(len(time)):
            try:
                K = Omega*time[k]
                if (np.isnan(K)).any():
                    raise Exception('WarningNaN')
            except Exception:
                K[np.isnan(K)] = (K[np.isnan(K)]).real
                K[np.isnan(K)] = -float('inf')
                K = np.matrix(K)
        
            if k == 0:
                u_dmd = np.r_['c',(Phi @ np.exp(K) @ b)]
            else:
                u_dmd = np.hstack((u_dmd, (Phi @ np.exp(K) @ b)))
        u_dmd = np.array(u_dmd)
    
    else:
        print('Wrong "how" input')

    return u_dmd, Omega

def matrixDelay (M, n_delay):
    M = np.matrix(M)
    n_rep = n_delay+1
    H = np.zeros((n_rep*M.shape[0], M.shape[1]-n_rep+1))

    for k in range(n_rep):
        H[k*M.shape[0]:M.shape[0]*(1+k),:] = M[:, k:M.shape[1]-n_rep+k+1]
    
    return H

def centralSchemeDerivative(X, dt, direction=1):

    if len(X.shape) == 1:
        X_t = np.matrix(X)
    else:
        X_t = X

    try: 
        if direction == 1:
            X_t = X_t
        elif direction == 0:
            X_t = X_t.conj().T
        else:
            raise Exception('InputError')
    except Exception as myErr:
        print(repr(myErr))
        print('"direction" input is not correct. It may be:')
        print(' - 0: to compute derivative along column-wise direction')
        print(' - 1: to compute derivative along row-wise direction')
        print('Default value (direction = 1) hhas been automatically set')
        X_t = X_t  

    x_add = np.zeros((X_t.shape[0],1))*float('nan')
    X_A = np.concatenate((X_t[:,1:],x_add),axis=1)
    X_B = np.concatenate((x_add, X_t[:,:-1]),axis=1)

    X_dot = (X_A-X_B)/(2*dt)
    X_dot = X_dot[:,1:-1]

    if direction == 0:
        X_dot = X_dot.conj().T
    if len(X.shape) == 1:
        X_dot = (np.array(X_dot))[0]


    return X_dot


def LSLF (equation, fit_par, ind_var, x_data_point, f_data_point):
    
    par_sym = [sy.Symbol(fit_par[k]) for k in range(len(fit_par))]
    var_sym = [sy.Symbol(ind_var[k]) for k in range(len(ind_var))]
    f = parse_expr(equation)
    f_true = sy.Symbol('f_true')

    def QE(f):
        return 1/2*(f-f_true)**2

    E = QE(f)
    dE = [sy.diff(E, par_sym[k]) for k in range(len(par_sym))]
    dE_ex = [sy.expand(dE[k]) for k in range(len(par_sym))]
    dE_ex_arg = [dE_ex[k].args for k in range(len(par_sym))]

    A_sym = sy.eye(len(par_sym))
    B_sym= sy.zeros(len(par_sym),1)
    for k in range(len(par_sym)):
        for i in range(len(dE_ex_arg[k])):
            comp = dE_ex_arg[k][i]
            kk = 0
            for l in range(len(par_sym)):
                if par_sym[l] in comp.free_symbols:
                    A_sym[k,l] = comp/par_sym[l]
                else:
                    kk = kk+1
            if kk == len(par_sym):
                B_sym[k,0] = -comp

    A = np.zeros(len(A_sym))
    for i in range(len(A_sym)):
        A_temp2 = 0
        for ii in range(len(x_data_point[0,:])):
            A_temp = A_sym[i]
            for k in range(len(var_sym)):
                A_temp = A_temp.subs(var_sym[k],x_data_point[k,ii])
            A_temp2 = A_temp2+A_temp
        A[i] = np.float64(A_temp2.evalf())
    
    A = np.reshape(A, A_sym.shape)
            
    B = np.zeros(len(B_sym))
    for i in range(len(B_sym)):
        B_temp2 = 0
        for ii in range(len(x_data_point[0,:])):
            B_temp = B_sym[i]
            for k in range(len(var_sym)):
                B_temp = B_temp.subs(var_sym[k], x_data_point[k,ii])
            B_temp2 = B_temp2+B_temp.subs(f_true, f_data_point[ii])
        B[i] = np.float64(B_temp2.evalf())
    
    B = np.reshape(B, B_sym.shape)

    C = np.linalg.solve(A,B)

    f_fit = np.zeros_like(f_data_point)
    for ii in range(len(f_data_point)):
        A_temp = f
        for k in range(len(par_sym)):
            A_temp = A_temp.subs(par_sym[k],C[k])
        for k in range(len(var_sym)):
            A_temp = A_temp.subs(var_sym[k], x_data_point[k,ii])
        f_fit[ii] = np.float64(A_temp.evalf())

    Err = np.linalg.norm(f_data_point-f_fit,ord=2)/np.linalg.norm(f_data_point,ord=2)



    return C, f_fit, Err
            
def lotka_rhs(t, x_y, b, p, d, r):
    x, y = x_y
    return [(b-p*y)*x, (r*x-d)*y]


def regression (equation, fit_par, ind_var, x_data_point, f_data_point, mode, add_par = [None, None]):

    par_sym = [sy.Symbol(fit_par[k]) for k in range(len(fit_par))]
    var_sym = [sy.Symbol(ind_var[k]) for k in range(len(ind_var))]
    f = parse_expr(equation)
    f_ex = sy.expand(f)
    f_args = f_ex.args

    a_sym= sy.zeros(1,len(par_sym))
    for i in range(len(f_args)):
        comp = f_args[i]
        for l in range(len(par_sym)):
            if par_sym[l] in comp.free_symbols:
                a_sym[l] = comp/par_sym[l]

    A = np.zeros((len(x_data_point[0,:]),len(par_sym)))
    for ii in range(len(x_data_point[0,:])):
        A_temp = a_sym
        for k in range(len(var_sym)):
            A_temp = A_temp.subs(var_sym[k],x_data_point[k,ii])
        A[ii,:] = np.float64(A_temp.evalf())

    #B = (np.matrix(f_data_point)).T
    B = f_data_point

    if mode == 'pinv':

        C = np.linalg.pinv(A) @ B
        
    
    elif mode == 'lstsq':

        C = np.linalg.lstsq(A, B, rcond=None)[0]
    
    elif mode == 'lasso':
        
        if add_par[0] != None:
            alpha = add_par[0]
        else:
            alpha = 1.0
        regr = linear_model.Lasso(alpha=alpha, copy_X=True, max_iter=10**5,random_state=0)
        regr.fit(A, B)
        C = regr.coef_
        
    elif mode == 'ridge':

        if add_par[0] != None:
            alpha = add_par[0]
        else:
            alpha = 1.0
        regr = linear_model.Ridge(alpha=alpha, copy_X=True, max_iter=10**5,random_state=0)
        regr.fit(A, B)
        C = regr.coef_
    
    elif mode == 'elastic':

        if add_par[0] != None:
            alpha = add_par[0]
        else:
            alpha = 1.0
        if add_par[1] != None:
            rho = add_par[1]
        else:
            rho = 0.5
        regr = linear_model.ElasticNet(alpha=alpha, l1_ratio=rho, copy_X=True, max_iter=10**5,random_state=0)
        regr.fit(A, B)
        C = regr.coef_

    elif mode == 'huber':

        regr = linear_model.HuberRegressor()
        regr.fit(A, B)
        C = regr.coef_


    f_fit = A @ C
    Err = np.linalg.norm(f_data_point-f_fit,ord=2)/np.linalg.norm(f_data_point,ord=2)

    return C, f_fit, Err


def buildLibrary(X, polyorder, sincos, hyperbole, log, exp):

    n = X.shape[0]
    n_var = X.shape[1]

    L = np.zeros((n,1))
    fun_list = []
    
    # poly order 0
    L[:,0] = np.ones(n)
    fun_list.append('1')

    # poly order 1
    for i in range(n_var):
        L = np.append(L,X[:,i].reshape((-1,1)),axis=1)
        fun_list.append('x{}'.format(i))     
    
    # poly order 2
    if polyorder >= 2:
        for i in range(n_var):
            for j in range(i,n_var):
                L = np.append(L,(X[:,i]*X[:,j]).reshape((-1,1)),axis=1)
                fun_list.append('x{}*x{}'.format(i,j))
                
    # poly order 3
    if polyorder >= 3:
        for i in range(n_var):
            for j in range(i,n_var):
                for k in range(j,n_var):
                    L = np.append(L,(X[:,i]*X[:,j]*X[:,k]).reshape((-1,1)),axis=1)
                    fun_list.append('x{}*x{}*x{}'.format(i,j,k))
    
    # poly order 4
    if polyorder >= 4:
        for i in range(n_var):
            for j in range(i,n_var):
                for k in range(j,n_var):
                    for l in range(k,n_var):
                        L = np.append(L,(X[:,i]*X[:,j]*X[:,k]*X[:,l]).reshape((-1,1)),axis=1)
                        fun_list.append('x{}*x{}*x{}*x{}'.format(i,j,k,l))
    
    if sincos >= 1:
        # sinorder 1 - 2
        for i in range(n_var):
            L = np.append(L, (np.sin(X[:,i])).reshape((-1,1)),axis=1)
            L = np.append(L, (np.cos(X[:,i])).reshape((-1,1)),axis=1)
            L = np.append(L, (np.tan(X[:,i])).reshape((-1,1)),axis=1)
            fun_list.append('sin(x{})'.format(i))
            fun_list.append('cos(x{})'.format(i))
            fun_list.append('tan(x{})'.format(i))
    
    if sincos >= 2:
        # sinorder 1 - 2
        for i in range(n_var):
            for j in range(i, n_var):
                L = np.append(L, (np.sin(X[:,i]*X[:,j])).reshape((-1,1)),axis=1)
                L = np.append(L, (np.cos(X[:,i]*X[:,j])).reshape((-1,1)),axis=1)
                fun_list.append('sin(x{}*x{})'.format(i,j))
                fun_list.append('cos(x{}*x{})'.format(i,j))
    
    if hyperbole >= 1:
        # hyperboleorder 1 - 2
        for i in range(n_var):
            L = np.append(L, (1/X[:,i]).reshape((-1,1)),axis=1)
            fun_list.append('1/(x{})'.format(i))
            

    if hyperbole >= 2:
        # hyperboleorder 1 - 2
        for i in range(n_var):
            for j in range(i, n_var):
                L = np.append(L, (1/(X[:,i]*X[:,j])).reshape((-1,1)),axis=1)
                fun_list.append('1/(x{}*x{})'.format(i,j))

    if log >= 1:
        # hyperboleorder 1 - 2
        for i in range(n_var):
            L = np.append(L, (np.log(np.abs(X[:,i]))).reshape((-1,1)),axis=1)
            fun_list.append('log(|x{}|)'.format(i))
            
    if log >= 2:
        # hyperboleorder 1 - 2
        for i in range(n_var):
            for j in range(i, n_var):
                L = np.append(L, (np.log(np.abs(X[:,i]*X[:,j]))).reshape((-1,1)),axis=1)
                fun_list.append('log(|x{}*x{}|)'.format(i,j))

    if exp >= 1:
        # hyperboleorder 1 - 2
        for i in range(n_var):
            L = np.append(L, (np.exp(X[:,i])).reshape((-1,1)),axis=1)
            fun_list.append('exp(x{})'.format(i))
            
    if exp >= 2:
        # hyperboleorder 1 - 2
        for i in range(n_var):
            for j in range(i, n_var):
                L = np.append(L, (np.exp(X[:,i]*X[:,j])).reshape((-1,1)),axis=1)
                fun_list.append('exp(x{}*x{})'.format(i,j))

                
    return L, fun_list


def sparsifyDynamics(L,dXdt,lamb):
    n = dXdt.shape[1]
    Xi = np.linalg.lstsq(L,dXdt,rcond=None)[0] # Initial guess: Least-squares
    
    for k in range(10):
        L_ave = np.repeat((np.average(L,axis =0)).reshape((-1,1)), n, axis=1)
        Xi_adim = Xi*L_ave
        Xi_adim_rel = Xi_adim/np.average(Xi_adim, axis=0)
        smallinds = np.abs(Xi_adim_rel) < lamb # Find small coefficients
        Xi[smallinds] = 0                          # and threshold
        for ind in range(n):                       # n is state dimension
            biginds = smallinds[:,ind] == 0
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds,ind] = np.linalg.lstsq(L[:,biginds],dXdt[:,ind],rcond=None)[0]
            
    return Xi 


def fitness_fun(p):
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join('.', 'data')
    name = os.path.join(path, 'animals_population.mat')

    population_mat = io.loadmat(name)
    population_full = population_mat['pop']
    years = population_mat['years'][0]
    x0 = population_full[:,0]
    time = (years-years[0])
    Dt = [time[0], time[-1]]

    sol = integrate.solve_ivp(lotka_rhs,Dt, x0, t_eval =time,\
        args = (p[0], p[1],p[2],p[3]), method='Radau')
    if sol.status !=0:
        Err = 100000
    else:
        y = sol.y
        Err = np.linalg.norm(population_full-y,ord=2)/np.linalg.norm(population_full,ord=2)

    return Err

def optDE1220 ():
    
    algo = algorithm(de1220(gen = 1000, ftol = 1e-5, xtol = 1e-6))
    algo.set_verbosity(2)
    prob = problem(OptProblem(4))
    pop = population(prob = prob, size = 10, seed = 444)
    pop = algo.evolve(pop)
    uda = algo.extract(de1220)
    
    return pop.champion_x, pop.champion_f


class OptProblem:

    def __init__( self, dim ):
        self.dim = dim
    
    def fitness( self, XXX ):
        YYY = fitness_fun(XXX)
        return[float(YYY)]

    def get_bounds(self):
        bound = ([-5,-1,-1,-1], [1, 1,1,1])
        return bound


