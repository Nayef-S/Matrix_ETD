import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm
from ETD import ETD1, ETD2

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 14})

#initialisation
rng = np.random.default_rng(1)

L = np.array([[-1,1],[1,-2]])
R = L.T
N = np.asarray([[7,2],[2,7]])
comm_NR = N@R - R@N 

dt_array = [0.00005, 0.0001, 0.0005, 0.001 ,0.005 , 0.01, 0.05, 0.1, 0.5, 1]

error_array_1 = np.zeros(len(dt_array))
error_array_2 = np.zeros(len(dt_array))
time_array = np.zeros(len(dt_array))

A_true = linalg.solve_sylvester(L,R,-1.0 * N)

for i, dt in tqdm(enumerate(dt_array)):
    A_0_1 = rng.standard_normal(size=L.shape)
    A_0_2 = rng.standard_normal(size=L.shape)

    t_final = 100
    n_steps = round(t_final/dt)

    m1_L_1, m1_R_1, m2_1 = ETD1(L, R, dt)
    m1_L_2 , m1_R_2 , m2_2 , _ , m4,_  =  ETD2(L, R, dt)

    for _ in range(n_steps):
        A_0_1 = m1_L_1 @ A_0_1 @ m1_R_1 + m2_1 @ N
        A_0_2 = m1_L_2 @ A_0_2 @ m1_R_2 + m2_2 @ N + m4 @ comm_NR

    error_array_1[i] = np.linalg.norm(A_true-A_0_1)
    error_array_2[i] = np.linalg.norm(A_true-A_0_2)
    time_array[i] = dt

plt.plot(time_array, error_array_1, 'b-', label=r'ETD1 error ')
plt.plot(time_array, error_array_2, 'r-', label=r'ETD2 error')
plt.plot(time_array, time_array, 'b--', label=r'$ \mathcal{O}(\Delta t)$')
plt.plot(time_array, time_array**2, 'r--', label=r'$ \mathcal{O}(\Delta t^2)$')
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'$|| C_{\infty} - C^{(n)}||_{2}$')
plt.xlabel(r'$\Delta t$')
plt.grid('both', linestyle='--', linewidth=1)
plt.legend(loc='best')
plt.savefig('ETD_toy1_error_vs_dt.pdf', dpi=250)