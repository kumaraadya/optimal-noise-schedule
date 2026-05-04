from src.cvxpy_solver import cvxpy_solver_separable, cvxpy_solver_full
from src.schedules import linear_schedule
from src.diffusion_model import DDPMProcess
from src.elbo import compute_total_elbo
import numpy as np

# Test 1: SLSQP full solver
print('=== SLSQP Full Solver Test (T=100) ===')
T = 100
beta_init = linear_schedule(T)
loss_before = compute_total_elbo(DDPMProcess(beta_init), d=1)
result = cvxpy_solver_full(T=T, d=1, verbose=True)
loss_after = compute_total_elbo(DDPMProcess(result['beta_opt']), d=1)
reduction = (loss_before - loss_after) / loss_before * 100
print(f'Loss before:  {loss_before:.2f}')
print(f'Loss after:   {loss_after:.2f}')
print(f'Reduction:    {reduction:.1f}%')
print(f'Status:       {result["status"]}')
print(f'Iterations:   {result["n_iters"]}')
print(f'Beta monotone: {np.all(np.diff(result["beta_opt"]) >= -1e-10)}')

# Test 2: CVXPY separable solver
print()
print('=== CVXPY Separable Solver Test (T=100) ===')
result_sep = cvxpy_solver_separable(T=T, d=1, verbose=True)
if result_sep['beta_opt'] is not None:
    loss_sep = compute_total_elbo(DDPMProcess(result_sep['beta_opt']), d=1)
    print(f'Loss (separable): {loss_sep:.2f}')
    print(f'Status:           {result_sep["status"]}')
    print(f'Beta monotone:    {np.all(np.diff(result_sep["beta_opt"]) >= -1e-10)}')
else:
    print(f'CVXPY result: {result_sep}')

# Test 3: SLSQP matches PGD and FW
print()
print('=== Solver Comparison (T=100) ===')
from src.pgd_solver import pgd_solver
from src.frank_wolfe import frank_wolfe_solver
r_pgd = pgd_solver(beta_init, lr=0.01, max_iters=300, verbose=False)
r_fw  = frank_wolfe_solver(beta_init, d=1, max_iters=200, verbose=False)
print(f'PGD   loss: {r_pgd["losses"][-1]:.4f}')
print(f'FW    loss: {r_fw["losses"][-1]:.4f}')
print(f'SLSQP loss: {loss_after:.4f}')