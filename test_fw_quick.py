from src.frank_wolfe import frank_wolfe_solver, linear_minimization_oracle
from src.schedules import linear_schedule, cosine_schedule
from src.diffusion_model import DDPMProcess
from src.elbo import compute_total_elbo
import numpy as np

# Test 1: LMO basic properties
print('=== LMO Test ===')
np.random.seed(42)
grad = np.random.randn(100)
s = linear_minimization_oracle(grad)
print(f'LMO monotone:      {np.all(np.diff(s) >= -1e-12)}')
print(f'LMO unique vals:   {len(np.unique(np.round(s, 8)))} (should be <= 2)')
print(f'LMO min value:     {s.min():.6f} (should be ~1e-4)')
print(f'LMO max value:     {s.max():.6f} (should be ~0.02)')

# Test 2: FW open_loop reduces loss
print()
print('=== FW Open Loop Test (T=100) ===')
T = 100
beta_init = linear_schedule(T)
loss_before = compute_total_elbo(DDPMProcess(beta_init), d=1)
result = frank_wolfe_solver(beta_init, d=1, max_iters=200,
                             step_rule='open_loop', verbose=True)
loss_after = result['losses'][-1]
reduction = (loss_before - loss_after) / loss_before * 100
print(f'Loss before: {loss_before:.2f}')
print(f'Loss after:  {loss_after:.2f}')
print(f'Reduction:   {reduction:.1f}%')
print(f'Converged:   {result["converged"]}')
print(f'Iterations:  {result["n_iters"]}')
print(f'Gap start:   {result["gaps"][0]:.4f}')
print(f'Gap end:     {result["gaps"][-1]:.6f}')

# Test 3: FW line_search
print()
print('=== FW Line Search Test (T=100) ===')
result_ls = frank_wolfe_solver(beta_init, d=1, max_iters=50,
                                step_rule='line_search', verbose=False)
print(f'Loss after:  {result_ls["losses"][-1]:.2f}')
print(f'Converged:   {result_ls["converged"]}')
print(f'Iterations:  {result_ls["n_iters"]}')

# Test 4: Cosine init
print()
print('=== FW Cosine Init Test (T=100) ===')
beta_cosine = cosine_schedule(T)
result_cos = frank_wolfe_solver(beta_cosine, d=1, max_iters=200,
                                 step_rule='open_loop', verbose=False)
print(f'Loss after:  {result_cos["losses"][-1]:.2f}')
print(f'Converged:   {result_cos["converged"]}')

# Test 5: Duality gap decreases
print()
print('=== Duality Gap Convergence ===')
gaps = result['gaps']
print(f'Gap[0]:   {gaps[0]:.4f}')
print(f'Gap[-1]:  {gaps[-1]:.6f}')
print(f'Gap decreased: {gaps[0] > gaps[-1]}')

# Test 6: Improvement summary
print()
print('=== Improvement Check ===')
improvement = (loss_before - loss_after) / loss_before * 100
print(f'FW beats linear: {loss_after < loss_before}')
print(f'Loss reduced by: {improvement:.1f}%')