from src.pgd_solver import pgd_solver, pgd_with_momentum
from src.schedules import linear_schedule, get_all_baselines
from src.diffusion_model import DDPMProcess
from src.elbo import compute_total_elbo
import numpy as np

# Test 1: Basic PGD run
print('=== PGD Solver Test (T=100) ===')
beta_init = linear_schedule(100)
result = pgd_solver(
    beta_init,
    lr=0.01,
    max_iters=300,
    verbose=True
)
print(f'Initial loss (linear): {compute_total_elbo(DDPMProcess(beta_init)):.2f}')
print(f'Final loss   (PGD):    {result["losses"][-1]:.2f}')
print(f'Iterations:            {result["n_iters"]}')
print(f'Converged:             {result["converged"]}')
print(f'Beta monotone:         {all(result["beta_opt"][i] <= result["beta_opt"][i+1] for i in range(99))}')

# Test 2: PGD with momentum
print()
print('=== PGD with Momentum Test (T=100) ===')
result_mom = pgd_with_momentum(
    beta_init,
    lr=0.01,
    momentum=0.9,
    max_iters=300,
    verbose=False
)
print(f'Final loss (momentum): {result_mom["losses"][-1]:.2f}')
print(f'Iterations:            {result_mom["n_iters"]}')

# Test 3: Compare all initializations
print()
print('=== Initialization Robustness Test ===')
for name, b in get_all_baselines(100).items():
    r = pgd_solver(b, lr=0.01, max_iters=300, verbose=False)
    print(f'Init={name:12s}: final_loss={r["losses"][-1]:.2f}')

# Test 4: Improvement check
print()
print('=== Improvement Check ===')
init_loss  = compute_total_elbo(DDPMProcess(beta_init))
final_loss = result['losses'][-1]
improvement = 100 * (init_loss - final_loss) / init_loss
print(f'Loss reduced by: {improvement:.1f}%')
print(f'PGD beats linear: {final_loss < init_loss}')