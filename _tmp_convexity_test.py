import sys
sys.path.insert(0, '.')
from src.convexity_analysis import run_full_convexity_analysis

report = run_full_convexity_analysis(T=20)
print('PSD:', report['psd_check']['is_psd'])
print('All components convex:', report['component_analysis']['all_convex'])
print('Fraction convex:', report['component_analysis']['fraction_convex'])
print('Diagonal all positive:', (report['diagonal_hessian'] >= -1e-8).all())