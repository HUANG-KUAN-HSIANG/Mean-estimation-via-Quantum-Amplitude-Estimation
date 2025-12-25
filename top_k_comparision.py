"""
Experiment A: How does the number of top-K outcomes affect accuracy?
This shows the optimal choice for weighted averaging.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import Aer

# Configuration
def f(x):
    return np.exp(x)

INTERVAL_START = 0
INTERVAL_END = 2*np.pi
N_DOMAIN_QUBITS = 5
N_COUNTING_QUBITS = 5
N_SHOTS = 8192

N_POINTS = 2**N_DOMAIN_QUBITS
M = 2**N_COUNTING_QUBITS

# Discretize and normalize
x_values = np.linspace(INTERVAL_START, INTERVAL_END, N_POINTS)
f_values = np.array([f(x) for x in x_values])
f_min, f_max = np.min(f_values), np.max(f_values)
g_values = (f_values - f_min) / (f_max - f_min)

mean_f_analytical = np.mean(f_values)
mean_g_analytical = np.mean(g_values)

print(f"Analytical mean of g(x): {mean_g_analytical:.6f}")
print(f"Analytical mean of f(x): {mean_f_analytical:.6f}")

# State preparation
def create_state_preparation(g_values, n_qubits):
    N = len(g_values)
    qc = QuantumCircuit(n_qubits + 1, name='A')
    
    for i in range(n_qubits):
        qc.h(i)
    
    for j in range(N):
        if 0 < g_values[j] < 1:
            theta = 2 * np.arcsin(np.sqrt(g_values[j]))
            binary_j = format(j, f'0{n_qubits}b')
            
            for qubit_idx, bit in enumerate(binary_j):
                if bit == '0':
                    qc.x(qubit_idx)
            
            qc.mcry(theta, list(range(n_qubits)), n_qubits)
            
            for qubit_idx, bit in enumerate(binary_j):
                if bit == '0':
                    qc.x(qubit_idx)
        
        elif g_values[j] >= 0.999:
            binary_j = format(j, f'0{n_qubits}b')
            for qubit_idx, bit in enumerate(binary_j):
                if bit == '0':
                    qc.x(qubit_idx)
            qc.mcx(list(range(n_qubits)), n_qubits)
            for qubit_idx, bit in enumerate(binary_j):
                if bit == '0':
                    qc.x(qubit_idx)
    
    return qc

# Grover operator
def create_grover_operator(state_prep, n_domain):
    n_qubits = n_domain + 1
    ancilla = n_domain
    qc = QuantumCircuit(n_qubits, name='Q')
    
    qc.x(ancilla)
    qc.z(ancilla)
    qc.x(ancilla)
    
    qc.compose(state_prep.inverse(), inplace=True)
    
    for i in range(n_qubits):
        qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i in range(n_qubits):
        qc.x(i)
    
    qc.compose(state_prep, inplace=True)
    
    return qc

# QAE circuit
def create_qae_circuit(state_prep, grover_op, n_counting, n_domain):
    n_workspace = n_domain + 1
    
    counting = QuantumRegister(n_counting, 'counting')
    workspace = QuantumRegister(n_workspace, 'workspace')
    c = ClassicalRegister(n_counting, 'c')
    
    qc = QuantumCircuit(counting, workspace, c)
    
    qc.compose(state_prep, workspace, inplace=True)
    qc.barrier()
    
    for j in range(n_counting):
        qc.h(counting[j])
    qc.barrier()
    
    for j in range(n_counting):
        for _ in range(2**j):
            ctrl_q = grover_op.control(1)
            qc.compose(ctrl_q, [counting[j]] + list(workspace), inplace=True)
    
    qc.barrier()
    
    qft_inv = QFT(n_counting, inverse=True)
    qc.compose(qft_inv, counting, inplace=True)
    qc.barrier()
    
    qc.measure(counting, c)
    
    return qc

# Run QAE
print("\nBuilding and running QAE circuit...")
state_prep = create_state_preparation(g_values, N_DOMAIN_QUBITS)
grover_op = create_grover_operator(state_prep, N_DOMAIN_QUBITS)
qae_circuit = create_qae_circuit(state_prep, grover_op, N_COUNTING_QUBITS, N_DOMAIN_QUBITS)

backend = Aer.get_backend('qasm_simulator')
qae_transpiled = transpile(qae_circuit, backend, optimization_level=1)

print(f"Running simulation with {N_SHOTS} shots...")
job = backend.run(qae_transpiled, shots=N_SHOTS)
result = job.result()
counts = result.get_counts()
print("Simulation complete!")

# Extract estimates
estimates = []
for bitstring, count in counts.items():
    y = int(bitstring, 2)
    prob = count / N_SHOTS
    theta_a = np.pi * y / M
    a_estimate = np.sin(theta_a)**2
    estimates.append((a_estimate, prob, y, theta_a))

estimates.sort(key=lambda x: x[1], reverse=True)

# EXPERIMENT: Test different K values
K_values = [1, 3, 5, 7, 10, 15, 20]
results = []

print("\n" + "="*70)
print("EXPERIMENT A: TOP-K COMPARISON")
print("="*70)

for K in K_values:
    if K > len(estimates):
        K = len(estimates)
    
    top_estimates = estimates[:K]
    total_prob = sum(prob for _, prob, _, _ in top_estimates)
    weighted_a = sum(a * prob for a, prob, _, _ in top_estimates) / total_prob
    
    error_g = abs(weighted_a - mean_g_analytical)
    mean_f_estimate = weighted_a * (f_max - f_min) + f_min
    error_f = abs(mean_f_estimate - mean_f_analytical)
    rel_error = error_f / mean_f_analytical * 100
    
    results.append({
        'K': K,
        'weighted_a': weighted_a,
        'error_g': error_g,
        'mean_f_estimate': mean_f_estimate,
        'error_f': error_f,
        'rel_error': rel_error,
        'total_prob': total_prob
    })
    
    print(f"\nK = {K:2d}:")
    print(f"  Total probability covered: {total_prob:.4f}")
    print(f"  Estimated μ(g): {weighted_a:.6f}")
    print(f"  Error in μ(g): {error_g:.6f}")
    print(f"  Estimated μ(f): {mean_f_estimate:.6f}")
    print(f"  Relative error: {rel_error:.2f}%")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Error vs K
ax1 = axes[0, 0]
K_plot = [r['K'] for r in results]
error_plot = [r['rel_error'] for r in results]
ax1.plot(K_plot, error_plot, 'o-', linewidth=2, markersize=8, color='blue')
ax1.axhline(error_plot[0], color='red', linestyle='--', linewidth=2, 
            label=f'K=1 (ML): {error_plot[0]:.2f}%', alpha=0.7)
optimal_idx = np.argmin(error_plot)
ax1.plot(K_plot[optimal_idx], error_plot[optimal_idx], 'g*', markersize=20,
         label=f'Optimal K={K_plot[optimal_idx]}', zorder=5)
ax1.set_xlabel('Number of Top Outcomes (K)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
ax1.set_title('Error vs Number of Outcomes Used', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Plot 2: Probability coverage
ax2 = axes[0, 1]
prob_plot = [r['total_prob'] for r in results]
ax2.plot(K_plot, prob_plot, 's-', linewidth=2, markersize=8, color='green')
ax2.set_xlabel('Number of Top Outcomes (K)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Total Probability Covered', fontsize=12, fontweight='bold')
ax2.set_title('Information Utilization vs K', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(0.8, color='orange', linestyle='--', alpha=0.5, label='80% threshold')
ax2.legend(fontsize=10)

# Plot 3: Improvement factor
ax3 = axes[1, 0]
ml_error = error_plot[0]
improvement = [ml_error / e for e in error_plot]
ax3.bar(K_plot, improvement, color='purple', alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.axhline(1, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax3.set_xlabel('Number of Top Outcomes (K)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Improvement Factor over ML', fontsize=12, fontweight='bold')
ax3.set_title('Weighted Averaging Advantage', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for i, (k, imp) in enumerate(zip(K_plot, improvement)):
    ax3.text(k, imp + 0.1, f'{imp:.2f}×', ha='center', fontsize=9, fontweight='bold')

# Plot 4: Summary table
ax4 = axes[1, 1]
ax4.axis('off')
table_data = []
for r in results[:8]:  # Show first 8
    table_data.append([
        f"{r['K']}",
        f"{r['total_prob']:.3f}",
        f"{r['rel_error']:.2f}%",
        f"{ml_error/r['rel_error']:.2f}×"
    ])

table = ax4.table(cellText=table_data,
                  colLabels=['K', 'Prob', 'Error', 'Improve'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Header styling
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight optimal
for i in range(len(table_data)):
    if int(table_data[i][0]) == K_plot[optimal_idx]:
        for j in range(4):
            table[(i+1, j)].set_facecolor('#90EE90')

ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

plt.suptitle('Experiment A: Optimal Number of Outcomes for Weighted Averaging', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('experiment_A_top_k_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure saved as 'experiment_A_top_k_comparison.png'")
plt.show()

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Optimal K: {K_plot[optimal_idx]}")
print(f"Best relative error: {error_plot[optimal_idx]:.2f}%")
print(f"Improvement over ML (K=1): {improvement[optimal_idx]:.2f}×")
print(f"Probability covered: {results[optimal_idx]['total_prob']:.3f}")