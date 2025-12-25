"""
Experiment B: How does QPE precision (counting qubits m) affect accuracy?
This demonstrates the relationship between circuit depth and estimation quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import Aer
import time

# Configuration
def f(x):
    return np.exp(x)

INTERVAL_START = 0
INTERVAL_END = 2*np.pi
N_DOMAIN_QUBITS = 5
N_SHOTS = 8192
TOP_K = 5

N_POINTS = 2**N_DOMAIN_QUBITS

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

# Prepare circuits once
state_prep = create_state_preparation(g_values, N_DOMAIN_QUBITS)
grover_op = create_grover_operator(state_prep, N_DOMAIN_QUBITS)
backend = Aer.get_backend('qasm_simulator')

# EXPERIMENT: Test different m values
M_values = [3, 4, 5, 6]  # Number of counting qubits
results = []

print("\n" + "="*70)
print("EXPERIMENT B: COUNTING QUBIT COMPARISON")
print("="*70)

for m in M_values:
    M = 2**m
    print(f"\n--- Testing m = {m} (M = {M} precision levels) ---")
    
    # Build circuit
    qae_circuit = create_qae_circuit(state_prep, grover_op, m, N_DOMAIN_QUBITS)
    qae_transpiled = transpile(qae_circuit, backend, optimization_level=1)
    
    circuit_depth = qae_transpiled.depth()
    total_qubits = qae_circuit.num_qubits
    
    print(f"Circuit depth: {circuit_depth}")
    print(f"Total qubits: {total_qubits}")
    
    # Run simulation
    start_time = time.time()
    job = backend.run(qae_transpiled, shots=N_SHOTS)
    result = job.result()
    counts = result.get_counts()
    sim_time = time.time() - start_time
    
    print(f"Simulation time: {sim_time:.2f}s")
    
    # Extract estimates
    estimates = []
    for bitstring, count in counts.items():
        y = int(bitstring, 2)
        prob = count / N_SHOTS
        theta_a = np.pi * y / M
        a_estimate = np.sin(theta_a)**2
        estimates.append((a_estimate, prob, y, theta_a))
    
    estimates.sort(key=lambda x: x[1], reverse=True)
    
    # Weighted average with top K
    K = min(TOP_K, len(estimates))
    top_estimates = estimates[:K]
    total_prob = sum(prob for _, prob, _, _ in top_estimates)
    weighted_a = sum(a * prob for a, prob, _, _ in top_estimates) / total_prob
    
    error_g = abs(weighted_a - mean_g_analytical)
    mean_f_estimate = weighted_a * (f_max - f_min) + f_min
    error_f = abs(mean_f_estimate - mean_f_analytical)
    rel_error = error_f / mean_f_analytical * 100
    
    # Phase resolution
    phase_resolution = 2 * np.pi / M
    
    results.append({
        'm': m,
        'M': M,
        'weighted_a': weighted_a,
        'error_g': error_g,
        'mean_f_estimate': mean_f_estimate,
        'error_f': error_f,
        'rel_error': rel_error,
        'circuit_depth': circuit_depth,
        'sim_time': sim_time,
        'phase_resolution': phase_resolution,
        'total_qubits': total_qubits
    })
    
    print(f"Estimated μ(g): {weighted_a:.6f}")
    print(f"Error in μ(g): {error_g:.6f}")
    print(f"Relative error: {rel_error:.2f}%")
    print(f"Phase resolution: {phase_resolution:.4f} rad")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Error vs m
ax1 = axes[0, 0]
m_plot = [r['m'] for r in results]
error_plot = [r['rel_error'] for r in results]
ax1.plot(m_plot, error_plot, 'o-', linewidth=2.5, markersize=10, color='blue')
for m, err in zip(m_plot, error_plot):
    ax1.text(m, err + 0.1, f'{err:.2f}%', ha='center', fontsize=10, fontweight='bold')
ax1.set_xlabel('Number of Counting Qubits (m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
ax1.set_title('Estimation Accuracy vs QPE Precision', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(m_plot)

# Plot 2: Phase resolution
ax2 = axes[0, 1]
phase_res = [r['phase_resolution'] for r in results]
ax2.semilogy(m_plot, phase_res, 's-', linewidth=2.5, markersize=10, color='green')
for m, pr in zip(m_plot, phase_res):
    ax2.text(m, pr * 1.3, f'{pr:.3f}', ha='center', fontsize=9, fontweight='bold')
ax2.set_xlabel('Number of Counting Qubits (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Phase Resolution (radians)', fontsize=12, fontweight='bold')
ax2.set_title('QPE Phase Resolution', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xticks(m_plot)

# Plot 3: Circuit depth
ax3 = axes[1, 0]
depths = [r['circuit_depth'] for r in results]
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(depths)))
bars = ax3.bar(m_plot, depths, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
for bar, depth in zip(bars, depths):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{depth}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_xlabel('Number of Counting Qubits (m)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Circuit Depth', fontsize=12, fontweight='bold')
ax3.set_title('Circuit Complexity vs Precision', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(m_plot)

# Plot 4: Summary table
ax4 = axes[1, 1]
ax4.axis('off')
table_data = []
for r in results:
    table_data.append([
        f"{r['m']}",
        f"{r['M']}",
        f"{r['rel_error']:.2f}%",
        f"{r['circuit_depth']}",
        f"{r['sim_time']:.1f}s"
    ])

table = ax4.table(cellText=table_data,
                  colLabels=['m', 'M', 'Error', 'Depth', 'Time'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Header styling
for i in range(5):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best error
best_idx = np.argmin(error_plot)
for j in range(5):
    table[(best_idx+1, j)].set_facecolor('#90EE90')

ax4.set_title('Performance vs Resource Trade-off', fontsize=14, fontweight='bold', pad=20)

plt.suptitle('Experiment B: Effect of QPE Precision on Estimation Quality', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('experiment_B_counting_qubits.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure saved as 'experiment_B_counting_qubits.png'")
plt.show()

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Best precision: m = {results[best_idx]['m']}, error = {results[best_idx]['rel_error']:.2f}%")
print(f"Trade-off: Each additional qubit roughly doubles circuit depth")
print(f"Phase resolution improves exponentially: Δφ = 2π/2^m")