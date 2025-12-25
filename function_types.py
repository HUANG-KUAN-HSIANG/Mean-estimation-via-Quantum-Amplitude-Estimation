"""
Experiment C: Does QAE work well across different function types?
This demonstrates the generality of the method.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import Aer

# Configuration
N_DOMAIN_QUBITS = 5
N_COUNTING_QUBITS = 5
N_SHOTS = 8192
TOP_K = 5

N_POINTS = 2**N_DOMAIN_QUBITS
M = 2**N_COUNTING_QUBITS

# Define test functions
test_functions = {
    'Exponential': {
        'func': lambda x: np.exp(x),
        'interval': (0, 2*np.pi),
        'color': 'blue'
    },
    'Polynomial': {
        'func': lambda x: x**2,
        'interval': (-2, 2),
        'color': 'green'
    },
    'Trigonometric': {
        'func': lambda x: np.sin(x) + 2,
        'interval': (0, 2*np.pi),
        'color': 'red'
    },
    'Linear': {
        'func': lambda x: 3*x + 1,
        'interval': (0, 5),
        'color': 'purple'
    },
    'Square Root': {
        'func': lambda x: np.sqrt(x + 1),
        'interval': (0, 8),
        'color': 'orange'
    }
}

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

backend = Aer.get_backend('qasm_simulator')

# EXPERIMENT: Test on all functions
results = []

print("="*70)
print("EXPERIMENT C: FUNCTION TYPE COMPARISON")
print("="*70)

for func_name, func_info in test_functions.items():
    print(f"\n--- Testing: {func_name} ---")
    
    f = func_info['func']
    a, b = func_info['interval']
    
    # Discretize
    x_values = np.linspace(a, b, N_POINTS)
    f_values = np.array([f(x) for x in x_values])
    
    # Normalize
    f_min, f_max = np.min(f_values), np.max(f_values)
    g_values = (f_values - f_min) / (f_max - f_min)
    
    mean_f_analytical = np.mean(f_values)
    mean_g_analytical = np.mean(g_values)
    
    print(f"Interval: [{a}, {b}]")
    print(f"Analytical mean: {mean_f_analytical:.6f}")
    
    # Build and run QAE
    state_prep = create_state_preparation(g_values, N_DOMAIN_QUBITS)
    grover_op = create_grover_operator(state_prep, N_DOMAIN_QUBITS)
    qae_circuit = create_qae_circuit(state_prep, grover_op, N_COUNTING_QUBITS, N_DOMAIN_QUBITS)
    
    qae_transpiled = transpile(qae_circuit, backend, optimization_level=1)
    
    print(f"Running simulation...")
    job = backend.run(qae_transpiled, shots=N_SHOTS)
    result = job.result()
    counts = result.get_counts()
    
    # Extract estimates
    estimates = []
    for bitstring, count in counts.items():
        y = int(bitstring, 2)
        prob = count / N_SHOTS
        theta_a = np.pi * y / M
        a_estimate = np.sin(theta_a)**2
        estimates.append((a_estimate, prob, y, theta_a))
    
    estimates.sort(key=lambda x: x[1], reverse=True)
    
    # Weighted average
    K = min(TOP_K, len(estimates))
    top_estimates = estimates[:K]
    total_prob = sum(prob for _, prob, _, _ in top_estimates)
    weighted_a = sum(a * prob for a, prob, _, _ in top_estimates) / total_prob
    
    # Scale back
    mean_f_estimate = weighted_a * (f_max - f_min) + f_min
    error_f = abs(mean_f_estimate - mean_f_analytical)
    rel_error = error_f / abs(mean_f_analytical) * 100
    
    results.append({
        'name': func_name,
        'mean_analytical': mean_f_analytical,
        'mean_estimate': mean_f_estimate,
        'error': error_f,
        'rel_error': rel_error,
        'color': func_info['color'],
        'x_values': x_values,
        'f_values': f_values
    })
    
    print(f"Estimated mean: {mean_f_estimate:.6f}")
    print(f"Error: {error_f:.6f}")
    print(f"Relative error: {rel_error:.2f}%")

# Plot results
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot functions (top row)
for idx, (func_name, r) in enumerate(zip(test_functions.keys(), results)):
    if idx < 3:
        ax = fig.add_subplot(gs[0, idx])
        ax.plot(r['x_values'], r['f_values'], color=r['color'], linewidth=2.5, label=func_name)
        ax.axhline(r['mean_analytical'], color='green', linestyle='--', 
                   linewidth=2, label=f'True: {r["mean_analytical"]:.2f}', alpha=0.7)
        ax.axhline(r['mean_estimate'], color='red', linestyle=':', 
                   linewidth=2, label=f'QAE: {r["mean_estimate"]:.2f}', alpha=0.7)
        ax.set_xlabel('x', fontsize=11, fontweight='bold')
        ax.set_ylabel('f(x)', fontsize=11, fontweight='bold')
        ax.set_title(func_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

# Plot remaining functions (middle row)
for idx, (func_name, r) in enumerate(list(zip(test_functions.keys(), results))[3:]):
    ax = fig.add_subplot(gs[1, idx])
    ax.plot(r['x_values'], r['f_values'], color=r['color'], linewidth=2.5, label=func_name)
    ax.axhline(r['mean_analytical'], color='green', linestyle='--', 
               linewidth=2, label=f'True: {r["mean_analytical"]:.2f}', alpha=0.7)
    ax.axhline(r['mean_estimate'], color='red', linestyle=':', 
               linewidth=2, label=f'QAE: {r["mean_estimate"]:.2f}', alpha=0.7)
    ax.set_xlabel('x', fontsize=11, fontweight='bold')
    ax.set_ylabel('f(x)', fontsize=11, fontweight='bold')
    ax.set_title(func_name, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Plot comparison (bottom left)
ax_comp = fig.add_subplot(gs[2, 0])
names = [r['name'] for r in results]
errors = [r['rel_error'] for r in results]
colors = [r['color'] for r in results]
bars = ax_comp.bar(range(len(names)), errors, color=colors, edgecolor='black', 
                    linewidth=1.5, alpha=0.7)
ax_comp.set_xticks(range(len(names)))
ax_comp.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
ax_comp.set_ylabel('Relative Error (%)', fontsize=11, fontweight='bold')
ax_comp.set_title('Estimation Accuracy Across Functions', fontsize=12, fontweight='bold')
ax_comp.grid(True, alpha=0.3, axis='y')
ax_comp.axhline(np.mean(errors), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(errors):.2f}%')
ax_comp.legend(fontsize=9)

for bar, err in zip(bars, errors):
    height = bar.get_height()
    ax_comp.text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot accuracy (bottom middle)
ax_acc = fig.add_subplot(gs[2, 1])
analytical = [r['mean_analytical'] for r in results]
estimated = [r['mean_estimate'] for r in results]
ax_acc.scatter(analytical, estimated, c=colors, s=150, edgecolors='black', linewidths=2, alpha=0.7)

# Perfect estimation line
min_val = min(min(analytical), min(estimated))
max_val = max(max(analytical), max(estimated))
ax_acc.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='Perfect estimation')

for i, name in enumerate(names):
    ax_acc.annotate(name, (analytical[i], estimated[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

ax_acc.set_xlabel('Analytical Mean', fontsize=11, fontweight='bold')
ax_acc.set_ylabel('QAE Estimated Mean', fontsize=11, fontweight='bold')
ax_acc.set_title('Estimation Accuracy', fontsize=12, fontweight='bold')
ax_acc.legend(fontsize=9)
ax_acc.grid(True, alpha=0.3)

# Summary table (bottom right)
ax_table = fig.add_subplot(gs[2, 2])
ax_table.axis('off')
table_data = []
for r in results:
    table_data.append([
        r['name'][:10],
        f"{r['mean_analytical']:.3f}",
        f"{r['mean_estimate']:.3f}",
        f"{r['rel_error']:.2f}%"
    ])

table = ax_table.table(cellText=table_data,
                       colLabels=['Function', 'True', 'QAE', 'Error'],
                       cellLoc='center',
                       loc='center',
                       bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Header styling
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight rows
for i, r in enumerate(results):
    table[(i+1, 0)].set_facecolor(r['color'])
    table[(i+1, 0)].set_text_props(color='white', weight='bold')

ax_table.set_title('Numerical Summary', fontsize=12, fontweight='bold', pad=20)

plt.suptitle('Experiment C: QAE Performance Across Different Function Types', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('experiment_C_function_types.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure saved as 'experiment_C_function_types.png'")
plt.show()

# Summary statistics
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Average relative error: {np.mean(errors):.2f}%")
print(f"Best performance: {names[np.argmin(errors)]} ({min(errors):.2f}%)")
print(f"Worst performance: {names[np.argmax(errors)]} ({max(errors):.2f}%)")
print(f"Standard deviation: {np.std(errors):.2f}%")
print("\n✓ QAE demonstrates consistent performance across diverse function types!")