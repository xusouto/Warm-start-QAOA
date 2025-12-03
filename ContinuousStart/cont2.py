# Imports
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import copy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import csv
import json
import argparse
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from docplex.mp.model import Model
from qiskit.primitives import StatevectorSampler
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer, MinimumEigenOptimizer
from qiskit_optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from pprint import pprint
from typing import Optional, Union, Iterable, List, Tuple, Dict, Any
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.optimizers import COBYLA
from qiskit_optimization.problems.variable import VarType
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.utils import algorithm_globals 
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit import transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from pathlib import Path
Array = Any
Tensor = Any
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parser
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Quantum 250 Finalcial Portfolios Optimization Script")
parser.add_argument("--iter", type=int, required=True, help="Iteration number")
args = parser.parse_args()
iteration = args.iter
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Problem generation and relaxation 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def create_problem(mu, Sigma, total, q, lam):
    n = len(mu)
    mdl = Model()
    x = [mdl.binary_var(name=f"x{i}") for i in range(n)]
    sumx = mdl.sum(x)

    mdl.maximize(
        mdl.sum(mu[i]*x[i] for i in range(n))
        - q * mdl.sum(Sigma[i,j]*x[i]*x[j] for i in range(n) for j in range(n))
        - lam * (sumx - total) * (sumx - total)
    )
    return from_docplex_mp(mdl)
# -------------------------------------------------------------------------------------------------------
def relax_problem(problem) -> QuadraticProgram:
    """Change all variables to continuous."""
    relaxed_problem = copy.deepcopy(problem)
    for variable in relaxed_problem.variables:
        variable.vartype = VarType.CONTINUOUS
    return relaxed_problem
# -------------------------------------------------------------------------------------------------------
def random_mu_sigma(
    n: int = 6,
    *,
    seed: 123,
    mu_range: tuple[float, float] = (0.6, 2.1),
    var_range: tuple[float, float] = (0.02, 0.32),
    corr_strength: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)

    # 1) Random mu in the desired range
    mu = rng.uniform(mu_range[0], mu_range[1], size=n)

    # 2) Random correlation matrix (SPD, unit diagonal), then shrink toward I to control strength
    A = rng.normal(size=(n, n))
    C = A @ A.T  # SPD
    d = np.sqrt(np.diag(C))
    Dinv = np.diag(1.0 / d)
    R = Dinv @ C @ Dinv        # correlation with diag=1
    # Shrinkage: keep correlations moderate and well-conditioned
    R = (1.0 - corr_strength) * np.eye(n) + corr_strength * R

    # Numeric guard: clip tiny negative eigenvalues (rare due to rounding)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-8, None)
    R = (V * w) @ V.T

    # 3) Set variances on the diagonal (to match example scale) and build Sigma
    variances = rng.uniform(var_range[0], var_range[1], size=n)
    S = np.diag(np.sqrt(variances))
    Sigma = S @ R @ S
    Sigma = 0.5 * (Sigma + Sigma.T)  # enforce symmetry
    return mu, Sigma
# -------------------------------------------------------------------------------------------------------
def simulate_portfolio_gbm( 
    n: int = 6,
    N: int = 250,
    mu_range = (-0.05, 0.05),      # μ_i ~ Uniform[-5%, 5%]
    sigma_range = (-0.20, 0.20),   # σ_i ~ Uniform[-20%, 20%]  (can be negative per text)
    seed = 123,
):
    """
    Simulate n asset price paths via GBM for N days (S_{i,0}=1), compute daily returns r_{i,k},
    then return (mu, Sigma) where mu[i] = mean_k r_{i,k}, and Sigma is the n×n covariance of returns.
    """
    print ("seed was:", seed)
    rng = np.random.default_rng(seed)
  
    # Draw μ_i and σ_i as specified
    mu_i    = rng.uniform(mu_range[0],    mu_range[1],    size=n)     # shape (n,)
    sigma_i = rng.uniform(sigma_range[0], sigma_range[1], size=n)     # shape (n,)

    # Brownian motion increments z_k ~ N(0,1); W_k = sum_{l=1..k} z_l / sqrt(N)
    z = rng.standard_normal(size=(n, N))            # independent per asset/day
    W = np.cumsum(z, axis=1) / np.sqrt(N)           # shape (n, N), W_k

    # Build S_{i,k} using S_{i,0}=1 and the closed-form GBM solution in the text:
    # S_{i,k} = exp( (μ_i - σ_i^2/2) * (k/N) + σ_i * W_k )
    k_over_N = (np.arange(1, N+1) / N)[None, :]     # shape (1, N)
    drift    = (mu_i[:, None] - 0.5 * sigma_i[:, None]**2) * k_over_N
    diff     = sigma_i[:, None] * W
    S = np.empty((n, N+1))
    S[:, 0] = 1.0
    S[:, 1:] = np.exp(drift + diff)                 # since S_{i,0}=1

    # Daily returns r_{i,k} = S_{i,k} / S_{i,k-1} - 1   for k=1..N
    R = S[:, 1:] / S[:, :-1] - 1.0                  # shape (n, N)

    # Mean return per asset (across days)
    mu_vec = R.mean(axis=1)                          # shape (n,)

    # Covariance across assets of daily returns (rows=assets, columns=days)
    Sigma = np.cov(R, rowvar=True, bias=False)       # shape (n, n)

    return mu_vec, Sigma
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# QAOA Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    cost = get_expval1(params, ansatz, hamiltonian)
    return cost
# -------------------------------------------------------------------------------------------------------
def _bs_from_pre_sol(pre_sol) -> str:
    return "".join("1" if float(b) > 0.5 else "0" for b in pre_sol)
# -------------------------------------------------------------------------------------------------------
def get_probs1(circuit: QuantumCircuit, result, pre_sol):
    """
    Exact P(pre_sol) using StatevectorSampler, with a measured copy of the circuit
    to avoid the 'no classical registers' warning.
    `result` can be OptimizeResult (.x) or array-like.
    """
    params = getattr(result, "x", result)
    bound = circuit.assign_parameters(params).copy()

    # Add classical bits + measurements to avoid the warning (on the copy only)
    bound.measure_all(add_bits=True)

    bs = _bs_from_pre_sol(pre_sol)
    n = bound.num_qubits
    if len(bs) != n:
        raise ValueError(f"pre_sol length ({len(bs)}) != circuit qubits ({n})")

    sampler = StatevectorSampler()
    res = sampler.run([bound]).result()

    # Case A: classic SamplerResult with quasi_dists (int keys, little-endian)
    qd = getattr(res, "quasi_dists", None)
    if qd is not None:
        return float(qd[0].get(int(bs[::-1], 2), 0.0))

    # Case B: PrimitiveResult style with bitstring probabilities
    try:
        probs_bin = res[0].data.meas.get_probabilities()  # {'0101': p, ...}
        return float(probs_bin.get(bs, 0.0))
    except Exception:
        # Fallback: pure statevector (shouldn't be hit with StatevectorSampler)
        sv = Statevector.from_instruction(bound.remove_final_measurements(inplace=False))
        return float(abs(sv.data[int(bs[::-1], 2)])**2)
# -------------------------------------------------------------------------------------------------------
def get_expval1(result, candidate_circuit, hamiltonian):
    # If ansatz has a layout (because you transpiled), map the observable to it:
    estimator = StatevectorEstimator()
    job = estimator.run([(candidate_circuit, hamiltonian, result)])
    result = job.result()[0]
    cost = float(result.data.evs)
    return cost
# -------------------------------------------------------------------------------------------------------
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parameter initialization
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def create_custom_array(length: int, lower1: float, upper1: float, lower2: float, upper2: float) -> np.ndarray:
    """
    Creates an array of a given length where the first half has identical random values between
    lower1 and upper1, and the second half has identical random values between lower2 and upper2.

    Parameters:
        length (int): The desired length of the array (should be an even number).
        lower1 (float): Lower boundary for the first half.
        upper1 (float): Upper boundary for the first half.
        lower2 (float): Lower boundary for the second half.
        upper2 (float): Upper boundary for the second half.

    Returns:
        np.ndarray: The resulting array with the described pattern.
    """
    if length % 2 != 0:
        raise ValueError("Length must be an even number to split into two halves.")

    # Generate random values for both halves
    first_half_value = np.random.uniform(lower1, upper1)
    second_half_value = np.random.uniform(lower2, upper2)

    # Create the array
    array = np.concatenate((
        np.full(length // 2, first_half_value),
        np.full(length // 2, second_half_value)
    ))

    return array
# -------------------------------------------------------------------------------------------------------
def rand_vec(i: int, seed) -> np.ndarray:
    """[-2π, 2π]"""
    rng = np.random.default_rng(seed)
    return rng.uniform(-2*np.pi, 2*np.pi, size=2*i)
# -------------------------------------------------------------------------------------------------------
def create_custom_array_even_odd(length: int,
                        lower1: float, upper1: float,
                        lower2: float, upper2: float) -> np.ndarray:
    """
    Creates an array where all even indices hold the same random value sampled
    from [lower1, upper1], and all odd indices hold the same random value
    sampled from [lower2, upper2].

    Parameters:
        length (int): Desired array length (any non-negative integer).
        lower1, upper1: Bounds for the value used at even indices.
        lower2, upper2: Bounds for the value used at odd indices.

    Returns:
        np.ndarray: Array of shape (length,).
    """
    if length < 0:
        raise ValueError("Length must be non-negative.")
    if lower1 > upper1:
        raise ValueError("lower1 must be <= upper1.")
    if lower2 > upper2:
        raise ValueError("lower2 must be <= upper2.")

    # Sample one value for even indices and one for odd indices
    even_val = np.random.uniform(lower1, upper1)
    odd_val  = np.random.uniform(lower2, upper2)

    arr = np.empty(length, dtype=float)
    arr[0::2] = even_val  # even indices: 0, 2, 4, ...
    arr[1::2] = odd_val   # odd indices: 1, 3, 5, ...

    return arr
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# QAOA ansatz and solver
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def solve_hm(n, warm: bool, lukewarm: bool, cost_operator, pauli_params, pauli_weights, pre_sol, eps, depth, iteration, solution, parameters):
    circuit_w_anszt= QAOA_hm_1(n, pauli_params, pauli_weights, warm, lukewarm, pre_sol , eps, depth)
    estimator = StatevectorEstimator()
    result = minimize(
        cost_func_estimator,
        parameters,
        args=(circuit_w_anszt, cost_operator, estimator),
        method="COBYLA",
        options={                 
        'tol': 1e-3,                    
        'disp': True
    })
    print(result)
    print("COBYLA Optimization ended", flush = True)

    probs = get_probs1(circuit_w_anszt, result, solution)
    expval = get_expval1(result.x, circuit_w_anszt, cost_operator)
    print("Probability of obtaining result string:", probs, flush=True)
    print("Expectation value of Hc after otimization:", expval, flush=True)
    return probs, expval
# -------------------------------------------------------------------------------------------------------
def QAOA_hm_1(n, pauli_params, pauli_weights, warm: bool, lukewarm : bool, pre_sol , eps, depth):
    c_stars=[]
    print(f"depth is {depth}", flush=True)
    for var in pre_sol:
        if var <= eps:
            c_stars.append(eps)     
        elif var >= 1-eps:    
            c_stars.append(1-eps)
        else:
            c_stars.append(var)
    thetas = [2 * np.arcsin(np.sqrt(c_star)) for c_star in c_stars]
    gammas = ParameterVector("gamma", depth)  # gamma[0],...,gamma[depth-1]
    betas  = ParameterVector("beta",  depth)  # beta[0],...,beta[depth-1]

    # Build initial state (your function)
    init_qc = init(n, warm, thetas)

    circuit = QuantumCircuit(n, name="QAOA")
    circuit.append(init_qc, range(n))

    for j in range(depth):
        # Build layer-j subcircuits with distinct parameters
        qc_p_j     = cost(n, pauli_params, pauli_weights, gammas[j])  # uses gamma_j                  # uses beta_j

        if lukewarm:                                                  # warm start - normal mixer
            mixer_qc_j = mixer(n, False, thetas, betas[j])
        else:
            mixer_qc_j = mixer(n, warm, thetas, betas[j])


        circuit.append(qc_p_j, range(n))
        circuit.append(mixer_qc_j, range(n))
    

    # Optionally return params so you can bind later in order [gamma0,beta0,...]
    return circuit
# -------------------------------------------------------------------------------------------------------
def cost(n,pauli_terms: List[List[float]],weights: List[float],gamma):
    """
    Append the QAOA cost unitary U_C(gamma) = exp(-i * gamma * H_C)
    for H_C = sum_k w_k * Z^{(term_k)} (offset ignored as global phase).

    pauli_terms: list of 0/1 masks (floats ok), len = m, each of length n (num qubits)
    weights:     list/array of length m
    gamma:       float or qiskit Parameter
    """
    qc_p = QuantumCircuit(n)
    w = np.asarray(weights, dtype=float)
    for wk, term in zip(w, pauli_terms):
        idx = [i for i, b in enumerate(term) if b > 0.5]
        if len(idx) == 1:
            q = idx[0]
            # RZ(θ) = exp(-i θ Z / 2)  => θ = 2 * gamma * w
            qc_p.rz( wk * gamma, q)
        elif len(idx) == 2:
            i, j = idx
            # RZZ(θ) = exp(-i θ ZZ / 2) => θ = 2 * gamma * w
            qc_p.rzz( wk * gamma, i, j)
    return qc_p
# -------------------------------------------------------------------------------------------------------
def mixer(n, warm: bool, thetas, beta):
    ws_mixer = QuantumCircuit(n)
    if warm:
        for idx, theta in enumerate(thetas):
            ws_mixer.ry(-theta, idx)
            ws_mixer.rz(-2 * beta, idx)
            ws_mixer.ry(theta, idx)
    else:
        for i in range(n):
            ws_mixer.rx(2 * beta, i)
    return ws_mixer
# -------------------------------------------------------------------------------------------------------
def init(n, warm: bool, thetas):
    init_qc = QuantumCircuit(n)
    if warm:
        for idx, theta in enumerate(thetas):
            init_qc.ry(theta, idx)
    else:
        for i in range(n):
            init_qc.h(i)
    return init_qc
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Portfolio Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def QUBO_to_Ising(Q: Tensor) -> Tuple[Tensor, List[float], float]:
    """
    Cnvert the Q matrix into a the indication of pauli terms, the corresponding weights, and the offset.
    The outputs are used to construct an Ising Hamiltonian for QAOA.

    :param Q: The n-by-n square and symmetric Q-matrix.
    :return pauli_terms: A list of 0/1 series, where each element represents a Pauli term.
    A value of 1 indicates the presence of a Pauli-Z operator, while a value of 0 indicates its absence.
    :return weights: A list of weights corresponding to each Pauli term.
    :return offset: A float representing the offset term of the Ising Hamiltonian.
    """

    # input is n-by-n symmetric numpy array corresponding to Q-matrix
    # output is the components of Ising Hamiltonian

    n = Q.shape[0]

    # square matrix check
    if Q[0].shape[0] != n:
        raise ValueError("Matrix is not a square matrix.")

    offset = (
        np.triu(Q, 0).sum() / 2
    )  # Calculate the offset term of the Ising Hamiltonian
    pauli_terms = []  # List to store the Pauli terms
    weights = (
        -np.sum(Q, axis=1) / 2
    )  # Calculate the weights corresponding to each Pauli term

    for i in range(n):
        term = np.zeros(n)
        term[i] = 1
        pauli_terms.append(
            term.tolist()
        )  # Add a Pauli term corresponding to a single qubit

    for i in range(n - 1):
        for j in range(i + 1, n):
            term = np.zeros(n)
            term[i] = 1
            term[j] = 1
            pauli_terms.append(
                term.tolist()
            )  # Add a Pauli term corresponding to a two-qubit interaction

            weight = (
                Q[i][j] / 2
            )  # Calculate the weight for the two-qubit interaction term
            weights = np.concatenate(
                (weights, weight), axis=None
            )  # Add the weight to the weights list

    return pauli_terms, weights, offset
# -------------------------------------------------------------------------------------------------------
def QUBO_from_portfolio(cov: Array, mean: Array, q: float, B: int, t: float) -> Tensor:
    """
    convert portfolio parameters to a Q matrix
    :param cov: n-by-n covariance numpy array
    :param mean: numpy array of means
    :param q: the risk preference of investor
    :param B: budget
    :param t: penalty factor
    :return Q: n-by-n symmetric Q matrix
    """
    n = cov.shape[0]
    R = np.diag(mean)
    S = np.ones((n, n)) - 2 * B * np.diag(np.ones(n))

    Q = q * cov - R + t * S
    return Q
# -------------------------------------------------------------------------------------------------------
def print_Q_cost(Q, wrap=False, reverse=False):
    n_stocks = len(Q)
    states = []
    for i in range(2**n_stocks):
        a = f"{bin(i)[2:]:0>{n_stocks}}"
        n_ones = 0
        for j in a:
            if j == "1":
                n_ones += 1
        states.append(a)

    cost_dict = {}
    for selection in states:
        x = np.array([int(bit) for bit in selection])
        cost_dict[selection] = np.dot(x, np.dot(Q, x))
    cost_sorted = dict(sorted(cost_dict.items(), key=lambda item: item[1]))
    if reverse == True:
        cost_sorted = dict(
            sorted(cost_dict.items(), key=lambda item: item[1], reverse=True)
        )
    num = 0
    print("\n-------------------------------------")
    print("    selection\t  |\t  cost")
    print("-------------------------------------")
    for k, v in cost_sorted.items():
        print("%10s\t  |\t%.4f" % (k, v))
        num += 1
        if (num >= 8) & (wrap == True):
            break
    print("     ...\t  |\t  ...")
    print("-------------------------------------")
# -------------------------------------------------------------------------------------------------------
def energy_from_terms(bits: List[int],pauli_terms: List[List[float]],weights: List[float],offset: float = 0.0):
    """
    Compute E(bits) for H = sum_k w_k * Z^{(term_k)} + offset * I.
    bits is a list like [b0, b1, ..., b_{n-1}] with b in {0,1}.
    """
    z = 1 - 2 * np.asarray(bits, dtype=int)  # 0->+1, 1->-1
    E = float(offset)
    for w, term in zip(weights, pauli_terms):
        idx = np.flatnonzero(np.asarray(term, dtype=float) > 0.5)
        if len(idx) == 0:
            # identity term (not expected here, but handle gracefully)
            E += w
        elif len(idx) == 1:
            E += w * z[idx[0]]
        elif len(idx) == 2:
            E += w * (z[idx[0]] * z[idx[1]])
        else:
            # Supports higher-order Z products too:
            E += w * np.prod(z[idx])
    return float(E)
# -------------------------------------------------------------------------------------------------------
def ising_from_terms_qiskit(
    pauli_terms: List[List[float]],
    weights: List[float],
    offset: float = 0.0,
    reverse_qubit_order: bool = True,
) -> SparsePauliOp:
    """
    Build H = sum_k w_k * Z^{(term_k)} + offset * I from:
      - pauli_terms: list of 0/1
      - weights:    list/array of coefficients 
    Returns Qiskit SparsePauliOp.
    """
    T = np.asarray(pauli_terms, dtype=float)
    w = np.asarray(weights, dtype=float)

    if T.ndim != 2:
        raise ValueError("pauli_terms must be a 2D list/array of shape (m, n).")
    m, n = T.shape
    if w.shape[0] != m:
        raise ValueError(f"weights length {w.shape[0]} != number of terms {m}.")

    labels = []
    for row in T:
        mask = row > 0.5  # accept 1.0 or near-1.0; ignore tiny noise
        if reverse_qubit_order:
            mask = mask[::-1]
        label = ''.join('Z' if b else 'I' for b in mask)
        labels.append(label)

    op = SparsePauliOp.from_list(list(zip(labels, w.astype(complex))))
    if offset != 0.0:
        op += SparsePauliOp.from_list([('I' * n, complex(offset))])
    return op
# -------------------------------------------------------------------------------------------------------
def ground_energy_from_terms(
    pauli_terms: List[List[int]],
    weights: List[float],
    offset: float = 0.0,
) -> Tuple[float, List[int]]:
  
    pauli_terms = [list(t) for t in pauli_terms]
    weights = np.asarray(weights, dtype=float)
    n = len(pauli_terms[0])
    m = len(pauli_terms)

    num = 1 << n
    a = np.arange(num, dtype=np.uint64)
    shifts = np.arange(n, dtype=np.uint64)           
    bits = ((a[:, None] >> shifts) & np.uint64(1)).astype(np.int8)
    z = 1 - 2 * bits  # 0->+1, 1->-1

    # Precompute indices for each term
    term_indices = [np.flatnonzero(t) for t in pauli_terms]

    # Accumulate energies
    E = np.full(num, offset, dtype=float)
    for w, idx in zip(weights, term_indices):
        if len(idx) == 1:
            E += w * z[:, idx[0]]
        elif len(idx) == 2:
            i, j = int(idx[0]), int(idx[1])
            E += w * (z[:, i] * z[:, j])
        else:
            # If you ever had higher-order Z terms, this still works:
            # E += w * np.prod(z[:, idx], axis=1)
            raise ValueError("Only Z and ZZ terms are supported.")

    # Ground energy and a minimizer
    k_min = int(np.argmin(E))
    E0 = float(E[k_min])
    ground_bits = bits[k_min].tolist()  # [b0, b1, ..., b_{n-1}]
    return E0, ground_bits
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Script
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Financial Portfolio parameters
# -------------------------------------------------------------------------------------------------------
n = 6
mu, sigma = simulate_portfolio_gbm(n, 250, mu_range = (-0.05, 0.05), sigma_range = (-0.20, 0.20),  seed = iteration)
B = 3                                                             # Number of selected assets
l = 3                                                             # Lambda for penalty term
q = 2                                                             # Risk-return trade-off
# -------------------------------------------------------------------------------------------------------
# Creation of problem and cost operator
# -------------------------------------------------------------------------------------------------------
qp = create_problem(mu, sigma, B, q, l)
Q = QUBO_from_portfolio(sigma, mu, q, B, l)
portfolio_pauli_terms, portfolio_weights, portfolio_offset = QUBO_to_Ising(Q)
print_Q_cost(Q, wrap=True)
print(portfolio_offset)
print("Iteration:", iteration, ground_energy_from_terms(portfolio_pauli_terms, portfolio_weights, portfolio_offset))
print(portfolio_pauli_terms, portfolio_weights, portfolio_offset)
H = ising_from_terms_qiskit(portfolio_pauli_terms, portfolio_weights, 0)
# -------------------------------------------------------------------------------------------------------
# Classical solution using CPLEX
# -------------------------------------------------------------------------------------------------------
sol= CplexOptimizer().solve(qp)
c_stars = sol.samples[0].x
print(c_stars, flush=True)
# -------------------------------------------------------------------------------------------------------
# Ground state energy 
# -------------------------------------------------------------------------------------------------------
solver_q = NumPyMinimumEigensolver()
res_q = solver_q.compute_minimum_eigenvalue(H)
E0_q = res_q.eigenvalue.real 
print("Qiskit E0:", E0_q, flush = True)
print("Portfolio offset:", portfolio_offset, flush = True)
# -------------------------------------------------------------------------------------------------------
# Relax problem and solve classically using CPLEX
# -------------------------------------------------------------------------------------------------------
qp_r = relax_problem(qp)
sol_r= CplexOptimizer().solve(qp_r)
c_stars_r = sol_r.samples[0].x
print(c_stars_r, flush=True)
# -------------------------------------------------------------------------------------------------------
# Warm-start QAOA and Cold-start QAOA
# -------------------------------------------------------------------------------------------------------
depth = 1
eps = 0
parms = create_custom_array_even_odd(2*depth, -2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi)
pw, ew = solve_hm(n, True, False, H, portfolio_pauli_terms, portfolio_weights, c_stars_r, eps, depth, iteration, c_stars, parms)
pc, ec = solve_hm(n, False, False, H, portfolio_pauli_terms, portfolio_weights, c_stars_r, eps, depth, iteration, c_stars, parms)
print("Ew:", ew , flush = True)
print("Ew - Portfolio offset:", ew - portfolio_offset, flush = True)
x = np.array(c_stars)
y = np.array(c_stars_r)
c_prod=(float(np.dot(x, y))/B)
# -------------------------------------------------------------------------------------------------------
# Out .jsons
# -------------------------------------------------------------------------------------------------------
out = {
    "iteration": iteration,
    "Ew/E0": (ew)/E0_q,
    "Ec/E0": (ec)/E0_q,
    "Delta_frac": (ew-E0_q)/(ec-E0_q),
    "cprod": c_prod,
    "E0": E0_q,
}
out_dir = Path("Results2")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"results_i{iteration}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------