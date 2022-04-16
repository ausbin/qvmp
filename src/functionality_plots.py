from .qvmp_grover import *
from qiskit import Aer, assemble, transpile
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms import Grover
from pathlib import Path


def run_experiment(dims, num_incorrect, seed, is_known, simulation_method="statevector"):
    nrows, ncols = dims
    A, y, z, idx = gen_input(dims, num_incorrect, seed=seed)

    if is_known:
        qc, num_iterations = qvmp_grover_submatrix(A, y, z, num_incorrect)
    else:
        qc, num_iterations = qvmp_grover_submatrix(
            A, y, z,
            num_incorrect,
            num_iterations=math.floor(nrows ** 0.5),
            with_amplitude_amplification=True
        )

    backend = AerSimulator(method=simulation_method)
    qobj = assemble(transpile(qc, backend))
    result = backend.run(qobj).result()

    addr_size = math.ceil(math.log2(nrows))
    optimal_num_iterations = Grover.optimal_num_iterations(num_incorrect, addr_size)

    return result, idx, num_iterations, optimal_num_iterations


if __name__ == "__main__":
    results_dir = Path("../results")
    figures_dir = results_dir / "figures"

    seed = 42

    # fig:qvmp_functionality_found_known
    result, _, _, _ = run_experiment((8,8), 3, seed, is_known=True)
    plot_histogram(result.get_counts()).savefig(figures_dir / "qvmp_functionality_found_known.pdf")
    
    # fig:qvmp_functionality_found_unknown
    result, _, _, _ = run_experiment((8,8), 3, seed, is_known=False)
    plot_histogram(result.get_counts()).savefig(figures_dir / "qvmp_functionality_found_unknown.pdf")
    
    # fig:qvmp_functionality_none_known
    result, _, _, _ = run_experiment((8,8), 0, seed, is_known=True)
    plot_histogram(result.get_counts()).savefig(figures_dir / "qvmp_functionality_none_known.pdf")
    
    # fig:qvmp_functionality_none_unknown
    result, _, _, _ = run_experiment((8,8), 0, seed, is_known=False)
    plot_histogram(result.get_counts()).savefig(figures_dir / "qvmp_functionality_none_unknown.pdf")
    
    # fig:qvmp_functionality_pfound_known__1
    result, idx, num_iterations, optimal_num_iterations = run_experiment((16,16), 3, seed, is_known=True, simulation_method="matrix_product_state")
    plot_histogram(result.get_counts(), bar_labels=False).savefig(
        figures_dir / "qvmp_functionality_pfound_known__1.pdf")
    print("pfound_known__1 idx:", idx)
    print("pfound_known__1 num_iterations:", num_iterations)
    print("pfound_known__1 num_iterations (optimal):", optimal_num_iterations)
    
    # fig:qvmp_functionality_pfound_unknown__1
    result, idx, num_iterations, optimal_num_iterations = run_experiment((16,16), 3, seed, is_known=False, simulation_method="matrix_product_state")
    plot_histogram(result.get_counts()).savefig(figures_dir / "qvmp_functionality_pfound_unknown__1.pdf")
    print("pfound_known__1 idx:", idx)
    print("pfound_known__1 num_iterations:", num_iterations)
    print("pfound_known__1 num_iterations (optimal):", optimal_num_iterations)

    
    # fig:qvmp_functionality_pfound_known__2
    result, idx, num_iterations, optimal_num_iterations = run_experiment((8,8), 5, seed, is_known=True)
    plot_histogram(result.get_counts()).savefig(figures_dir / "qvmp_functionality_pfound_known__2.pdf")
    print("pfound_known__2 idx:", idx)
    print("pfound_known__2 num_iterations:", num_iterations)
    print("pfound_known__2 num_iterations (optimal):", optimal_num_iterations)

    # fig:qvmp_functionality_pfound_unknown__2
    result, idx, num_iterations, optimal_num_iterations = run_experiment((8,8), 5, seed, is_known=False)
    plot_histogram(result.get_counts()).savefig(figures_dir / "qvmp_functionality_pfound_unknown__2.pdf")
    print("pfound_unknown__2 idx:", idx)
    print("pfound_unknown__2 num_iterations:", num_iterations)
    print("pfound_unknown__2 num_iterations (optimal):", optimal_num_iterations)
