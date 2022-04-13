from qvmp_grover import *
import pickle
import timeit
import json
from pathlib import Path
from multiprocessing import Process
import numpy as np
import sys

def get_results(dims, num_incorrect, num_solutions_are_known, output_dir, timeit_repeat=5, timeit_number=1):
    nrows, ncols = dims
    if num_solutions_are_known:
        results_dir = output_dir / f"{nrows}x{ncols}-{num_incorrect}-known"
    else:
        results_dir = output_dir / f"{nrows}x{ncols}-{num_incorrect}-unknown"

    if not results_dir.exists():
        results_dir.mkdir()

    A, y, z, idx = gen_input(dims, num_incorrect)

    if num_solutions_are_known:
        qc, num_iterations = qvmp_grover_submatrix(A, y, z, num_incorrect)
    else:
        # TODO
        num_iterations = dims[1]
        qc, num_iterations = qvmp_grover_submatrix(A, y, z, num_incorrect,
                                                   num_iterations, with_amplitude_amplification=True) 

    if qc.num_qubits > 32:
        method_device_tuples = [("matrix_product_state", "CPU")]
    else:
        method_device_tuples = [("statevector", "CPU"), ("matrix_product_state", "CPU")]

    stats = {}
    for (method, device) in method_device_tuples:
        method_device_key = f"{method}-{device}"
        stats = {}

        backend = AerSimulator(method=method, device=device)

        transpiled_qc = transpile(qc, backend)

        qobj = assemble(transpiled_qc)
        with open(results_dir / "qobj.pickle", "wb") as f:
            pickle.dump(qobj, f)

        result = backend.run(qobj).result()

        with open(results_dir / "result.pickle", "wb") as f:
            pickle.dump(result, f)

        transpilation_times = timeit.repeat(lambda: transpile(qc, backend),
                repeat=timeit_repeat, number=timeit_number)
        simulation_times = timeit.repeat(lambda: backend.run(qobj).result(),
                repeat=timeit_repeat, number=timeit_number)

        decomposed_qc = qc.decompose()
        stats["transpilation_time"] = np.mean(transpilation_times)
        stats["simulation_time"] = np.mean(simulation_times)
        stats["gate_counts_before_transpilation"] = decomposed_qc.count_ops()
        stats["gate_counts_after_transpilation"] = transpiled_qc.count_ops()
        stats["circuit_depth_before_transpilation"] = decomposed_qc.depth()
        stats["circuit_depth_after_transpilation"] = transpiled_qc.depth()
        stats["qubit_count"] = qc.num_qubits
        stats["num_grover_iterations"] = num_iterations

        with open(results_dir / f"{method_device_key}.json", "w") as f:
            json.dump(stats, f)

        print(f"dims={dims}, num_incorrect={num_incorrect}, method={method}, device={device}")

 
if __name__ == "__main__":
    results_dir = Path("../results")
    if not results_dir.exists():
        results_dir.mkdir()

    dimensions = [(4, 2), (4, 4), (16, 4), (16, 8), (16, 16), (32, 4), (32, 8), 
                 (32, 16), (32, 32), (64, 8), (64, 16), (64, 32), (64, 64)]
    num_incorrect_rows = [0, 1, 2, 3]

    handles = []
    for dim in dimensions:
        for num_solns in num_incorrect_rows:
            p = Process(
                target=get_results,
                args=(dim, num_solns, True, results_dir)
            )
            p.start()
            handles.append(p)

    for handle in handles:
        handle.join()
