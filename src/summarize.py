"""
Output files:
- mps_summary.csv : Op counts, depth, transpilation times, simulation times for matrix_product_state
- statevector_cpu_summary.csv : Op counts, depth, transpilation times, simulation times for statvector
- figures/
    - Circuit depth vs transpilation time
    - Dimension vs transpilation time and simulation time
    - Circuit depth before/after transpilation
"""

import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


dimension_header = "Dimension"
num_row_mismatches_header = "Number of row mismatches"
transpilation_time_header = "Transpilation time (s)"
simulation_time_header = "Simulation time (s)"
qubit_count_header = "Qubit count"
num_grover_iterations_header = "Number of Grover iterations"
circuit_depth_before_transpilation_header = "Circuit depth (before transpilation)"
circuit_depth_after_transpilation_header = "Circuit depth (after transpilation)"

BLUE_COLOR = "#636AF6"
RED_COLOR = "#EE706B"

def df_of_files(files):
    df = defaultdict(list)

    for file in files:
        with open(file) as f:
            data = json.load(f)
            
            nrows, ncols, num_mismatches = result_parts_of_str(file)
            
            df[dimension_header].append((nrows, ncols))
            df[num_row_mismatches_header].append(num_mismatches)
            df[transpilation_time_header].append(data["transpilation_time"])
            df[simulation_time_header].append(data["simulation_time"])
            df[qubit_count_header].append(data["qubit_count"])
            df[num_grover_iterations_header].append(data["num_grover_iterations"])
            df[circuit_depth_before_transpilation_header].append(data["circuit_depth_before_transpilation"])
            df[circuit_depth_after_transpilation_header].append(data["circuit_depth_after_transpilation"])

    return pd.DataFrame.from_dict(df)


def plot_circuit_depth_v_tran_time_and_sim_time(df, name, output_filename):
    df = df[[circuit_depth_before_transpilation_header
        , transpilation_time_header
        , simulation_time_header]].sort_values(by=circuit_depth_before_transpilation_header)
    
    fig, ax = plt.subplots()
    ax.plot(
        df[circuit_depth_before_transpilation_header],
        df[transpilation_time_header],
        ls='-',
        label="Transpilation time",
        color=BLUE_COLOR
    )
    ax.plot(
        df[circuit_depth_before_transpilation_header],
        df[simulation_time_header],
        ls='-',
        label="Simulation time",
        color=RED_COLOR
    )
    ax.legend()
    ax.set(xlabel="Circuit depth (before transpilation)", ylabel="Time (s)")
    
    plt.savefig(output_filename)


def plot_dimension_vs_transpilation_time_and_simulation_time(df, name, output_filename):
    labels = [(4,4,1), (16,4,2), (16,16,2), (32,4,3), (32,16,1), (64,32,1), (64,64,3)]
    
    def helper(*x):
        (nrows, ncols) = x[0]
        num_mismatches = x[1]
        return (nrows, ncols, num_mismatches) in labels

    filter = df[[
        dimension_header,
        num_row_mismatches_header
    ]].apply(lambda x: helper(*x), axis=1)

    df = df[filter]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, df[transpilation_time_header], width, label="Transpilation time", color=BLUE_COLOR)
    ax.bar(x + width/2, df[simulation_time_header], width, label="Simulation time", color=RED_COLOR)

    ax.set_ylabel("Time (s)")
    ax.set_xlabel("(#rows, #cols, #row mismatches)")
    ax.set_xticks(x, labels)
    ax.legend()
    
    plt.savefig(output_filename)


def plot_circuit_depth_before_after_transpilation(df, name, output_filename):
    labels = [(16,4,2), (16,8,2), (16,16,2)]
    
    def helper(*x):
        (nrows, ncols) = x[0]
        num_mismatches = x[1]
        return (nrows, ncols, num_mismatches) in labels

    filter = df[[
        dimension_header,
        num_row_mismatches_header
    ]].apply(lambda x: helper(*x), axis=1)

    df = df[filter]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(
        x - width/2,
        df[circuit_depth_before_transpilation_header],
        width,
        label=circuit_depth_before_transpilation_header,
        color=BLUE_COLOR
    )
    ax.bar(
        x + width/2,
        df[circuit_depth_after_transpilation_header],
        width,
        label=circuit_depth_after_transpilation_header,
        color=RED_COLOR
    )

    ax.set_ylabel("Depth")
    ax.set_xlabel("(#rows, #cols, #row mismatches)")
    ax.set_xticks(x, labels)
    ax.legend()
    
    plt.savefig(output_filename)


def gate_counts_df_of_files(files, gates):
    df = defaultdict(list)

    for file in files:
        with open(file) as f:
            data = json.load(f)

            nrows, ncols, num_mismatches = result_parts_of_str(file)
            df[dimension_header].append((nrows, ncols))
            df[num_row_mismatches_header].append(num_mismatches)

            gate_counts_after_transpilation = data["gate_counts_after_transpilation"]

            for gate in gates:
                if gate in gate_counts_after_transpilation:
                    df[gate].append(gate_counts_after_transpilation[gate])
                else:
                    df[gate].append(0)
    
    return pd.DataFrame.from_dict(df)


if __name__ == "__main__":
    def result_parts_of_str(s):
        path = Path(s)
        result_name_parts = path.parts[1].split("-")
        dims_str, num_mismatches_str = (result_name_parts[0]).split("x"), result_name_parts[1]
        return int(dims_str[0]), int(dims_str[1]), int(num_mismatches_str)

    def sort_helper(path_str):
        return result_parts_of_str(path_str)

    mps_files = sorted(glob.glob("results/**/matrix_product_state-CPU.json"), key=sort_helper)
    statevector_cpu_files = sorted(glob.glob("results/**/statevector-CPU.json"), key=sort_helper)

    # Generate summaries
    summary_workload = [(mps_files, "mps"), (statevector_cpu_files, "statevector_cpu")]
    gates = ["ccx", "cx", "x", "h", "z", "u1", "u2", "u3"]
    overall_dfs = {}

    for task in summary_workload:
        (files, name) = task
        overall_df = df_of_files(files)
        circuit_metrics_df = gate_counts_df_of_files(files, gates)
        circuit_metrics_df[circuit_depth_after_transpilation_header] = overall_df[circuit_depth_after_transpilation_header]
        circuit_metrics_df[qubit_count_header] = overall_df[qubit_count_header]
        circuit_metrics_df["Total Gate count"] = circuit_metrics_df[gates].sum(axis=1)

        overall_df.to_csv(f"results/{name}_summary.csv")
        circuit_metrics_df.to_csv(f"results/{name}_circuit_metrics_summary.csv")

        overall_dfs[name] = overall_df

    # Generate plots
    plot_circuit_depth_v_tran_time_and_sim_time(
        overall_dfs["mps"],
        "Circuit depth vs Transpilation/Simulation time for the MPS simulation method",
        "results/circuit_depth_v_tran_time_and_sim_time-MPS.pdf"
    )
    plot_circuit_depth_v_tran_time_and_sim_time(
        overall_dfs["statevector_cpu"],
        "Circuit depth vs Transpilation/Simulation time for the Statevector simulation method",
        "results/circuit_depth_v_tran_time_and_sim_time-statevector_cpu.pdf"
    )
    plot_circuit_depth_before_after_transpilation(
        overall_dfs["mps"],
        "Circuit depth before/after transpilation of matrices with 16 rows",
        "results/circuit_depth_before_after_transpilation.pdf"
    )
