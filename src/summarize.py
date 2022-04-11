"""
Output files:
- mps_summary.csv : Op counts, depth, transpilation times, simulation times for matrix_product_state
- statevector_cpu_summary.csv : Op counts, depth, transpilation times, simulation times for statvectore
"""

import glob
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def df_of_files(files):
    df = defaultdict(list)

    for file in files:
        with open(file) as f:
            data = json.load(f)
            
            nrows, ncols, num_mismatches = result_parts_of_str(file)
            
            df["Dimension"].append((nrows, ncols))
            df["Number of row mismatches"].append(num_mismatches)
            df["Transpilation time (s)"].append(data["transpilation_time"])
            df["Simulation time (s)"].append(data["simulation_time"])
            df["Qubit count"].append(data["qubit_count"])
            df["Number of Grover iterations"].append(data["num_grover_iterations"])
            df["Circuit depth (before transpilation)"].append(data["circuit_depth_before_transpilation"])
            df["Circuit depth (after transpilation)"].append(data["circuit_depth_after_transpilation"])

    return pd.DataFrame.from_dict(df)

if __name__ == "__main__":
    def result_parts_of_str(s):
        path = Path(s)
        result_name_parts = path.parts[1].split("-")
        dims_str, num_mismatches_str = (result_name_parts[0]).split("x"), result_name_parts[1]
        return int(dims_str[0]), int(dims_str[1]), int(num_mismatches_str)

    def sort_helper(path_str):
        return result_parts_of_str(path_str)

    mps_stat_files = sorted(glob.glob("results/**/matrix_product_state-CPU.json"), key=sort_helper)
    statevector_cpu_files = sorted(glob.glob("results/**/statevector-CPU.json"), key=sort_helper)

    
    mps_stat_df = df_of_files(mps_stat_files) 
    statevector_cpu_df = df_of_files(statevector_cpu_files)

    mps_stat_df.to_csv("results/mps_summary.csv")
    statevector_cpu_df.to_csv("results/statevector_cpu_summary.csv")
