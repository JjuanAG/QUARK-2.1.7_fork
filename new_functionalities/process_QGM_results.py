import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from pathlib import Path

# Data visualization: Still does not work properly and needs refinement

class QGMDataReader:
    """
    A class to read and visualize QUARK data from .npy and .pkl files.
    """
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.data = None

    def _load_data(self, data_path: str):
        """
        Loads data from the specified file path.
        """
        file_path = os.path.join(self.dir_path, data_path)

        if not os.path.exists(file_path):
            print(f"Error: File not found at {os.path.abspath(file_path)}")
            return None

        _, ext = os.path.splitext(file_path)

        try:
            if ext == '.npy':
                self.data = np.load(file_path, allow_pickle=True)
            elif ext == '.pkl':
                with open(file_path, 'rb') as f:
                    self.data = pickle.load(f)
            else:
                print(f"Unsupported format: {ext}")
                return None
        except Exception as e:
            print(f"Failed to load: {e}")
            return None
    
    def console_print(self, data_path: str):
        """
        Dumps the raw contents of the specified file.
        """
        self._load_data(data_path)
        print(f"RAW CONTENTS: {data_path}")
        print(f"Path: {os.path.abspath(os.path.join(self.dir_path, data_path))}")
        
        if self.data is None:
            print("No data loaded.")
            return
        
        print("TYPE:", type(self.data))

        if isinstance(self.data, pd.DataFrame):
            print("SHAPE:", self.data.shape)
            print("COLUMNS:", list(self.data.columns))
            print("\nRAW HEAD (first 10 rows):")
            print(f"{self.data.head(10).to_dict('records')}\n")

        elif isinstance(self.data, np.ndarray):
            print("SHAPE:", self.data.shape)
            print("DTYPE:", self.data.dtype)
            print("\nRAW HEAD (first 10 elements/rows):")
            print(f"{self.data[:10]}\n")
        else:
            print("RAW DATA:")
            print(f"{repr(self.data)}\n")

    def console_print_multi_files(self, data_paths: list[str]):
        """
        Dumps the raw contents of multiple specified files.
        """
        counter = 1
        for data_path in data_paths:
            print(f"\n# {'-'*78}")
            print(f"console print of file number {counter} of {len(data_paths)}")
            self.console_print(data_path)
            counter += 1

######
dir_path_itwm = r"\\itwm\u\g\garciabetancour\QUARK-2.1.7_fork\benchmark_runs\generativemodeling_sweep_run_2026-08-05-17-13-30\volumetric_aer_simulator_gpu_vs_aer_simulator_cpu\aer_simulator_cpu\comb_idx_16\benchmark_0\rep_1_Cardinality_Constraint_qubits6"
files = [
    r"best_parameters_1.npy",
    r"data_1.pkl",
    r"histogram_generated.npy",
    r"histogram_train.npy",
    r"record_gen_metrics_1.pkl",
    r"training_results-1.pkl"
]

QGMDataReader(dir_path_itwm).console_print_multi_files(files)