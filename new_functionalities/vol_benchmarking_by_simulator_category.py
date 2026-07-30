import os
import json
import pickle
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class Runtime:
    def __init__(self, time, unit):
        self.time = time
        self.unit = unit
    
    def __repr__(self):
        return f"{self.time} {self.unit}"
    

class QGMDataExtractor:
    """Class to extract data from benchmark runs for volumetric benchmarking of generative quantum modeling applications."""
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.extract_run_data_for_config_group_was_called = False

    def get_config_parameters(self, file_path: str) -> dict:
        """Parses a YAML configuration file to extract module parameters into a dictionary.

        Recursively traverses the hierarchical 'application' structure (and its 'submodules')
        to create a flat dictionary mapping each module's name to its configuration key-value
        pairs.
        Args:
            file_path (str): The local path to the target .yml or .yaml configuration file.
        Returns:
            dict: A dictionary where keys are module names (e.g., 'LibraryQiskit') and
                values are dictionaries of their parameters. Returns None if the
                file does not exist, uses an unsupported extension, or fails to parse.
        Example:
            >>> config = obj.get_config_parameters("config.yml")
            >>> print(config)
            {
                'GenerativeModeling': {'n_qubits': 6},
                'LibraryQiskit': {'backend': 'aer_statevector_simulator_cpu', 'n_shots': 100}
            }
        """

        if not os.path.exists(file_path):
            print(f"Error: File not found at {os.path.abspath(file_path)}")
            return None

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            if ext in ['.yml', '.yaml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            else:
                print(f"Unsupported format: {ext}")
                return None

        except Exception as e:
            print(f"Failed to load: {e}")
            return None

        result = {}

        def extract_module(node: dict):
            if not isinstance(node, dict):
                return

            name = node.get("name")
            config = node.get("config") or {}  # Handles cases where config is explicitly None/null

            if name:
                # Clean up config values (unpack single-element lists)
                cleaned_config = {}
                for key, val in config.items():
                    # Unpack single-element lists
                    if isinstance(val, list) and len(val) == 1:
                        val = val[0]
                    
                    # Convert string 'False'/'True' to actual booleans if present
                    if val == 'False':
                        val = False
                    elif val == 'True':
                        val = True
                    
                    cleaned_config[key] = val

                result[name] = cleaned_config

            # Recursively process submodules
            submodules = node.get("submodules") or []
            for submodule in submodules:
                extract_module(submodule)

        # Start extraction from top-level application structure
        if isinstance(data, dict) and "application" in data:
            extract_module(data["application"])
        else:
            extract_module(data)

        return result

    def get_precision(self, file_path):
        if not os.path.exists(file_path):
            print(f"Error: File not found at {os.path.abspath(file_path)}")
            return None

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            if ext == '.pkl':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                print(f"Unsupported format: {ext}")
                return None
        except Exception as e:
            print(f"Failed to load: {e}")
            return None
        
        precision = data.get('precision', None) if isinstance(data, dict) else None

        return precision

    # TODO: Make this function more flexible because the loss function may not be KL and for example NNL
    def get_KL_best(self, file_path):
        """"Extracts the best KL divergence value from the results.json file of a benchmark run. Returns the best KL divergence value."""
        if not os.path.exists(file_path):
            print(f"Error: File not found at {os.path.abspath(file_path)}")
            return None

        root, ext = os.path.splitext(file_path)

        try:
            if ext == '.json':
                with open(file_path, 'r') as file:
                    results = json.load(file)[0]  # returns a dictionary
            else:
                print(f"Unsupported format: {ext}")
                return None

        except Exception as e:
            print(f"Failed to load: {e}")
            return None

        # --- Extract KL best ---

        # 2. Recursively go into every submodule and extract the runtimes
        def extract_KL_best(current_node):
            # Base case: safe guard against empty/non-dict nodes
            if not isinstance(current_node, dict) or not current_node:
                print("Warning: Reached an empty or non-dictionary node while searching for 'KL_best'. This path will be skipped.")
                return None
                
            # 1. Check if we found it
            if "KL_best" in current_node:
                return current_node["KL_best"]
                
            # 2. Check "module" branch explicitly
            if "module" in current_node:
                result = extract_KL_best(current_node["module"])
                if result is not None:
                    return result # Found it down this path! Bubble it up.

            # 3. Check "submodule" branch explicitly
            if "submodule" in current_node:
                result = extract_KL_best(current_node["submodule"])
                if result is not None:
                    return result # Found it down this path! Bubble it up.
            
            # Reached a dead end leaf node
            print("Warning: 'KL_best' not found in this branch of the results tree. This path will be skipped.")
            return None
        
        KL_best = extract_KL_best(results)

        return KL_best[0]
    
    def get_probability_distribution(self, file_path):
        if not os.path.exists(file_path):
            print(f"Error: File not found at {os.path.abspath(file_path)}")
            return None

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            if ext == '.npy':
                data = np.load(file_path, allow_pickle=True)
            else:
                print(f"Unsupported format: {ext}")
                return None
        except Exception as e:
            print(f"Failed to load: {e}")
            return None
        
        probability_distribution = np.asarray(data)

        return probability_distribution
    
    def get_runtimes(self, file_path):
        """Extracts runtimes from the results.json file of a benchmark run. Returns a dictionary with the runtimes for each module and submodule, as well as the overall runtime."""

        if not os.path.exists(file_path):
            print(f"Error: File not found at {os.path.abspath(file_path)}")
            return None

        root, ext = os.path.splitext(file_path)

        try:
            if ext == '.json':
                with open(file_path, 'r') as file:
                    results = json.load(file)[0]  # returns a dictionary
            else:
                print(f"Unsupported format: {ext}")
                return None

        except Exception as e:
            print(f"Failed to load: {e}")
            return None

        # --- Extract run times ---

        # 1. Total run time
        runtimes = {"all_modules": {"total_time": Runtime(results.get("total_time"), results.get("total_time_unit"))}}

        # 2. Recursively go into every submodule and extract the runtimes
        def extract_module_times(current_node):
            # Base case: if distionry is empty, return None
            if not current_node:
                return None
                
            # Check if this current layer is the one we want
            if "module_name" in current_node:
                runtimes.update({current_node["module_name"]: 
                {
                    "total_time": Runtime(current_node["total_time"], current_node["total_time_unit"]),
                    "preprocessing_time": Runtime(current_node.get("preprocessing_time"), current_node.get("preprocessing_time_unit")),
                    "postprocessing_time": Runtime(current_node.get("postprocessing_time"), current_node.get("postprocessing_time_unit"))
                    }
                     })
                
            # If not, check if there is a 'module' or 'submodule' layer to dive into
            next_layer = current_node.get("module") or current_node.get("submodule")
            if next_layer:
                return extract_module_times(next_layer)

            # In case there is no module or submodule layer, we have reached a leaf node without finding the module_name, return None    
            return None
        
        extract_module_times(results)

        return runtimes

    # TODO: Note that if there are two or more runs that have the exact same number of qubits and circuit depth, the function will throw a warning 
    def extract_run_data_for_config_group(self, simulator_path_name, print_results=False, print_constant_config=False):
        # Setup the target path       
        target_dir = self.base_path / simulator_path_name
        
        if not target_dir.exists():
            print(f"Directory not found: {target_dir}")
            return [], {}

        all_run_results = []
        global_constant_config = None
        consistency_error = False

        # Track (n_qubits, depth) to detect duplicates
        seen_combinations = {}

        # Iterate through all folders starting with 'generativemodeling'
        for gen_folder in target_dir.iterdir():
            if gen_folder.is_dir():
            
                # Locate required files based on image_e02a1d.png structure
                config_file = next(gen_folder.glob("config.yml"), None)
                results_file = next(gen_folder.glob("results.json"), None)
                # Using rglob for nested files in benchmark_0/rep_1 subfolders
                metrics_file = next(gen_folder.rglob("record_gen_metrics*.pkl"), None)
                histogram_file = next(gen_folder.rglob("histogram_generated.npy"), None)

                if not all([config_file, results_file, metrics_file, histogram_file]):
                    print(f"Skipping {gen_folder.name}: Missing one or more required files.")
                    continue

                # --- Data extraction for each run ---
                config_data = self.get_config_parameters(str(config_file))
                runtimes_data = self.get_runtimes(str(results_file))
                precision_data = self.get_precision(str(metrics_file))
                pmf_data = self.get_probability_distribution(str(histogram_file))
                kl_best = self.get_KL_best(str(results_file))

                if config_data is None:
                    continue
                else:
                    # Flatten the config_data dictionary
                    flat_config_data = {key: val for sub_dict in config_data.values() for key, val in sub_dict.items()}


                # --- Duplicate run check ---
                current_qubits = flat_config_data.get('n_qubits')
                current_depth = flat_config_data.get('depth')
                combination = (current_qubits, current_depth)

                if combination in seen_combinations and not self.extract_run_data_for_config_group_was_called:
                    print(f"WARNING: Two runs found with the exact same number of qubits ({current_qubits}) "
                        f"and circuit depth ({current_depth}) in parent folder {target_dir.name}. First found in: {seen_combinations[combination]}, duplicate found in: {gen_folder.name}")
                else:
                    seen_combinations[combination] = gen_folder.name

                # --- CONFIG CONSISTENCY CHECK ---
                # Extract constants (everything EXCEPT n_qubits and depth)
                current_constants = {k: v for k, v in flat_config_data.items() if k not in ['n_qubits', 'depth']}
                
                if global_constant_config is None:
                    # First folder sets the baseline for constants
                    global_constant_config = current_constants
                else:
                    # Compare current constants to the baseline
                    if current_constants != global_constant_config:
                        print(f"Warning: Mismatch of constant configuration parameters in folder {gen_folder.name}")
                        consistency_error = True

                # --- RESTRUCTURE RUN DATA ---
                run_entry = {
                    'n_qubits': current_qubits,
                    'circuit_depth': current_depth,
                    'precision': precision_data,
                    'pmf': pmf_data,
                    'runtimes': runtimes_data,
                    'KL_best': kl_best
                }
                all_run_results.append(run_entry)

        if consistency_error:
            print("Note: Some benchmark runs had differing constant parameters. Check logs above.")
            return None

        # Printing results for verification
        if print_results:
            for idx, run in enumerate(all_run_results):
                print(f"Benchmark run {idx+1}: {all_run_results[idx]}")
        
        if print_constant_config:
            print(f"Data from benchmark runs in directory: {target_dir} extracted successfully.")
            print(f"Number of runs processed: {len(all_run_results)} with constant configuration parameters:" )
            print("Global Constant Config:", global_constant_config)
            print()

        self.extract_run_data_for_config_group_was_called = True  # Set the flag to True after the first call to prevent duplicate warnings in subsequent calls
       
        return all_run_results, global_constant_config


class VolBenchBySimulatorCategory:
    """Class to perform volumetric benchmarking comparisons between different simulator categories (e.g., noisy vs noise-free)."""
    def __init__(self, qgm_data_object, notnoisy_simulator_path, noisy_simulator_path):
        self.qgm_data_object = qgm_data_object
        self.notnoisy_simulator_path = notnoisy_simulator_path
        self.noisy_simulator_path = noisy_simulator_path

    def check_for_compatible_config_groups_across_simulators(self):
        notnoisy_data_list, notnoisy_constants = self.qgm_data_object.extract_run_data_for_config_group(self.notnoisy_simulator_path, print_results=False, print_constant_config=True)
        noisy_data_list, noisy_constants = self.qgm_data_object.extract_run_data_for_config_group(self.noisy_simulator_path, print_results=False, print_constant_config=True)
        noisy_constants.pop('backend')
        notnoisy_constants.pop('backend')
        # Extract the number of qubits and circuit depth for each run in both lists
        n_qubits_list_notnoisy = [notnoisy_data_list[idx]['n_qubits'] for idx, _ in enumerate(notnoisy_data_list)]
        circuit_depth_list_notnoisy = [notnoisy_data_list[idx]['circuit_depth'] for idx, _ in enumerate(notnoisy_data_list)]
        n_qubits_list_noisy = [noisy_data_list[idx]['n_qubits'] for idx, _ in enumerate(noisy_data_list)]
        circuit_depth_list_noisy = [noisy_data_list[idx]['circuit_depth'] for idx, _ in enumerate(noisy_data_list)]

        if len(notnoisy_data_list) != len(noisy_data_list):
            print("Error: The number of runs with the not-noisy and noisy simulators must be equal.")
            return

        elif notnoisy_constants != noisy_constants:
            print("Error: The constant configuration parameters do not match between the not-noisy and noisy simulators.")
            print("Not-noisy constants:", notnoisy_constants)
            print("Noisy constants:", noisy_constants)
            return False
        elif set(n_qubits_list_notnoisy) != set(n_qubits_list_noisy) or set(circuit_depth_list_notnoisy) != set(circuit_depth_list_noisy):
            print("Error: The variable configuration parameters (n_qubits and circuit_depth) do not match between the not-noisy and noisy simulators.")
            print("Not-noisy n_qubits:", n_qubits_list_notnoisy)
            print("Not-noisy circuit_depth:", circuit_depth_list_notnoisy)
            print("Noisy n_qubits:", n_qubits_list_noisy)
            print("Noisy circuit_depth:", circuit_depth_list_noisy)
            return False

        print("Success: The variable configuration parameters match between the not-noisy and noisy simulators.")
        print("Not-noisy n_qubits:", n_qubits_list_notnoisy)
        print("Not-noisy circuit_depth:", circuit_depth_list_notnoisy)
        print("Noisy n_qubits:", n_qubits_list_noisy)
        print("Noisy circuit_depth:", circuit_depth_list_noisy)
        return True


    def noisy_notnoisy_precision_comparison(self): 
        # Run compatibility check
        self.check_for_compatible_config_groups_across_simulators()

        # Extract data (Unpacking the tuple: data_list, constant_config)
        notnoisy_data_list, notnoisy_constant_config = self.qgm_data_object.extract_run_data_for_config_group(self.notnoisy_simulator_path)
        noisy_data_list, noisy_constant_config = self.qgm_data_object.extract_run_data_for_config_group(self.noisy_simulator_path)

        # Helper function to extract and sort axes data
        def prepare_plot_vectors(data_list):
            # Sorting ensures the "curve" connects points in a logical order
            sorted_data = sorted(data_list, key=lambda x: (x['n_qubits'], x['circuit_depth']))

            if not sorted_data:
                print("Warning: No data available to plot.")
                return [], [], []

            x = [d['n_qubits'] for d in sorted_data]
            y = [d['circuit_depth'] for d in sorted_data]
            z = [d['precision'] for d in sorted_data]
            return x, y, z

        # Prepare vectors for both simulators
        x_free, y_free, z_free = prepare_plot_vectors(notnoisy_data_list)  # n_qubits, circuit_depth, precision for noise-free simulator
        x_noisy, y_noisy, z_noisy = prepare_plot_vectors(noisy_data_list)  # n_qubits, circuit_depth, precision for noisy simulator

        # Calculate precision difference between the two simulators for each corresponding run (they are in the same order after sorting)
        precision_diff = [zf - zn for zf, zn in zip(z_free, z_noisy)]
        precision_diff_per_nqubit_depth = {(x, y): diff for x, y, diff in zip(x_free, y_free, precision_diff)}

        # Restructure dictionary into a sorted matrix DataFrame
        df_data = [{"n_qubits": q, "circuit_depth": d, "diff": diff} for (q, d), diff in precision_diff_per_nqubit_depth.items()]
        df = pd.DataFrame(df_data)
        matrix_df = df.pivot(index="circuit_depth", columns="n_qubits", values="diff")  # turn into 2D matrix with circuit_depth as rows and n_qubits as columns
        matrix_df = matrix_df.sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)  # .sort_index(axis=0, ascending=True) sorts the rows (axis 0) numerically from lowest depth to highest depth and .sort_index(axis=1, ascending=True) sorts the columns (axis 1) numerically from lowest qubit count to highest qubit count.


        # ==========================================
        # Plotting both the 3D curve and the matrix heatmap side by side
        # ==========================================
        fig = plt.figure(figsize=(20, 8))

        # Subplot 1: The Original 3D Plot
        ax1 = fig.add_subplot(121, projection="3d")  # 1 row, 2 cols, position 1

        ax1.plot(x_free, y_free, z_free, label=notnoisy_constant_config.get("backend"), marker="o", linewidth=2, color="blue")
        ax1.plot(x_noisy, y_noisy, z_noisy, label=noisy_constant_config.get("backend"), marker="x", linestyle="--", linewidth=2, color="blue")

        ax1.set_xlabel("n_qubits (X)")
        ax1.set_ylabel("circuit_depth (Y)")
        ax1.set_zlabel("Precision (Z)")
        ax1.set_title("Precision Comparison: Noise-Free vs Noisy")
        ax1.legend()


        # Subplot 2: Matrix Heatmap
        ax2 = fig.add_subplot(122)  # 1 row, 2 cols, position 2

        sns.heatmap(matrix_df,  annot=True, fmt=".4f", cmap="coolwarm", center=0, ax=ax2)  #  coolwarm
        ax2.set_title("Precision Difference Matrix")
        ax2.set_xlabel("Number of Qubits (Columns)")
        ax2.set_ylabel("Circuit Depth (Rows)")

        plt.show()


    def noisy_notnoisy_single_module_runtime_comparison(self, module_name=None): 
        # Run compatibility check
        self.check_for_compatible_config_groups_across_simulators()

        # Extract data (Unpacking the tuple: data_list, constant_config)
        notnoisy_data_list, notnoisy_constant_config = self.qgm_data_object.extract_run_data_for_config_group(self.notnoisy_simulator_path)
        noisy_data_list, noisy_constant_config = self.qgm_data_object.extract_run_data_for_config_group(self.noisy_simulator_path)

        # Check if module name is provided and exists in the runtimes data
        if module_name is None:
            print("Error: A module name to extract the runtimes for must be provided as an argument to the function.")
            return None 
        elif module_name not in notnoisy_data_list[0]['runtimes']:  # Note: check_for_compatible_config_groups_across_simulators() should ensure that the module is present in both simulators, so we can just check one of them
            print(f"Error: Module '{module_name}' not found")
            print("Available modules are:")
            print(list(notnoisy_data_list[0]['runtimes'].keys()))
            return None

        # Helper function to extract and sort axes data
        def prepare_plot_vectors(data_list):
            # Sorting ensures the "curve" connects points in a logical order
            sorted_data = sorted(data_list, key=lambda x: (x['n_qubits'], x['circuit_depth']))

            if not sorted_data:
                print("Warning: No data available to plot.")
                return [], [], []

            x = [d['n_qubits'] for d in sorted_data]
            y = [d['circuit_depth'] for d in sorted_data]
            z = [d['runtimes'][module_name]['total_time'].time for d in sorted_data]

            # extract units and check for consistency
            units = [d['runtimes'][module_name]['total_time'].unit for d in sorted_data]
            if len(set(units)) > 1:
                print(f"Error: Inconsistent time units found in the runtimes for module '{module_name}'. Units found: {set(units)}")
                return None, None, None
            return x, y, z, units[0]  # return the common unit as well

        # Prepare vectors for both simulators
        x_free, y_free, z_free, unit_free = prepare_plot_vectors(notnoisy_data_list)  # n_qubits, circuit_depth, runtime for a chosen noise-free simulator module
        x_noisy, y_noisy, z_noisy, unit_noisy = prepare_plot_vectors(noisy_data_list)  # n_qubits, circuit_depth, runtime for a chosen noisy simulator module

        if unit_free != unit_noisy:
            print(f"Error: Time units for the runtimes of module '{module_name}' do not match between the not-noisy and noisy simulators. Not-noisy unit: {unit_free}, Noisy unit: {unit_noisy}")
            return None

        # Calculate runtime difference between the two simulators for each corresponding run (they are in the same order after sorting)
        runtime_diff = [zf - zn for zf, zn in zip(z_free, z_noisy)]
        runtime_diff_per_nqubit_depth = {(x, y): diff for x, y, diff in zip(x_free, y_free, runtime_diff)}

        # Restructure dictionary into a sorted matrix DataFrame
        df_data = [{"n_qubits": q, "circuit_depth": d, "diff": diff} for (q, d), diff in runtime_diff_per_nqubit_depth.items()]
        df = pd.DataFrame(df_data)
        matrix_df = df.pivot(index="circuit_depth", columns="n_qubits", values="diff")  # turn into 2D matrix with circuit_depth as rows and n_qubits as columns
        matrix_df = matrix_df.sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)  # .sort_index(axis=0, ascending=True) sorts the rows (axis 0) numerically from lowest depth to highest depth and .sort_index(axis=1, ascending=True) sorts the columns (axis 1) numerically from lowest qubit count to highest qubit count.

        # ==========================================
        # Plotting both the 3D curve and the matrix heatmap side by side
        # ==========================================
        fig = plt.figure(figsize=(20, 8))

        # Subplot 1: The Original 3D Plot
        ax1 = fig.add_subplot(121, projection="3d")  # 1 row, 2 cols, position 1

        ax1.plot(x_free, y_free, z_free, label=notnoisy_constant_config.get("backend"), marker="o", linewidth=2, color="blue")
        ax1.plot(x_noisy, y_noisy, z_noisy, label=noisy_constant_config.get("backend"), marker="x", linestyle="--", linewidth=2, color="blue")

        ax1.set_xlabel("n_qubits")
        ax1.set_ylabel("circuit_depth")
        ax1.set_zlabel(f"Runtime ({unit_free})")
        ax1.set_title(f"{module_name} total runtime Comparison: Noise-Free vs Noisy")
        ax1.legend()


        # Subplot 2: Matrix Heatmap
        ax2 = fig.add_subplot(122)  # 1 row, 2 cols, position 2

        sns.heatmap(matrix_df,  annot=True, fmt=".4f", cmap="coolwarm", center=0, ax=ax2)  #  coolwarm
        ax2.set_title(f"{module_name} total runtime Difference Matrix in {unit_free} (Noise-Free - Noisy)")
        ax2.set_xlabel("Number of Qubits (Columns)")
        ax2.set_ylabel("Circuit Depth (Rows)")

        plt.show()

    def noisy_notnoisy_modular_runtime_comparison(self):
        # Run compatibility check
        self.check_for_compatible_config_groups_across_simulators()

        # Extract data (Unpacking the tuple: data_list, constant_config)
        notnoisy_data_list, notnoisy_constant_config = self.qgm_data_object.extract_run_data_for_config_group(self.notnoisy_simulator_path)
        noisy_data_list, noisy_constant_config = self.qgm_data_object.extract_run_data_for_config_group(self.noisy_simulator_path)

        # Helper function to extract and sort axes data
        def prepare_plot_vectors(data_list):
            # Sorting ensures the "curve" connects points in a logical order
            sorted_data = sorted(data_list, key=lambda x: (x['n_qubits'], x['circuit_depth']))
            module_names = sorted_data[0]['runtimes'].keys() if sorted_data else []

            print(module_names)
            if not sorted_data:
                print("Warning: No data available to plot.")
                return [], [], []

            x = [d['n_qubits'] for d in sorted_data]
            y = [d['circuit_depth'] for d in sorted_data]
    
            runtime_by_module = {}
            for module_name in module_names:
                z = [d['runtimes'][module_name]['total_time'].time for d in sorted_data]
                runtime_by_module[module_name] = z

                # extract units and check for consistency
                units = [d['runtimes'][module_name]['total_time'].unit for d in sorted_data]
                if len(set(units)) > 1:
                    print(f"Error: Inconsistent time units found in the runtimes for module '{module_name}'. Units found: {set(units)}")
                    return None, None, None
            return x, y, runtime_by_module, units[0]  # return the common unit as well

        # Prepare vectors for both simulators
        x_free, y_free, runtime_by_module_free, unit_free = prepare_plot_vectors(notnoisy_data_list)  # n_qubits, circuit_depth, runtime for every noise-free simulator module
        x_noisy, y_noisy, runtime_by_module_noisy, unit_noisy = prepare_plot_vectors(noisy_data_list)  # n_qubits, circuit_depth, runtime for every noisy simulator module

        if unit_free != unit_noisy:
            print(f"Error: Time units for the runtimes of the modules between the not-noisy and noisy simulators do not match. Not-noisy unit: {unit_free}, Noisy unit: {unit_noisy}")
            return None

        # ==========================================
        # Plotting both the 3D curve and the matrix heatmap
        # ==========================================
        
        # ------------------------------------------
        # 1. 3D plots for each module (Figure 1)
        # ------------------------------------------

        # Figure out how many modules there are to determine the grid shape for the subplots
        num_modules = len(runtime_by_module_free)

        # Decide on a fixed number of columns (e.g., 3 subplots per row looks good)
        num_cols = 3
        # Calculate how many rows we need (e.g., 6 modules / 3 cols = 2 rows)
        num_rows = math.ceil(num_modules / num_cols)

        # Adjust figure size dynamically based on the grid shape
        fig3d = plt.figure(figsize=(5 * num_cols, 5 * num_rows))

        # Using enumerate(..., 1) lets 'idx' count up starting from 1
        for idx, module_name in enumerate(list(runtime_by_module_free.keys()), 1):
            
            # Grid dimensions are dynamically calculated (num_rows, num_cols)
            ax = fig3d.add_subplot(num_rows, num_cols, idx, projection="3d")
            
            # Plot the noise-free data for this specific module
            ax.plot(x_free, y_free, runtime_by_module_free[module_name], marker="o", linewidth=2, label=f"{notnoisy_constant_config.get('backend')}")
            
            # Plot the noisy data for this specific module
            ax.plot(x_noisy, y_noisy, runtime_by_module_noisy[module_name], marker="x", linestyle="--", linewidth=2, label=f"{noisy_constant_config.get('backend')}")

            # Set up labels and unique title for THIS specific module's subplot
            ax.set_xlabel("n_qubits")
            ax.set_ylabel("circuit_depth")
            ax.set_zlabel(f"Runtime ({unit_free})")
            ax.set_title(f"{module_name} Runtime Comparison")
            ax.legend()

        # Automatically adjust padding so nothing overlaps
        fig3d.tight_layout()
        plt.show()  # Displays the 3D grid figure window



        # ------------------------------------------
        # 2. Calculate Difference Matrices (Data Prep)
        # ------------------------------------------
        
        # Calculate runtime difference between the two simulators for each corresponding run
        runtime_diff_by_module = {
            module_name: [zf - zn for zf, zn in zip(runtime_by_module_free[module_name], runtime_by_module_noisy[module_name])] 
            for module_name in runtime_by_module_free.keys()
        }

        # Restructure data into sorted matrix DataFrames for every module name
        matrices_by_module = {}
        for module_name, diff_list in runtime_diff_by_module.items():
            runtime_diff_per_nqubit_depth = {(q, d): diff for q, d, diff in zip(x_free, y_free, diff_list)}
            df_data = [{"n_qubits": q, "circuit_depth": d, "diff": diff} for (q, d), diff in runtime_diff_per_nqubit_depth.items()]
            df = pd.DataFrame(df_data)
            
            matrix_df = df.pivot(index="circuit_depth", columns="n_qubits", values="diff")
            matrix_df = matrix_df.sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)
            matrices_by_module[module_name] = matrix_df



        # ------------------------------------------
        # 3. Matrix Heatmaps for each module (Figure 2)
        # ------------------------------------------
        
        # We use the same dynamic grid layout logic as the 3D plots
        num_cols_heatmap = 3
        num_rows_heatmap = math.ceil(num_modules / num_cols_heatmap)

        # Create a completely separate, dedicated figure window for the heatmaps
        fig_heatmap = plt.figure(figsize=(6 * num_cols_heatmap, 5 * num_rows_heatmap))

        for idx, (module_name, matrix_df) in enumerate(matrices_by_module.items(), 1):
            
            # Dynamically set up the subplot grid position
            ax2 = fig_heatmap.add_subplot(num_rows_heatmap, num_cols_heatmap, idx)
            
            # Generate the heatmap for this module
            sns.heatmap(
                matrix_df, 
                annot=True, 
                fmt=".4f", 
                cmap="coolwarm", 
                center=0, 
                ax=ax2,
                cbar_kws={'label': f'Difference ({unit_free})'}
            )
            
            # Titles and axes configuration
            ax2.set_title(f"{module_name} Runtime Difference\n(Noise-Free - Noisy)")
            ax2.set_xlabel("Number of Qubits (Columns)")
            ax2.set_ylabel("Circuit Depth (Rows)")

        # Layout fix for the heatmap figure and then reveal it
        fig_heatmap.tight_layout()
        plt.show()  # Displays the Heatmap grid figure window
    
    def noisy_notnoisy_KL_divergence_comparison(self):
        # Run compatibility check
        self.check_for_compatible_config_groups_across_simulators()

        # Extract data (Unpacking the tuple: data_list, constant_config)
        notnoisy_data_list, notnoisy_constant_config = self.qgm_data_object.extract_run_data_for_config_group(self.notnoisy_simulator_path)
        noisy_data_list, noisy_constant_config = self.qgm_data_object.extract_run_data_for_config_group(self.noisy_simulator_path)

        # Helper function to extract and sort axes data
        def prepare_plot_vectors(data_list):
            # Sorting ensures the "curve" connects points in a logical order
            sorted_data = sorted(data_list, key=lambda x: (x['n_qubits'], x['circuit_depth']))

            if not sorted_data:
                print("Warning: No data available to plot.")
                return [], [], []

            x = [d['n_qubits'] for d in sorted_data]
            y = [d['circuit_depth'] for d in sorted_data]
            z = [d['KL_best'] for d in sorted_data]
            return x, y, z

        # Prepare vectors for both simulators
        x_free, y_free, z_free = prepare_plot_vectors(notnoisy_data_list)  # n_qubits, circuit_depth, KL_divergence for noise-free simulator
        x_noisy, y_noisy, z_noisy = prepare_plot_vectors(noisy_data_list)  # n_qubits, circuit_depth, KL_divergence for noisy simulator

        # Calculate KL divergence difference between the two simulators for each corresponding run (they are in the same order after sorting)
        kl_diff = [zf - zn for zf, zn in zip(z_free, z_noisy)]
        kl_diff_per_nqubit_depth = {(x, y): diff for x, y, diff in zip(x_free, y_free, kl_diff)}

        # Restructure dictionary into a sorted matrix DataFrame
        df_data = [{"n_qubits": q, "circuit_depth": d, "diff": diff} for (q, d), diff in kl_diff_per_nqubit_depth.items()]
        df = pd.DataFrame(df_data)
        matrix_df = df.pivot(index="circuit_depth", columns="n_qubits", values="diff")  # turn into 2D matrix with circuit_depth as rows and n_qubits as columns
        matrix_df = matrix_df.sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)  # .sort_index(axis=0, ascending=True) sorts the rows (axis 0) numerically from lowest depth to highest depth and .sort_index(axis=1, ascending=True) sorts the columns (axis 1) numerically from lowest qubit count to highest qubit count.


        # ==========================================
        # Plotting both the 3D curve and the matrix heatmap side by side
        # ==========================================
        fig = plt.figure(figsize=(20, 8))

        # Subplot 1: The Original 3D Plot
        ax1 = fig.add_subplot(121, projection="3d")  # 1 row, 2 cols, position 1

        ax1.plot(x_free, y_free, z_free, label=notnoisy_constant_config.get("backend"), marker="o", linewidth=2, color="blue")
        ax1.plot(x_noisy, y_noisy, z_noisy, label=noisy_constant_config.get("backend"), marker="x", linestyle="--", linewidth=2, color="blue")

        ax1.set_xlabel("n_qubits (X)")
        ax1.set_ylabel("circuit_depth (Y)")
        ax1.set_zlabel("KL Divergence (Z)")
        ax1.set_title("KL Divergence Comparison: Noise-Free vs Noisy")
        ax1.legend()


        # Subplot 2: Matrix Heatmap
        ax2 = fig.add_subplot(122)  # 1 row, 2 cols, position 2

        sns.heatmap(matrix_df,  annot=True, fmt=".4f", cmap="coolwarm", center=0, ax=ax2)  #  coolwarm
        ax2.set_title("KL Divergence Difference Matrix")
        ax2.set_xlabel("Number of Qubits (Columns)")
        ax2.set_ylabel("Circuit Depth (Rows)")

        plt.show()





# --- Example Usage Data Extractor ---
base_path_pc = Path(r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\sorted")
base_path_itwm = Path(r"\\ITWM\u\g\garciabetancour\QUARK-2.1.7_fork\benchmark_runs\sorted")
base_path_gpu_cluster = Path(r"/home/garciabetancour/QUARK-2.1.7_fork/benchmark_runs/sorted")
qgm_data_extractor = QGMDataExtractor(base_path_gpu_cluster)

# Example of run parameters
# print("Start of examples:")
# run_data_example = qgm_data_extractor.extract_run_data_for_config_group(r"constant_config_1\fake_sherbrooke_simulator", print_results=True, print_constant_config=True)
# print(run_data_example)
# print()
# print("End of examples.")
# print("--------------------------------------------------")
# print()


# --- Volumetric Benchmarking Comparison ---
# aer_statevector_simulator_gpu_path = r"constant_config_1/aer_statevector_simulator_gpu"
aer_statevector_simulator_gpu_path = (Path("constant_config_1") / "aer_statevector_simulator_gpu")
# fake_sherbrooke_simulator_path = r"constant_config_1/fake_sherbrooke_simulator"
fake_sherbrooke_simulator_path = (Path("constant_config_1") / "fake_sherbrooke_simulator")
aer_statevector_fake_sherbrooke_comparator = VolBenchBySimulatorCategory(qgm_data_extractor, aer_statevector_simulator_gpu_path, fake_sherbrooke_simulator_path)

# aer_statevector_fake_sherbrooke_comparator.noisy_notnoisy_precision_comparison()
# aer_statevector_fake_sherbrooke_comparator.noisy_notnoisy_single_module_runtime_comparison('LibraryQiskit')
# aer_statevector_fake_sherbrooke_comparator.noisy_notnoisy_modular_runtime_comparison()
aer_statevector_fake_sherbrooke_comparator.noisy_notnoisy_KL_divergence_comparison()


# benchmark_runs/sorted/constant_config_1/aer_statevector_simulator_gpu
# /home/garciabetancour/QUARK-2.1.7_fork/benchmark_runs/sorted/constant_config_1/aer_statevector_simulator_gpu