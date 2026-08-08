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
import fnmatch
from collections import Counter

class Runtime:
    def __init__(self, time, unit):
        self.time = time
        self.unit = unit

    def __repr__(self):
        return f"{self.time} {self.unit}"

    def __add__(self, other):
        if not isinstance(other, Runtime):
            return NotImplemented
        if self.unit != other.unit:
            raise ValueError(
                f"Cannot add Runtime objects with different units: '{self.unit}' and '{other.unit}'"
            )
        return Runtime(self.time + other.time, self.unit)

    def __sub__(self, other):
        if not isinstance(other, Runtime):
            return NotImplemented
        if self.unit != other.unit:
            raise ValueError(
                f"Cannot subtract Runtime objects with different units: '{self.unit}' and '{other.unit}'"
            )
        return Runtime(self.time - other.time, self.unit)
    
class DataFetcher:

    def __init__(self, dir_path: str):
        self.dir_path = dir_path

    def load_file(self, file_path: str):
        """Loads data from a given file path based on extension."""
        target_path = os.path.join(self.dir_path, file_path)

        if not os.path.exists(target_path):
            print(f"Error: File not found at {os.path.abspath(target_path)}")
            return None

        _, ext = os.path.splitext(target_path)

        try:
            if ext == ".npy":
                return np.load(target_path, allow_pickle=True)
            elif ext == ".pkl":
                with open(target_path, "rb") as f:
                    return pickle.load(f)
            elif ext == ".json":
                with open(target_path, "r") as f:
                    return json.load(f)
            elif ext in [".yml", ".yaml"]:
                with open(target_path, "r") as f:
                    return yaml.safe_load(f)
            else:
                print(f"Unsupported format: {ext}")
                return None
        except Exception as e:
            print(f"Failed to load {target_path}: {e}")
            return None

    def load_config(self, filename: str = "config.yml"):
        """Loads non-recursively directly from top-level self.dir_path."""
        return self.load_file(filename)

    def load_nested_files(self, pattern: str, ignore_surface: bool = False) -> list:
        """Recursively traverses subdirectories to find and load files matching wildcard patterns

        (e.g., 'data*.pkl', 'histogram*.npy', 'results.json').
        """
        results = []
        root_abs = os.path.abspath(self.dir_path)

        for current_root, _, filenames in os.walk(self.dir_path):
            current_abs = os.path.abspath(current_root)

            # Skip top-level directory if requested
            if ignore_surface and current_abs == root_abs:
                continue

            # Load matching files using Unix wildcard matching
            for filename in filenames:
                if fnmatch.fnmatch(filename, pattern):
                    full_path = os.path.join(current_root, filename)
                    loaded_data = self.load_file(full_path)
                    if loaded_data is not None:
                        results.append(loaded_data)

        return results


class QgmRunResultsExtractor:
    """Class to extract data from benchmark runs for volumetric benchmarking of generative quantum modeling applications."""
    def __init__(self, run_results_dir_path: str):
        self.run_results_dir_path = run_results_dir_path
        self.fetcher = DataFetcher(self.run_results_dir_path)
        self.extract_run_data_for_config_group_was_called = False
    
    
    def get_results_json(self, module_name: str | None, parameter_name: str, nested_parameter: str | None = None):
        """Reads a JSON file and extracts the parameters of a certain module or top-level metadata.

        The extraction is recursive and handles nested submodules and nested
        dictionaries.
        """
        loaded_files = self.fetcher.load_nested_files("results.json")  # returns a list of loaded JSON data, which is in turn a list of dictionaries

        if not loaded_files:
            print("Error: No results.json files found.")
            return None

        data = loaded_files[0][0]  # TODO: Handle multiple results.json files if needed

        def extract_module_params(current_data: list | dict, target_name: str | None) -> dict | None:
            EXCLUDE_KEYS = {"submodule", "module_name", "module_src", "module_config"}

            # Handle top-level metadata extraction when module_name is None or "None"
            if target_name is None or target_name == "None":
                root_obj = (
                    current_data[0]
                    if isinstance(current_data, list) and current_data
                    else current_data
                )
                if isinstance(root_obj, dict):
                    return {k: v for k, v in root_obj.items() if k != "module"}
                return None

            # Recursively search through lists
            if isinstance(current_data, list):
                for item in current_data:
                    res = extract_module_params(item, target_name)
                    if res is not None:
                        return res

            # Recursively search through dictionaries
            elif isinstance(current_data, dict):
                if current_data.get("module_name") == target_name:
                    return {
                        k: v
                        for k, v in current_data.items()
                        if k not in EXCLUDE_KEYS
                    }

                for value in current_data.values():
                    if isinstance(value, (dict, list)):
                        res = extract_module_params(value, target_name)
                        if res is not None:
                            return res

            return None

        # Fetch dictionary of parameters once
        module_parameters = extract_module_params(data, module_name)

        if module_parameters is None:
            print(
                f"Error: Module '{module_name}' or top-level metadata not found."
            )
            return None

        # Handle Runtime objects for time parameters
        if "time" in parameter_name and nested_parameter is None:
            parameter_name_unit = parameter_name.replace("time", "time_unit")
            parameter_value = module_parameters.get(parameter_name)
            parameter_unit = module_parameters.get(parameter_name_unit)
            return Runtime(parameter_value, parameter_unit)

        # Fetch top-level parameter value
        parameter_value = module_parameters.get(parameter_name)

        # If user specifies a nested parameter, drill down into the dictionary
        if nested_parameter is not None:
            if isinstance(parameter_value, dict):
                return parameter_value.get(nested_parameter)
            else:
                print(
                    f"Warning: '{parameter_name}' is not a dictionary. Cannot fetch '{nested_parameter}'."
                )
                return None

        return parameter_value

    def get_runtimes_from_results_json(self):
        """Extracts runtimes from the results.json file of a benchmark run. Returns a dictionary with the runtimes for each module and submodule, as well as the overall runtime."""
        loaded_files = self.fetcher.load_nested_files("results.json")
        data = loaded_files[0][0]  # TODO: Handle multiple results.json files if needed

        # total run time (from all modules combined) is stored in the top-level of the results.json file, so we can extract it directly
        runtimes = {"all_modules": {"total_time": Runtime(data.get("total_time"), data.get("total_time_unit"))}}

        # recursively go into every submodule and extract the runtimes
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
        
        extract_module_times(data)

        return runtimes

    # TODO: Implement a function that goes through all the combix_idx and fetches their data. But this function should be within the Class below. This class right now, is only meant to access to data inside a single bnechmark run result!

    def get_config_yml(self) -> dict:
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
            >>> config = obj.get_config_yml()
            >>> print(config)
            {
                'GenerativeModeling': {'n_qubits': 6},
                'LibraryQiskit': {'backend': 'aer_statevector_simulator_cpu', 'n_shots': 100}
            }
        """
        data = self.fetcher.load_file("config.yml")

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
    
    def get_npy_or_pkl(self, file_name: str):
        loaded_files = self.fetcher.load_file(file_name)
        data = loaded_files[0]  # TODO: Handle multiple files if needed
        
        ... # TODO: Implement logic to extract and return the desired data from the loaded .npy or .pkl file

class VolBenchBackend:
    def __init__(self, backend_path: str):
        self.backend_path = backend_path
        self._backend_group_config_consistency_check_was_called = False  # Check for consistency of constant parameters across runs in the backend

    def _backend_group_config_consistency_check(self):
        """Checks for consistency of constant parameters across all runs in the backend directory."""
        global_constant_config = None
        seen_combinations: dict[tuple, str] = {}  # dictionary containing (n_qubits, depth) as keys and the corresponding run folder name as values to detect duplicates
        for run_results_dir in self.backend_path.iterdir():
            if run_results_dir.is_dir():        
                run_results_extractor = QgmRunResultsExtractor(run_results_dir)
                config_yml = run_results_extractor.get_config_yml()  # Load config.yml for the run
                flat_config_data = {key: val for sub_dict in config_yml.values() for key, val in sub_dict.items()}

                # --- Duplicate run check ---
                current_qubits = flat_config_data.get('n_qubits')
                current_depth = flat_config_data.get('depth')
                combination = (current_qubits, current_depth)

                if combination in seen_combinations and not self._backend_group_config_consistency_check_was_called:
                    print(f"Error: Two runs found with the exact same number of qubits ({current_qubits}) "
                        f"and circuit depth ({current_depth}) in parent folder {self.backend_path.name}. First found in: {seen_combinations[combination]}, duplicate found in: {run_results_dir.name}")
                    return None
                else:    
                    seen_combinations[combination] = run_results_dir.name  # update the seen_combinations dictionary with the current run's folder name

                # --- CONFIG CONSISTENCY CHECK ---
                # Extract constants (everything EXCEPT n_qubits and depth)
                current_constants = {k: v for k, v in flat_config_data.items() if k not in ['n_qubits', 'depth']}
                
                if global_constant_config is None:
                    # First folder sets the baseline for constants
                    global_constant_config = current_constants
                else:
                    # Compare current constants to the baseline
                    if current_constants != global_constant_config:
                        print(f"Error: Mismatch of constant configuration parameters in folder {run_results_dir.name}")
                        return None  
        
        nqubits_depth_list: list[tuple] = list(seen_combinations.keys())  # Extract the list of (n_qubits, depth) combinations for all runs in the backend

        return global_constant_config, nqubits_depth_list     
    
    def _get_backend_group_data(self, datafile_name: str, module_name: str | None = None, parameter_name: str | None = None, nested_parameter: str | None = None):

        global_constant_config, nqubits_depth_list = self._backend_group_config_consistency_check()
        data_list = []
        for run_results_dir, (n_qubits, circuit_depth) in zip(self.backend_path.iterdir(), nqubits_depth_list):
            if run_results_dir.is_dir():
                run_results_extractor = QgmRunResultsExtractor(run_results_dir)
                if datafile_name == "results.json":
                    data = run_results_extractor.get_results_json(module_name, parameter_name, nested_parameter)  # Load results.json for the run
                elif [".pkl", ".npy"] in datafile_name:
                    data = run_results_extractor.get_npy_or_pkl(datafile_name)  # Load .pkl or .npy for the run
                else:
                    print(f"Unsupported datafile_name: {datafile_name}")
                    return None
                if isinstance(data, float) or  isinstance(data, Runtime):  # The data must be a number or Runtime object in order to make plots later
                    if nested_parameter is None:
                        data_name = parameter_name
                    else:
                        data_name = nested_parameter
                    data_list.append({
                        "n_qubits": n_qubits,
                        "circuit_depth": circuit_depth,
                        "data": data,
                        data_name: data
                    })
        return global_constant_config, data_list

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

class VolBenchBackendPair():
    def __init__(self, backend_path_1: str, backend_path_2: str):
        self.vol_backend_1 = VolBenchBackend(backend_path_1)
        self.vol_backend_2 = VolBenchBackend(backend_path_2)
        self._backend_group_config_consistency_check_was_called = False  # Check for consistency of constant parameters across runs in the backend

    def _check_consistency_across_backends(self):
        global_config_1, nqubits_depth_list_1 = self.vol_backend_1._backend_group_config_consistency_check()
        global_config_2, nqubits_depth_list_2 = self.vol_backend_2._backend_group_config_consistency_check()

        # Only either the noise_configuration or the backend should differ across global configurations
        if "noise_configuration" in global_config_1.keys() and "noise_configuration" in global_config_2.keys():
            if global_config_1["noise_configuration"] != global_config_2["noise_configuration"] and global_config_1["backend"] == global_config_2["backend"]:
                noise_config_or_backend = "noise_configuration"  # TODO: This might not necessarily be only "noise_configuration"
            elif global_config_1["backend"] != global_config_2["backend"] and global_config_1["noise_configuration"] == global_config_2["noise_configuration"]:
                noise_config_or_backend = "backend"  # TODO: This might not necessarily be only "backend"
            elif global_config_1["backend"] != global_config_2["backend"] and global_config_1["noise_configuration"] != global_config_2["noise_configuration"]:
                raise ValueError(
                    "Only either the backend or noise configuration can differ between backends, not both.\n"
                    f"Backend 1 constants: {global_config_1}\n"
                    f"Backend 2 constants: {global_config_2}"
                )
        else:
            noise_config_or_backend = "backend" 

        global_config_1_filtered = {k: v for k, v in global_config_1.items() if k != noise_config_or_backend}
        global_config_2_filtered = {k: v for k, v in global_config_2.items() if k != noise_config_or_backend}

        if global_config_1_filtered != global_config_2_filtered:
            raise ValueError(
                    "Constant configuration parameters mismatch between backends.\n"
                    f"Backend 1 constants: {global_config_1}\n"
                    f"Backend 2 constants: {global_config_2}"
                )
        if Counter(nqubits_depth_list_1) != Counter(nqubits_depth_list_2):
            raise ValueError(
                "Mismatch in run (n_qubits, circuit_depth) configurations or run counts between backends.\n"
                f"Backend 1 ({len(nqubits_depth_list_1)} runs): {nqubits_depth_list_1}\n"
                f"Backend 2 ({len(nqubits_depth_list_2)} runs): {nqubits_depth_list_2}"
            )
        
        return noise_config_or_backend
    
    def _backend_pair_data_extraction(self, datafile_name: str, module_name: str | None = None, parameter_name: str | None = None, nested_parameter: str | None = None):
        # Check for consistency across backends
        noise_config_or_backend = self._check_consistency_across_backends()

        # Extract data from both backends
        global_config_1, data_list_1 = self.vol_backend_1._get_backend_group_data(datafile_name, module_name, parameter_name, nested_parameter)
        global_config_2, data_list_2 = self.vol_backend_2._get_backend_group_data(datafile_name, module_name, parameter_name, nested_parameter)

        return (global_config_1, data_list_1), (global_config_2, data_list_2), noise_config_or_backend
    
    def backend_pair_single_parameter_vol_bench(self, datafile_name: str, module_name: str | None = None, parameter_name: str | None = None, nested_parameter: str | None = None):
        # Extract data from both backends
        (global_config_1, data_list_1), (global_config_2, data_list_2), noise_config_or_backend = self._backend_pair_data_extraction(datafile_name, module_name, parameter_name, nested_parameter)

        if noise_config_or_backend == "noise_configuration":
            backend_name_1 = global_config_1.get("noise_configuration", "Noise Config 1")
            backend_name_2 = global_config_2.get("noise_configuration", "Noise Config 2")
        else:
            backend_name_1 = global_config_1.get("backend", "Backend 1")
            backend_name_2 = global_config_2.get("backend", "Backend 2")

        if nested_parameter is None:
            data_name = parameter_name
        else:
            data_name = nested_parameter

        def prep_plot_vectors(data_list):
            sorted_data = sorted(data_list, key=lambda x: (x['n_qubits'], x['circuit_depth']))  # sort by n_qubits and circuit_depth to ensure the "curve" connects points in a logical order
            x = [d['n_qubits'] for d in sorted_data]
            y = [d['circuit_depth'] for d in sorted_data]
            z = [d[data_name] for d in sorted_data]
            if isinstance(z[0], Runtime):
                z = [d[data_name].time for d in sorted_data]  # Extract the time value from the Runtime object
                z_units = [d[data_name].unit for d in sorted_data]  # Extract the unit from the Runtime object
            return x, y, z, z_units[0] if isinstance(z[0], Runtime) else None  # Return the common unit if it's a Runtime object

        x1, y1, z1, z1_units = prep_plot_vectors(data_list_1)
        x2, y2, z2, z2_units = prep_plot_vectors(data_list_2)

        # calculate parameter difference between the two backends for each corresponding run
        # Relative Percentage Change (Capped at 100%): Normalizes the difference against the maximum of the two values so that the scale stays strictly bounded inside [0, 100]%:
        param_diff = [
            ((z1_val - z2_val) / max(z1_val, z2_val) * 100) if max(z1_val, z2_val) > 0 else 0.0
            for z1_val, z2_val in zip(z1, z2)
        ]
        # Map to (n_qubits, circuit_depth)
        param_diff_per_nqubit_depth = {(x, y): diff for x, y, diff in zip(x1, y1, param_diff)}

        # Restructure dictionary into a sorted matrix DataFrame
        df_data = [{"n_qubits": q, "circuit_depth": d, "diff": diff} for (q, d), diff in param_diff_per_nqubit_depth.items()]
        df = pd.DataFrame(df_data)
        matrix_df = df.pivot(index="circuit_depth", columns="n_qubits", values="diff")
        matrix_df = matrix_df.sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)

        # Plotting both the 3D curve and the matrix heatmap side by side
        fig = plt.figure(figsize=(20, 8))

        # Subplot 1: The Original 3D Plot
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot(x1, y1, z1, label=backend_name_1, marker="o", linewidth=2, color="blue")
        ax1.plot(x2, y2, z2, label=backend_name_2, marker="x", linestyle="--", linewidth=2, color="orange")
        ax1.set_xlabel("n_qubits")
        ax1.set_ylabel("circuit_depth")
        if z1_units is not None and z2_units is not None and z1_units == z2_units:
            ax1.set_zlabel(f"{data_name} ({z1_units})")
        else:
            ax1.set_zlabel(f"{data_name} (arb. units)")
        ax1.set_title(f"{data_name} volumetric comparison: {backend_name_1} vs {backend_name_2}")
        ax1.legend()

        # Subplot 2: Matrix Heatmap
        ax2 = fig.add_subplot(122)  # 1 row, 2 cols, position 2

        sns.heatmap(matrix_df,  annot=True, fmt=".4f", cmap="PuOr", center=0, ax=ax2)  #  coolwarm
        ax2.set_title(f"{data_name} Difference Matrix (Backend 1 - Backend 2) in %")
        ax2.set_xlabel("Number of Qubits (Columns)")
        ax2.set_ylabel("Circuit Depth (Rows)")

        plt.show()

    def backend_par_multi_modular_runtime_vol_bench(self):
        # TODO implement this function
        pass
    

# --- HPC Volumetric comparison ---
# base_path_itwm = Path(r"\\itwm\u\g\garciabetancour\QUARK-2.1.7_fork\benchmark_runs\generativemodeling_sweep_run_2026-08-05-17-13-30\volumetric_aer_simulator_gpu_vs_aer_simulator_cpu")
# aer_simulator_gpu_path = "aer_simulator_gpu"
# aer_simulator_cpu_path = "aer_simulator_cpu"

# run_results_directory = Path(r"\\itwm\u\g\garciabetancour\QUARK-2.1.7_fork\benchmark_runs\generativemodeling_sweep_run_2026-08-05-17-13-30\volumetric_aer_simulator_gpu_vs_aer_simulator_cpu\aer_simulator_cpu\comb_idx_19")
# test_data_extractor = QgmRunResultsExtractor(run_results_directory)
# config = test_data_extractor.get_config_yml()
# results_json = test_data_extractor.get_results_json(module_name="DiscreteData", parameter_name="generalization_metrics", nested_parameter="precision")
# results_times = test_data_extractor.get_runtimes_from_results_json()

# aer_simulator_cpu_directory_itwm = Path(r"\\itwm\u\g\garciabetancour\QUARK-2.1.7_fork\benchmark_runs\generativemodeling_sweep_run_2026-08-05-17-13-30\volumetric_aer_simulator_gpu_vs_aer_simulator_cpu\aer_simulator_cpu")
# vol_bench_aer_simulator_cpu = VolBenchBackend(aer_simulator_cpu_directory)
# aer_simulator_gpu_directory_itwm = Path(r"\\itwm\u\g\garciabetancour\QUARK-2.1.7_fork\benchmark_runs\generativemodeling_sweep_run_2026-08-05-17-13-30\volumetric_aer_simulator_gpu_vs_aer_simulator_cpu\aer_simulator_gpu")
# vol_bench_aer_simulator_gpu = VolBenchBackend(aer_simulator_gpu_directory)

aer_simulator_cpu_directory_pc = Path("/home/juana/QUARK-2.1.7_fork/benchmark_runs/generativemodeling_sweep_run_2026-08-05-17-13-30/volumetric_aer_simulator_gpu_vs_aer_simulator_cpu/aer_simulator_cpu")
aer_simulator_gpu_directory_pc = Path("/home/juana/QUARK-2.1.7_fork/benchmark_runs/generativemodeling_sweep_run_2026-08-05-17-13-30/volumetric_aer_simulator_gpu_vs_aer_simulator_cpu/aer_simulator_gpu")


# global_config, nqubits_depth_list = vol_bench_aer_simulator_cpu._backend_group_config_consistency_check()
# _, data_list = vol_bench_aer_simulator_cpu._get_backend_group_data(datafile_name="results.json", module_name="DiscreteData", parameter_name="generalization_metrics", nested_parameter="precision")

vol_bench_pair = VolBenchBackendPair(aer_simulator_cpu_directory_pc, aer_simulator_gpu_directory_pc)
bool = vol_bench_pair._check_consistency_across_backends()

vol_bench_pair.backend_pair_single_parameter_vol_bench(datafile_name="results.json", module_name="DiscreteData", parameter_name="total_time", nested_parameter=None)

# aer_simulator_comparator = VolBenchBySimulatorCategory(test_data_extractor, aer_simulator_gpu_path, aer_simulator_cpu_path)
# aer_simulator_comparator.noisy_notnoisy_modular_runtime_comparison()