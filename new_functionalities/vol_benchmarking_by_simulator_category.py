import os
import json
import pickle
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Runtime:
    def __init__(self, time, unit):
        self.time = time
        self.unit = unit
    
    def __repr__(self):
        return f"{self.time} {self.unit}"
    

# TODO: Only works for discrete datasets, extend functionality to also work for continuous datasets (e.g., by adding an if statement that checks the dataset type and then extracts the relevant parameters accordingly)
class DiscreteQGMDataExtractor:
    """Class to extract data from benchmark runs for volumetric benchmarking of generative quantum modeling applications."""
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.extract_run_data_for_config_group_was_called = False

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
    
    # Note: The following function only works for the case that the user selects the discrete dataset 
    def get_config_parameters(self, file_path):
        if not os.path.exists(file_path):
            print(f"Error: File not found at {os.path.abspath(file_path)}")
            return None

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            if ext in ['.yml', '.yaml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            else:
                print(f"Unsupported format: {ext}")
                return None

        except Exception as e:
            print(f"Failed to load: {e}")
            return None

        try:
            application = data["application"]

            # Top level
            n_qubits = application["config"]["n_qubits"][0]

            # Discrete Data
            data_module = application["submodules"][0]
            train_size = data_module["config"]["train_size"][0]

            # CircuitCardinality
            circuit_module = data_module["submodules"][0]
            depth = circuit_module["config"]["depth"][0]

            # LibraryQiskit
            library_module = circuit_module["submodules"][0]
            backend = library_module["config"]["backend"][0]
            n_shots = library_module["config"]["n_shots"][0]

            # QGAN
            training_module = library_module["submodules"][0]
            training_config = training_module["config"]

            config_parameters = {
                "n_qubits": n_qubits,
                "depth": depth,
                "data": data_module["name"].lower(),
                "circuit": circuit_module["name"]
                    .replace("Circuit", "")
                    .lower(),
                "library": library_module["name"]
                    .replace("Library", "")
                    .lower(),
                "backend": backend,
                "n_shots": n_shots,
                "training": training_module["name"],
                "repetitions": data.get("repetitions"),
                "ML metrics": {
                    "train_size": train_size,
                    "batch_size": training_config["batch_size"][0],
                    "device": training_config["device"][0],
                    "epochs": training_config["epochs"][0],
                    "learning_rate_discriminator":
                        training_config["learning_rate_discriminator"][0],
                    "learning_rate_generator":
                        training_config["learning_rate_generator"][0],
                    "loss": training_config["loss"][0],
                    "pretrained": training_config["pretrained"][0],
                }
            }

        except Exception as e:
            print(f"Failed to parse config structure: {e}")
            return None

        return config_parameters
    
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
        runtimes = {"all modules": {"total_runtime": Runtime(results.get("total_time"), results.get("total_time_unit"))}}

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
    # TODO: Note that the algorith is hard coded to go into the subfolders names "generativemodeling..." If the user changes the name of these folders, the function will not work.
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
        for gen_folder in target_dir.glob("generativemodeling-*"):
            
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

            if config_data is None:
                continue

            # --- Duplicate run check ---
            current_qubits = config_data.get('n_qubits')
            current_depth = config_data.get('depth')
            combination = (current_qubits, current_depth)

            if combination in seen_combinations and not self.extract_run_data_for_config_group_was_called:
                print(f"WARNING: Two runs found with the exact same number of qubits ({current_qubits}) "
                    f"and circuit depth ({current_depth}) in parent folder {target_dir.name}. First found in: {seen_combinations[combination]}, duplicate found in: {gen_folder.name}")
            else:
                seen_combinations[combination] = gen_folder.name

            # --- CONFIG CONSISTENCY CHECK ---
            # Extract constants (everything EXCEPT n_qubits and depth)
            current_constants = {k: v for k, v in config_data.items() if k not in ['n_qubits', 'depth']}
            
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
                'runtimes': runtimes_data
            }
            all_run_results.append(run_entry)

        if consistency_error:
            print("Note: Some benchmark runs had differing constant parameters. Check logs above.")

        # Printing results for verification
        if print_results:
            for idx, run in enumerate(all_run_results):
                print(f"Benchmark run {idx+1}: {all_run_results[idx]}")
        
        if print_constant_config:
            print(f"Data from benchmark runs in directory: {target_dir} extracted successfully.")
            print(f"Number of runs processed: {len(all_run_results)} with constant configuration parameters:" )
            print("Global Constant Config:", global_constant_config)
            print()

        self.extract_run_data_for_config_group_was_called = True  
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
        print("Precision differences per (n_qubits, circuit_depth):")
        for (n_qubits, circuit_depth), diff in precision_diff_per_nqubit_depth.items():
            print(f"  ({n_qubits}, {circuit_depth}): {diff}")

        # Initialize 3D Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot curves
        # Points/markers are included to visualize individual benchmark runs
        ax.plot(x_free, y_free, z_free, label=notnoisy_constant_config.get("backend"), marker='o', linewidth=2)
        ax.plot(x_noisy, y_noisy, z_noisy, label=noisy_constant_config.get("backend"), marker='x', linestyle='--', linewidth=2)

        # Labels and Formatting
        ax.set_xlabel('n_qubits (X)')
        ax.set_ylabel('circuit_depth (Y)')
        ax.set_zlabel('Precision (Z)')
        ax.set_title('Precision Comparison: Noise-Free vs Noisy Simulator')
        ax.legend()

        plt.show()

    # TODO: For now, the function only takes into account the run time of all modules combined. It can be easily changed to take into account the runtime per module. 
    def noisy_notnoisy_runtime_comparison(self): 
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
            z = [d['runtimes']['all modules']['total_runtime'].time for d in sorted_data]
            return x, y, z

        # Prepare vectors for both simulators
        x_free, y_free, z_free = prepare_plot_vectors(notnoisy_data_list)  # n_qubits, circuit_depth, runtime for noise-free simulator
        x_noisy, y_noisy, z_noisy = prepare_plot_vectors(noisy_data_list)  # n_qubits, circuit_depth, runtime for noisy simulator
    

        # Initialize 3D Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot curves
        # Points/markers are included to visualize individual benchmark runs
        ax.plot(x_free, y_free, z_free, label=notnoisy_constant_config.get("backend"), marker='o', linewidth=2)
        ax.plot(x_noisy, y_noisy, z_noisy, label=noisy_constant_config.get("backend"), marker='x', linestyle='--', linewidth=2)

        # Labels and Formatting
        ax.set_xlabel('n_qubits (X)')
        ax.set_ylabel('circuit_depth (Y)')
        ax.set_zlabel('Runtime (Z)')
        ax.set_title('Total Runtime Comparison: Noise-Free vs Noisy Simulator')
        ax.legend()

        plt.show()


# --- Example Usage Data Extractor ---
base_path_pc = Path(r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\sorted")
base_path_itwm = Path(r"\\ITWM\u\g\garciabetancour\QUARK-2.1.7_fork\benchmark_runs\sorted")
qgm_data_extractor = DiscreteQGMDataExtractor(base_path_pc)

# Example of run parameters
print("Start of examples:")
run_data_example = qgm_data_extractor.extract_run_data_for_config_group(r"constant_config_1\fake_sherbrooke_simulator", print_results=True, print_constant_config=True)
print(run_data_example)
print()
print("End of examples.")
print("--------------------------------------------------")
print()


# --- Volumetric Benchmarking Comparison ---
aer_statevector_simulator_gpu_path = r"constant_config_1\aer_statevector_simulator_gpu"
fake_sherbrooke_simulator_path = r"constant_config_1\fake_sherbrooke_simulator"
aer_statevector_fake_sherbrooke_comparator = VolBenchBySimulatorCategory(qgm_data_extractor, aer_statevector_simulator_gpu_path, fake_sherbrooke_simulator_path)
# aer_statevector_fake_sherbrooke_comparator.noisy_notnoisy_precision_comparison()
aer_statevector_fake_sherbrooke_comparator.noisy_notnoisy_runtime_comparison()




def noisy_notnoisy_pmf_comparison(qgm_data_object, notnoisy_simulator_path, noisy_simulator_path):
    # Implement similar structure to the precision comparison function, but instead of plotting precision, we will plot the probability mass functions (PMFs) for each run.
    pass