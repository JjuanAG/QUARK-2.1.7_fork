import os
import json
import pickle
import numpy as np
import yaml
from pathlib import Path



def fetch_data(file_path):
    """
    Loads data from .npy, .pkl, .json, or .yml/.yaml files.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {os.path.abspath(file_path)}")
        return None

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    try:
        if ext == '.npy':
            data = np.load(file_path, allow_pickle=True)

        elif ext == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        elif ext in ['.yml', '.yaml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

        else:
            print(f"Unsupported format: {ext}")
            return None

    except Exception as e:
        print(f"Failed to load: {e}")
        return None
    
    return data


def get_precission(file_path):
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
    
    precission = data.get('precision', None) if isinstance(data, dict) else None

    return precission


def get_probability_distribution(file_path):
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
def get_config_parameters(file_path):
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


def get_runtimes(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {os.path.abspath(file_path)}")
        return None

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    try:
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            print(f"Unsupported format: {ext}")
            return None

    except Exception as e:
        print(f"Failed to load: {e}")
        return None

    runtimes = {}

    def extract_module_name(module):
        """
        Extracts module name from module_src path.
        Example:
        src/.../discrete_data.py -> discrete_data
        """
        module_src = module.get("module_src", "")

        if module_src:
            return os.path.splitext(os.path.basename(module_src))[0]

        return "unknown_module"

    def extract_module_times(module):
        """
        Recursively extracts timing information from modules.
        """
        module_name = extract_module_name(module)

        metrics = module.get("metrics", {})

        runtimes[module_name] = {
            "total_time": metrics.get("total_time"),
            "total_time_unit": metrics.get("total_time_unit"),
            "preprocessing_time": metrics.get("preprocessing_time"),
            "preprocessing_time_unit": metrics.get("preprocessing_time_unit"),
            "postprocessing_time": metrics.get("postprocessing_time"),
            "postprocessing_time_unit": metrics.get("postprocessing_time_unit"),
        }

        # Recursively process submodules
        for submodule in module.get("submodules", []):
            extract_module_times(submodule)

    # Start recursion from application root
    if "application" in data:
        extract_module_times(data["application"])

    # Add overall runtime
    runtimes["summed_time"] = {
        "total_time": data.get("total_time"),
        "total_time_unit": data.get("total_time_unit")
    }

    return runtimes


############
base_path_pc = Path(r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\sorted")
base_path_itwm = Path(r"\\ITWM\u\g\garciabetancour\QUARK-2.1.7_fork\benchmark_runs\sorted")

def extract_run_data_for_config_group(simulator_path_name, base_path):  # TODO: Extend functionality so that it can also extract the runtimes
    # 1. Setup the target path       
    target_dir = base_path / simulator_path_name
    
    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        return [], {}

    all_run_results = []
    global_constant_config = None
    consistency_error = False

    # 2. Iterate through all folders starting with 'generativemodeling'
    for gen_folder in target_dir.glob("generativemodeling-*"):
        
        # Locate required files
        config_file = next(gen_folder.glob("config.yml"), None)
        metrics_file = next(gen_folder.rglob("record_gen_metrics*.pkl"), None)
        histogram_file = next(gen_folder.rglob("histogram_generated.npy"), None)

        if not all([config_file, metrics_file, histogram_file]):
            print(f"Skipping {gen_folder.name}: Missing one or more required files.")
            continue

        # --- DATA EXTRACTION ---
        config_data = get_config_parameters(str(config_file))
        precision_val = get_precission(str(metrics_file))
        pmf_data = get_probability_distribution(str(histogram_file))

        if config_data is None:
            continue

        # --- CONFIG CONSISTENCY CHECK ---
        # Extract constants (everything EXCEPT n_qubits and depth)
        current_constants = {k: v for k, v in config_data.items() if k not in ['n_qubits', 'depth']}
        
        if global_constant_config is None:
            # First folder sets the baseline for constants
            global_constant_config = current_constants
        else:
            # Compare current constants to the baseline
            if current_constants != global_constant_config:
                print(f"Warning: Configuration mismatch in folder {gen_folder.name}")
                consistency_error = True

        # --- RESTRUCTURE RUN DATA ---
        run_entry = {
            'n_qubits': config_data.get('n_qubits'),
            'circuit_depth': config_data.get('depth'),
            'precission': precision_val,
            'pmf': pmf_data
        }
        all_run_results.append(run_entry)

    if consistency_error:
        print("Note: Some benchmark runs had differing constant parameters. Check logs above.")

    print(f"Number of runs processed: {len(all_run_results)}")

    # Printing results for verification
    for idx, run in enumerate(all_run_results):
        print(f"Benchmark run {idx+1}: {all_run_results[idx]}")
    print()
    print("Global Constant Config:", global_constant_config)
    return all_run_results, global_constant_config

# --- Example Usage ---
extract_run_data_for_config_group(r"constant_config_1\aer_statevector_simulator_gpu", base_path_itwm)
# extract_benchmark_run_data_for_config_group(r"constant_config_1\aer_statevector_simulator_gpu", base_path_pc)
# --- ---

def check_for_compatible_config_groups_across_simulators(notnoisy_simulator_path, noisy_simulator_path, base_path):
    notnoisy_data_list, notnoisy_constants = extract_run_data_for_config_group(notnoisy_simulator_path, base_path)
    noisy_data_list, noisy_constants = extract_run_data_for_config_group(noisy_simulator_path, base_path)

    if notnoisy_constants != noisy_constants:
        print("Error: The constant configuration parameters do not match between the not-noisy and noisy simulators.")
        print("Not-noisy constants:", notnoisy_constants)
        print("Noisy constants:", noisy_constants)
        return False
    for idx, (notnoisy_run, noisy_run) in enumerate(zip(notnoisy_data_list, noisy_data_list)):
        if notnoisy_run['n_qubits'] != noisy_run['n_qubits'] or notnoisy_run['circuit_depth'] != noisy_run['circuit_depth']:
            print(f"Error: Run {idx+1} has mismatching n_qubits or circuit_depth between not-noisy and noisy simulators.")
            print(f"Not-noisy run: n_qubits={notnoisy_run['n_qubits']}, circuit_depth={notnoisy_run['circuit_depth']}")
            print(f"Noisy run: n_qubits={noisy_run['n_qubits']}, circuit_depth={noisy_run['circuit_depth']}")
            return False

    print("Success: The constant configuration parameters match between the not-noisy and noisy simulators.")
    return True



#TODO: Extend the functionality so that it processes probability distributions and run times as well
#TODO: Not runs with noisy simulators have been done yet
def noisy_notnoisy_precision_comparison(notnoisy_simulator_path, noisy_simulator_path = None):  
    # Extract data
    notnoisy_data_list = extract_run_data_for_config_group(notnoisy_simulator_path, base_path_itwm)
    noisy_data_list = extract_run_data_for_config_group(noisy_simulator_path, base_path_itwm) if noisy_simulator_path else None
    if len(notnoisy_data_list) != len(noisy_data_list):
        print("Error: The number of runs with the not-noisy and noisy simulators must be equal.")
        return

    # volumetric benchmarking for not-noisy and noisy simulators
