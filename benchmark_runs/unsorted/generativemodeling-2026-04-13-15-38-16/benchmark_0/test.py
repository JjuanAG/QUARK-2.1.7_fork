import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def process_quark_data(file_path, title='QUARK Data Visualization', print_summary=True):
    """
    Loads, visualizes, and returns data from .npy or .pkl files.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {os.path.abspath(file_path)}")
        return None

    _, ext = os.path.splitext(file_path)
    
    try:
        # Load logic
        if ext == '.npy':
            data = np.load(file_path, allow_pickle=True)
        elif ext == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            print(f"Unsupported format: {ext}")
            return None
    except Exception as e:
        print(f"Failed to load: {e}")
        return None

    # --- Data Extraction for Listing ---
    # Convert data to a consistent list of (key, value) pairs for the readout
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, np.ndarray):
        # Handle multidimensional arrays by flattening or taking first row if needed
        flat_data = data.flatten()
        items = list(enumerate(flat_data))
    else:
        items = list(enumerate(data))

    # --- Console Output (The "List" view) ---
    if print_summary:
        print(f"\n{'Index/Key':<20} | {'Value':<10}")
        print("-" * 35)
        for key, val in items:
            # Formatting value to 4 decimal places if it's a float
            val_str = f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
            print(f"{str(key):<20} | {val_str:<10}")
        print("-" * 35)

    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    keys = [str(i[0]) for i in items]
    values = [i[1] for i in items]
    
    plt.bar(keys, values, color='skyblue', edgecolor='navy')
    plt.xticks(rotation=45 if len(keys) > 10 else 0)
    plt.xlabel('Index or Key')
    plt.ylabel('Value')
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return data  # Returns the original object for further use

# histogram_train = '/home/juana/QUARK-2.1.7_fork/benchmark_runs/generativemodeling-2026-04-15-09-28-03/benchmark_0/rep_1/histogram_generated.npy'
# process_quark_data(histogram_train, title='Histogram Train Data')

# best_params = '/home/juana/QUARK-2.1.7_fork/benchmark_runs/generativemodeling-2026-04-15-09-28-03/benchmark_0/rep_1/best_parameters_1.npy'
# process_quark_data(best_params, title='Best Parameters Data')

# gen_metrics = '/home/juana/QUARK-2.1.7_fork/benchmark_runs/generativemodeling-2026-04-15-09-28-03/benchmark_0/rep_1/record_gen_metrics_1.pkl'
# process_quark_data(gen_metrics, title='Generated Metrics')

training_results = '/home/juana/QUARK-2.1.7_fork/benchmark_runs/generativemodeling-2026-04-15-09-28-03/benchmark_0/rep_1/training_results-1.pkl'
process_quark_data(training_results, title='Training Results')