import itertools
import subprocess
import os
import json
from multiprocessing import Pool

learning_rates = [0.01, 0.001]
dropouts = [0.1, 0.3, 0.5]
weight_decays = [1e-4, 0]
num_model_layers = [1, 3]
num_prediction_layers = [1, 3]
embedding_dims = [128, 256]

param_grid = list(itertools.product(
    learning_rates, dropouts, weight_decays, num_model_layers, num_prediction_layers, embedding_dims
))

best_precision = 0
best_params = None

def run_experiment(params):
    lr, dropout, l2, num_layers, num_layers_predictor, hidden_channels, device = params
    print(f"Running experiment with lr={lr}, dropout={dropout}, l2={l2}, num_layers={num_layers}, "
          f"num_layers_predictor={num_layers_predictor}, hidden_channels={hidden_channels} on device {device}")
    
    command = [
        "python", "main_gnn_default.py",
        "--data_name", "bluesky",
        "--gnn_model", "GCN",
        "--lr", str(lr),
        "--dropout", str(dropout),
        "--l2", str(l2),
        "--num_layers", str(num_layers),
        "--num_layers_predictor", str(num_layers_predictor),
        "--hidden_channels", str(hidden_channels),
        "--epochs", "9999",
        "--kill_cnt", "10",
        "--eval_steps", "5",
        "--batch_size", "8192",
        "--runs", "1",
        "--output_dir", "gridsearch_results",
        "--save",
        "--device", str(device)  # 指定设备
    ]
    
    subprocess.run(command)
    
    result_file = os.path.join("gridsearch_results", f'lr{lr}_drop{dropout}_l2{l2}_numlayer{num_layers}_numPredlay{num_layers_predictor}_dim{hidden_channels}_best_run_0', "result.json")
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            result = json.load(f)
            avg_precision = result['Precision'][1]
            return avg_precision, (lr, dropout, l2, num_layers, num_layers_predictor, hidden_channels)
    return 0, None

if __name__ == "__main__":
    param_grid_with_device = [(lr, dropout, l2, num_layers, num_layers_predictor, hidden_channels, i % 2) for i, (lr, dropout, l2, num_layers, num_layers_predictor, hidden_channels) in enumerate(param_grid)]
    
    with Pool(processes=2) as pool:
        results = pool.map(run_experiment, param_grid_with_device)
    
    for avg_precision, params in results:
        if avg_precision > best_precision:
            best_precision = avg_precision
            best_params = params

    print(f"Best Precision: {best_precision}")
    if best_params:
        print(f"Best Parameters: lr={best_params[0]}, dropout={best_params[1]}, l2={best_params[2]}, num_layers={best_params[3]}, num_layers_predictor={best_params[4]}, hidden_channels={best_params[5]}")