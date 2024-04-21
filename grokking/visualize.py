import wandb
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch

import datetime
from model import *

import shutil
import os

def delete_directory(dir_path):
    # Check if the directory exists
    if os.path.exists(dir_path):
        # Remove the directory and all its contents
        shutil.rmtree(dir_path)
        print(f"Directory {dir_path} has been deleted.")
    else:
        print(f"The directory {dir_path} does not exist.")


def filter_and_process_runs(entity, project_name, filters, file_name="final_model_checkpoint.pth"):
    api = wandb.Api()
    project_path = f"{entity}/{project_name}"
    runs = api.runs(path=project_path, filters=filters)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_dir = f"vis_{current_time}"
    os.makedirs(main_dir, exist_ok=True)

    artifact_paths = []
    configs = []
    delays = []

    for run in runs:
        print(f"Processing Run: {run.id}")
        config = run.config
        configs.append(config)
        delays.append(run.summary.get("grokking/delay", 0))

        run_dir = os.path.join(main_dir, run.id)
        os.makedirs(run_dir, exist_ok=True)

        file_path = os.path.join(run_dir, file_name)
        run_file = run.file(file_name)
        if run_file:
            run_file.download(replace=True, root=run_dir)
            artifact_paths.append(file_path)
            print(f"Checkpoint file for Run {run.id} downloaded to: {file_path}")
        else:
            print(f"No checkpoint file named '{file_name}' found for Run {run.id}")

    return artifact_paths, configs, delays, main_dir




def load_model(checkpoint_path, config):
    # Load the model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Initialize the model using the provided configuration
    model = Transformer(
        num_layers=config['num_layers'],
        dim_model=config['dim_model'],
        num_heads=config['num_heads'],
        num_tokens=2 * config['prime'] + 2,
        seq_len=5
    )

    # Load the state dictionary into the model
    model.load_state_dict(checkpoint['model_state_dict'])


    return model

def main():
    layer_index = 5
    entity = "huanran-research"
    project_name = "grokking"
    filters = {
        "state": {"$eq": "finished"},
        "sweep": {"$eq": "guql2u6m"}     # Assuming prime is a numeric field
    }


    artifact_dirs, configs, delays, main_dir = filter_and_process_runs(entity, project_name, filters)

    if len(artifact_dirs) == 0:
        print('No file to found to download!')
        assert False

    layer_weights = []
    expected_shape = None
    expected_name = None
    for dir, config in zip(artifact_dirs, configs):
        model = load_model(dir, config)  # Specify layer index
        # Get the state dictionary
        state_dict = model.state_dict()

        # Convert state_dict keys or items to a list and access by index
        keys_list = list(state_dict.keys())
        values_list = list(state_dict.values())

        # Access a parameter by index
        parameter_key = keys_list[layer_index]
        parameter_value = values_list[layer_index]

        layer_weights.append(parameter_value.detach().cpu().numpy())

        # Check if all tensors are of the same shape
        if expected_shape is None:
            expected_shape = parameter_value.shape
            expected_name = parameter_key  # Set the expected name on first iteration
        else:
            if parameter_value.shape != expected_shape:
                raise ValueError(f"Shape mismatch: Expected {expected_shape}, but got {parameter_value.shape} in {parameter_key}")
            if parameter_key != expected_name:
                raise ValueError(f"Layer name mismatch: Expected {expected_name}, but got {parameter_key}")


    layer_weights = np.array(layer_weights)
    print("Collection of layer weights shape: ", layer_weights.shape)
    # Check the number of samples
    n_samples = layer_weights.shape[0]

    # Set the perplexity to a valid value
    perplexity_value = min(30, n_samples - 1)  # Common practice: choose 30 or the number of samples minus one


    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity_value)
    transformed_weights = tsne.fit_transform(layer_weights)

    # Plotting
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(transformed_weights[:, 0], transformed_weights[:, 1], c=delays, cmap='viridis')
    plt.colorbar(scatter, label='Grokking Delay')
    plt.title(f't-SNE of Transformer Layer {expected_name} Weights')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


    delete_directory(main_dir)


# If this script is intended to be run directly, use:
if __name__ == "__main__":
    main()
