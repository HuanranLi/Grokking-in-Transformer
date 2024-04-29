import wandb
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch

import datetime
from model import *
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar  # Correct import for ColorbarBase


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


def filter_and_process_runs(entity, project_name, filters, metric_names, file_name="final_model_checkpoint.pth", max_runs = 5):
    api = wandb.Api()
    project_path = f"{entity}/{project_name}"
    runs = api.runs(path=project_path, filters=filters)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_dir = f"vis_{current_time}"
    os.makedirs(main_dir, exist_ok=True)

    artifact_paths = []
    configs = []
    metrics = []

    for run in runs:
        if len(configs) > max_runs:
            break

        print(f"Processing Run: {run.id}")
        # delays.append(run.summary.get("grokking/epoch_delay", 0))
        # Attempt to retrieve the 'grokking/epoch_delay' from the run's summary
        metric_dict = {}
        for metric_name in metric_names:
            metric_dict[metric_name] = run.summary.get(metric_name)
        metrics.append(metric_dict)

        config = run.config
        configs.append(config)


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

    return artifact_paths, configs, metrics, main_dir




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


def extract_layer_weights(artifact_dirs, configs, layer_index):
    layer_weights = []
    expected_shape = None
    expected_name = None

    for dir, config in zip(artifact_dirs, configs):
        model = load_model(dir, config) # Ensure load_model is designed to return model and optionally layer names
        # Get the state dictionary
        state_dict = model.state_dict()

        # Convert state_dict keys or items to a list and access by index
        keys_list = list(state_dict.keys())
        values_list = list(state_dict.values())

        if layer_index < len(keys_list):  # Check if layer_index is within range
            parameter_key = keys_list[layer_index]
            parameter_value = values_list[layer_index]
        else:
            raise IndexError("Layer index out of range")

        # Access a parameter by index
        parameter_key = keys_list[layer_index]
        parameter_value = values_list[layer_index]

        # Append the detached and numpy-converted tensor to the list
        layer_weights.append(parameter_value.detach().cpu().numpy())

        # Check if all tensors are of the same shape and name
        if expected_shape is None:
            expected_shape = parameter_value.shape
            expected_name = parameter_key  # Set the expected name on first iteration
        else:
            if parameter_value.shape != expected_shape:
                raise ValueError(f"Shape mismatch: Expected {expected_shape}, but got {parameter_value.shape} in {parameter_key}")
            if parameter_key != expected_name:
                raise ValueError(f"Layer name mismatch: Expected {expected_name}, but got {parameter_key}")

    layer_weights = np.array(layer_weights)
    layer_weights = layer_weights.reshape(layer_weights.shape[0], -1)
    print("Collection of layer weights shape: ", layer_weights.shape)
    return layer_weights, expected_name



def main():
    entity = "huanran-research"
    project_name = "grokking"
    filters = {
        "state": {"$eq": "finished"},  # Correct: Filters runs that have finished
        "sweep": {"$eq": "cir87vhd"},  # Correct: Filters runs belonging to a specific sweep
    }

    max_runs = 1000
    metric_names = ["grokking/epoch_delay", "training/accuracy"]
    layer_max = 30

    artifact_dirs, configs, metrics, main_dir = filter_and_process_runs(entity = entity,
                                                                    project_name = project_name,
                                                                    filters = filters,
                                                                    metric_names = metric_names,
                                                                    max_runs = max_runs)

    if len(artifact_dirs) == 0:
        print('No file to found to download!')
        assert False

    for layer_index in range(1, layer_max + 1):
        try:
            layer_weights, expected_name = extract_layer_weights(artifact_dirs, configs, layer_index)
            n_samples = layer_weights.shape[0]
            perplexity_value = min(30, n_samples - 1)

            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity_value)
            transformed_weights = tsne.fit_transform(layer_weights)

            delays =[i['grokking/epoch_delay'] for i in metrics]
            train_acc =[i['training/accuracy'] for i in metrics]


            # Assuming 'delays' contains 'grokking/epoch_delay' metrics
            valid_delays = [d for d in delays if d is not None]
            norm = mcolors.Normalize(vmin=min(valid_delays), vmax=min(max(valid_delays), 1000), clip=True)
            cmap = plt.get_cmap("plasma_r")

            # Convert delay values to colors using the colormap
            color_values = []
            alphas = []
            for d, acc in zip(delays, train_acc):
                if d:
                    color_values.append(cmap(norm(d)))
                    alphas.append(0.7)
                elif acc > 0.95:
                    color_values.append("darkgreen")
                    alphas.append(0.5)
                else:
                    color_values.append('black')
                    alphas.append(0.2)

            # alphas = [0.7 if d is not None else 0.2 for d in delays]

            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity_value)
            transformed_weights = tsne.fit_transform(layer_weights)

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6), dpi = 200)

            # Create scatter plot on the specific axes 'ax'
            scatter = ax.scatter(transformed_weights[:, 0], transformed_weights[:, 1], c=color_values, alpha = alphas)

            # Set title and labels
            ax.set_title(f't-SNE of Transformer Layer Weights')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax, label='Grokking Delay')
            plt.savefig(os.path.join(main_dir, f"{layer_index}_{expected_name}.png"))  # Save the plot with layer index and expected name
            # plt.show()

        except IndexError as e:
            if str(e) == "Layer index out of range":
                print(f"Stopping: {e}")
                break  # Break the loop if layer index is out of range
            else:
                print(f"Other IndexError: {str(e)}")
                continue  # Continue the loop if the error is not about the layer index range

        except Exception as e:
            print(f"Failed to process layer {layer_index}: {str(e)}")


    # delete_directory(main_dir)

# If this script is intended to be run directly, use:
if __name__ == "__main__":
    main()
