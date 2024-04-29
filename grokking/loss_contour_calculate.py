import torch
import yaml
from model import Transformer  # Assuming the model definition is in this module
from data import get_data      # Assuming the data loading function is in this module

import os
import wandb

import torch
import itertools
from torch import nn

import torch


import numpy as np

import json
import argparse


def load_model_and_data(checkpoint_path, config_path, device):

    # Load the config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract relevant configurations
    config_values = {k: v['value'] for k, v in config.items() if isinstance(v, dict)}
    print(config_values)

    # Load the model from checkpoint
    model = Transformer(
        num_layers=config_values['num_layers'],
        dim_model=config_values['dim_model'],
        num_heads=config_values['num_heads'],
        num_tokens=2 * config_values['prime'] + 2,  # Assuming num_tokens derived from prime as seen earlier
        seq_len=5  # This needs to be specified or calculated based on your application
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode if not training

    # Load the dataset
    train_loader, val_loader, train_size, val_size = get_data(
        config_values['operation'],
        config_values['prime'],
        config_values['training_fraction'],
        config_values['batch_size'],
    )

    return model, train_loader, val_loader, train_size, val_size


def download_wandb_run_files(run_path, file_names, base_dir='visualized_runs'):
    api = wandb.Api()

    # Split the run path to get the run_id as the folder name
    run_id = run_path.split('/')[-1]
    run_dir = os.path.join(base_dir, run_id)

    # Create the run directory if it doesn't exist
    os.makedirs(run_dir, exist_ok=True)

    downloaded_files = {'run_dir': run_dir}

    for file_name in file_names:
        downloaded_files[file_name] = None
        file_path = os.path.join(run_dir, file_name)

        # Check if file already exists
        if os.path.exists(file_path):
            print(f"File already exists: {file_path}")
        else:
            # Access the run
            run = api.run(run_path)
            # Download the file
            try:
                print(f"Downloading {file_name} to {run_dir}")
                run.file(file_name).download(root=run_dir, replace=True)
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")
                continue

        downloaded_files[file_name] = file_path

    return downloaded_files


def explore_gradient_directions(model, train_loader, test_loader, device, steps = 10, search_range = 1, init_model = None):
    model.to(device).eval()   # Set the model to evaluation mode to disable dropout, etc.
    original_state_dict = {name: param.clone() for name, param in model.named_parameters() if param.requires_grad}

    random_directions = {}
    if not init_model:
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Generate two random directions
                random_dir1 = torch.randn_like(param)
                random_dir2 = torch.randn_like(param)

                # Normalize each direction by its norm
                norm1 = torch.norm(random_dir1) + 1e-10
                norm2 = torch.norm(random_dir2) + 1e-10

                random_dir1.div_(norm1)
                random_dir2.div_(norm2)

                # Store the normalized directions in the dictionary
                random_directions[name] = [random_dir1, random_dir2]
    else:
        print('Detect Init model')
        init_model.to(device).eval() 
        for (name, param), (init_name, init_param) in zip(model.named_parameters(), init_model.named_parameters()):
            assert name == init_name

            if param.requires_grad:
                # Generate two random directions
                random_dir1 = param - init_param
                random_dir2 = torch.randn_like(param)

                # Normalize each direction by its norm
                norm1 = torch.norm(random_dir1) + 1e-10
                norm2 = torch.norm(random_dir2) + 1e-10

                random_dir1.div_(norm1)
                random_dir2.div_(norm2)

                # Store the normalized directions in the dictionary
                random_directions[name] = [random_dir1, random_dir2]


    step_sizes = torch.linspace(-1 * search_range, 1 * search_range, steps=steps)  # Define step sizes, e.g., -1 to 1 in 10 steps

    # Results dictionary
    results = {}

    # Iterate over all combinations of step sizes for two directions
    for steps1, steps2 in itertools.product(step_sizes, repeat=2):
        direction_loss = []

        # Apply each random direction with the respective step size
        for name, param in model.named_parameters():
            if param.requires_grad:
                direction1, direction2 = random_directions[name]
                perturbed_param = param + direction1 * steps1 + direction2 * steps2
                param.data.copy_(perturbed_param)

        # Compute losses
        training_loss = compute_loss(model, train_loader, device)
        testing_loss = compute_loss(model, test_loader, device)

        # Store results
        direction_loss.append((steps1.item(), steps2.item(), training_loss, testing_loss))

        # Reset model parameters to original
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(original_state_dict[name])

        results[(steps1.item(), steps2.item())] = (training_loss, testing_loss)

    return results

def compute_loss(model, data_loader, device):
    criterion = nn.CrossEntropyLoss()  # Adjust this based on your actual loss function
    total_loss = 0
    count = 0
    model.to(device)
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)[-1, :, :]
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            count += 1
    return total_loss / count


def save_results_to_file(results, filename):
    # Convert tuple keys to string keys
    results_str_keys = {str(key): value for key, value in results.items()}
    with open(filename, 'w') as f:
        json.dump(results_str_keys, f)
    print(f"Results saved to {filename}")



def main(args):
    run_id = args.run_path.split('/')[-1]
    wandb.init(project='grokking', id=run_id, resume="allow")

    run_path = args.run_path
    if args.twin_model:
        files = download_wandb_run_files(run_path, ['config.yaml', 'final_model_checkpoint.pth', 'first_model_checkpoint.pth'])
    else:
        files = download_wandb_run_files(run_path, ['config.yaml', 'final_model_checkpoint.pth'])
        files['first_model_checkpoint.pth'] = None


    device = torch.device(args.device)
    model, train_loader, val_loader, train_size, val_size = load_model_and_data(
        files['final_model_checkpoint.pth'], files['config.yaml'], device)

    if files['first_model_checkpoint.pth']:
        init_model, _, _, _, _ = load_model_and_data(
            files['first_model_checkpoint.pth'], files['config.yaml'], device)
    else:
        init_model = None


    results = explore_gradient_directions(
        model, train_loader, val_loader, device, steps=args.steps, search_range=args.search_range, init_model = init_model)

    print(results)

    if init_model:
        results_filename = os.path.join(wandb.run.dir, f'Twin_results_steps_{args.steps}_range_{args.search_range}.json')
    else:
        results_filename = os.path.join(wandb.run.dir, f'results_steps_{args.steps}_range_{args.search_range}.json')

    save_results_to_file(results, results_filename)
    wandb.save(results_filename)

    wandb.finish()


    # plot_loss_contours(results, save_path=os.path.join(files['run_dir'], f'steps_{args.steps}_range_{args.search_range}'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gradient Exploration on a Model')
    parser.add_argument('--run_path', type=str, required=True, help='WandB run path to download files from')
    parser.add_argument('--steps', type=int, default=5, help='Number of steps for gradient exploration')
    parser.add_argument('--search_range', type=float, default=1.0, help='Range of exploration around the gradient')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (cpu or cuda)')


    parser.add_argument('--twin_model', type=int, default=1, help='two models with one being the init model')


    args = parser.parse_args()
    main(args)
