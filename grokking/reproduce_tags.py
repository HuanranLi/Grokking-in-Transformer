import os
import wandb
from argparse import ArgumentParser
from data import ALL_OPERATIONS
from training import main
import subprocess

import wandb

def get_wandb_runs_with_tag(project_name, tag):
    api = wandb.Api()
    print('tag:', tag)
    runs = api.runs(f"{project_name}", filters={"tags": {"$in": [tag]}})
    print('Find runs:', runs)

    # Generate new tag
    base_new_tag = tag + "_v2"
    new_tag = base_new_tag
    version = 2

    # Check if the new tag already exists
    existing_tags = set()
    for run in api.runs(f"{project_name}"):
        existing_tags.update(run.tags)

    while new_tag in existing_tags:
        version += 1
        new_tag = f"{tag}_v{version}"

    print(new_tag)
    return runs, new_tag


def extract_hyperparams(run):
    config = run.config
    return {
        '--optimizer': config.get('optimizer', 'adamw'),
        '--momentum': config.get('momentum', 0.9),
        '--scheduler': config.get('scheduler', 'linear'),
        '--total_iters': config.get('total_iters', 9),
        '--gamma': config.get('gamma', 0.1),
        '--operation': config.get('operation', 'x/y'),
        '--training_fraction': config.get('training_fraction', 0.5),
        '--prime': config.get('prime', 97),
        '--num_layers': config.get('num_layers', 2),
        '--dim_model': config.get('dim_model', 128),
        '--num_heads': config.get('num_heads', 4),
        '--batch_size': config.get('batch_size', 512),
        '--learning_rate': config.get('learning_rate', 1e-3),
        '--weight_decay': config.get('weight_decay', 1),
        '--num_epochs': config.get('num_epochs', 1000),
        '--device': config.get('device', 'cpu'),
        '--noise_level': config.get('noise_level', 0.0),
        '--scale_factor': config.get('scale_factor', 1),
    }

def execute_cli_with_args(args, new_tag, run_name):
    cli_command = ["python", "cli.py"]  # Assuming this script is saved as cli_script.py
    for key, value in args.items():
        cli_command.append(key)
        cli_command.append(str(value))

    cli_command.append('--tag')
    cli_command.append(new_tag)

    cli_command.append('--run_name')
    cli_command.append(run_name)

    subprocess.run(cli_command)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--project', type=str, default='huanran-research/grokking-v2', help='WandB project name')
    parser.add_argument('--tag', type=str, required=True, help='Tag to filter runs')
    args = parser.parse_args()

    runs, new_tag = get_wandb_runs_with_tag(args.project, args.tag)
    for run in runs:
        run_name = run.name
        print(f"Processing run: {run_name}")
        # assert False
        hyperparams = extract_hyperparams(run)
        # assert False
        execute_cli_with_args(hyperparams, new_tag, run_name)
