from argparse import ArgumentParser

from data import ALL_OPERATIONS
from training import main


import os
import wandb

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--tag', type=str, default=None, help='tag for wandb to use')
    parser.add_argument('--run_name', type=str, default=None, help='name for wandb to use')



    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimizer to use')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer (if used)')
    parser.add_argument('--scheduler', type=str, default='linear', choices=['linear', 'step'], help='Learning rate scheduler to use')
    parser.add_argument('--total_iters', type=int, default=9, help='Total iterations for linear scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for step scheduler')


    parser.add_argument("--operation", type=str, choices=ALL_OPERATIONS.keys(), default="x/y")
    parser.add_argument("--training_fraction", type=float, default=0.5)
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_interval", type=int, default=50)


    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='Standard deviation of the Gaussian noise to be added to the dataset columns.')
        # Adding arguments
    parser.add_argument('--scale_factor', type=float, default=1,
                        help='Scale factor to multiply the initial parameters of the model.')


    args = parser.parse_args()

    main(args)
