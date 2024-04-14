from math import ceil
import torch
from tqdm import tqdm
import wandb

from data import get_data
from model import Transformer
import os


def define_gradient_norm_metrics(model):
    for name, _ in model.named_parameters():
        # Create a hierarchical name for the metric
        metric_name = f"grad_norm/{name.replace('.', '/')}"
        wandb.define_metric(metric_name, step_metric="step")


def main(args: dict):
    current_path = os.getcwd()
    os.environ["WANDB_DIR"] = current_path
    os.environ["WANDB_CACHE_DIR"] = os.path.join(current_path, '.cache/wandb')
    os.environ["WANDB_CONFIG_DIR"] = os.path.join(current_path, '.config/wandb')
    print('Current Path', current_path)
    # wandb.login(key=args.wandb_api_key)
    wandb.init(project="grokking", config=args, dir = current_path)
    config = wandb.config
    device = torch.device(config.device)

    # Define time scales
    wandb.define_metric("step")
    wandb.define_metric("epoch")

    # Define metrics
    wandb.define_metric("training/accuracy", step_metric='step')
    wandb.define_metric("training/loss", step_metric='step')
    wandb.define_metric("validation/accuracy", step_metric='epoch')
    wandb.define_metric("validation/loss", step_metric='epoch')


    train_loader, val_loader = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size,
        )
    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=config.prime + 2,
        seq_len=5
        ).to(device)

    define_gradient_norm_metrics(model)
    
    for param in model.parameters():
        param.data *= config.scale_factor

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=config.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor = 0.1, total_iters=9
    )

    num_epochs = ceil(config.num_steps / len(train_loader))

    for epoch in tqdm(range(num_epochs)):
        train(model, train_loader, optimizer, scheduler, device, config.num_steps, config.noise_level, config.noise_cols_mode)
        evaluate(model, val_loader, device, epoch)

def train(model, train_loader, optimizer, scheduler, device, num_steps, noise_level, noise_cols_mode):
    # Set model to training mode
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    # Loop over each batch from the training set
    for batch in train_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Zero gradient buffers
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs, noise_level, noise_cols_mode)[-1,:,:]
        loss = criterion(output, labels)
        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

        # Backward pass
        loss.backward()

        # Collect and log gradient norms
        gradient_norms = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                grad_norm = parameter.grad.norm(2).item()
                gradient_norm_name = f"grad_norm/{name.replace('.', '/')}"
                gradient_norms[gradient_norm_name] = grad_norm


        # Update weights
        optimizer.step()
        scheduler.step()

        metrics = {
            "training/accuracy": acc,
            "training/loss": loss,
            "step": wandb.run.step,
            **gradient_norms
        }
        wandb.log(metrics)

        # Finish training at maximum gradient updates
        if wandb.run.step == num_steps:
            return

def evaluate(model, val_loader, device, epoch):
    # Set model to evaluation mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    loss = 0.

    # Loop over each batch from the validation set
    for batch in val_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Forward pass
        with torch.no_grad():
            output = model(inputs)[-1,:,:]
            correct += (torch.argmax(output, dim=1) == labels).sum()
            loss += criterion(output, labels) * len(labels)

    acc = correct / len(val_loader.dataset)
    loss = loss / len(val_loader.dataset)

    metrics = {
        "validation/accuracy": acc,
        "validation/loss": loss,
        "epoch": epoch
    }
    wandb.log(metrics, commit=False)
