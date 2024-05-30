from math import ceil
import torch
from tqdm import tqdm
import wandb

from data import get_data
from model import Transformer
import os

import torch.nn.functional as F
import random
import numpy as np
from torch.nn.functional import cosine_similarity

def define_gradient_norm_metrics(model):
    for name, _ in model.named_parameters():
        # Create a hierarchical name for the metric
        wandb.define_metric(f"grad_norm/{name}", step_metric="step")
        wandb.define_metric(f"weight_norm/{name}", step_metric="step")
        wandb.define_metric(f"grad_changes_cossim/{name}", step_metric="step")
        wandb.define_metric(f"grad_changes_norm/{name}", step_metric="step")



def save_checkpoint(model, optimizer, filename="final_model_checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    return filename


def main(args: dict):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(123)
    random.seed(0)
    np.random.seed(0)


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
    wandb.define_metric("training/real_mse", step_metric='step')

    wandb.define_metric("validation/accuracy", step_metric='epoch')
    wandb.define_metric("validation/loss", step_metric='epoch')
    wandb.define_metric("validation/real_mse", step_metric='epoch')

    wandb.define_metric("grokking/epoch_train>95%")
    wandb.define_metric("grokking/epoch_val>95%")
    wandb.define_metric("grokking/epoch_delay")


    wandb.define_metric("grokking/step_train>95%")
    wandb.define_metric("grokking/step_val>95%")
    wandb.define_metric("grokking/step_delay")


    train_loader, val_loader, train_size, val_size = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size,
        )

    wandb.log({"data/train_size": train_size})
    wandb.log({"data/val_size": val_size})

    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=2 * config.prime + 2,
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


    first_95_train = None
    first_95_eval = None

    # first_checkpoint_filename = save_checkpoint(model, optimizer, filename="first_model_checkpoint.pth")
    # wandb.save(first_checkpoint_filename)

    for epoch in tqdm(range(num_epochs)):
        train_acc = train(model, train_loader, optimizer, scheduler, device, config.num_steps, config.noise_level)
        eval_acc = evaluate(model, val_loader, device, epoch)

        # Check and log when training and evaluation accuracy first exceed 95%
        if train_acc >= 0.95 and first_95_train is None:
            first_95_train = epoch
            wandb.log({"grokking/epoch_train>95%": first_95_train})
            wandb.log({"grokking/step_train>95%": first_95_train * len(train_loader)})

        if eval_acc >= 0.95 and first_95_eval is None:
            first_95_eval = epoch
            wandb.log({"grokking/epoch_val>95%": first_95_eval})
            wandb.log({"grokking/step_val>95%": first_95_eval * len(train_loader)})

    if first_95_eval is not None and first_95_train is not None:
        wandb.log({"grokking/epoch_delay": first_95_eval - first_95_train})
        wandb.log({"grokking/step_delay": (first_95_eval - first_95_train) * len(train_loader)})

    # final_checkpoint_filename = save_checkpoint(model, optimizer)
    # wandb.save(final_checkpoint_filename)
    wandb.finish()



def train(model, train_loader, optimizer, scheduler, device, num_steps, noise_level):
    # Set model to training mode
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    total_acc = 0.0
    total_count = 0
    old_gradients = None
    # Loop over each batch from the training set
    for batch in train_loader:
        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Zero gradient buffers
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs, noise_level)[-1,:,:]
        loss = criterion(output, labels)
        predictions = torch.argmax(output, dim=1)
        acc = (predictions == labels).float().sum().item()  # Get total correct predictions as Python scalar
        mse = F.mse_loss(predictions.float(), labels.float()).item()

        # Save a copy of the current gradients (initialized to zero before the forward pass)
        if not old_gradients:
            old_gradients = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param) for name, param in model.named_parameters()}
        else:
            old_gradients = new_gradients

        # Backward pass
        loss.backward()

        # Calculate the new gradients after the backward pass
        new_gradients = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param) for name, param in model.named_parameters()}

        gradient_changes = {}
        # Calculate cosine similarity and L2 difference between old and new gradients
        for name in old_gradients.keys():
            old_grad = old_gradients[name].view(-1)
            new_grad = new_gradients[name].view(-1)

            cos_sim = cosine_similarity(old_grad.unsqueeze(0), new_grad.unsqueeze(0)).item()
            l2_diff = torch.norm(new_grad - old_grad, p=2).item()

            gradient_changes[f"grad_changes_cossim/{name}"] = cos_sim
            gradient_changes[f"grad_changes_norm/{name}"] = l2_diff


        # Collect and log gradient norms
        gradient_norms = {}
        weight_norms = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                gradient_norms[f"grad_norm/{name}"] = parameter.grad.norm(2).item()

                weight_norms[f"weight_norm/{name}"] = parameter.norm(2).item()


        # Update weights
        optimizer.step()
        scheduler.step()


        metrics = {
            "training/accuracy": acc / len(labels),
            "training/loss": loss,
            "training/real_mse": mse,
            "step": wandb.run.step,
            **gradient_norms,
            **weight_norms,
            **gradient_changes
        }
        wandb.log(metrics)

        # Accumulate total accuracy
        total_acc += acc
        total_count += labels.size(0)

        # Finish training at maximum gradient updates
        if wandb.run.step >= num_steps:
            break

    # Calculate overall training accuracy
    overall_training_accuracy = total_acc / total_count

    return overall_training_accuracy

# You should replace placeholders like wandb.run.step with actual control logic for step counting and termination if not using wandb.

def evaluate(model, val_loader, device, epoch):
    # Set model to evaluation mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    loss = 0.
    mse = 0.

    # Loop over each batch from the validation set
    for batch in val_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Forward pass
        with torch.no_grad():
            output = model(inputs)[-1,:,:]

            predictions = torch.argmax(output, dim=1)
            correct += (predictions == labels).sum()
            loss += criterion(output, labels) * len(labels)

            mse += F.mse_loss(predictions.float(), labels.float()).item()

    acc = correct / len(val_loader.dataset)
    loss = loss / len(val_loader.dataset)
    mse = mse / len(val_loader.dataset)

    metrics = {
        "validation/accuracy": acc,
        "validation/loss": loss,
        "validation/real_mse": mse,
        "epoch": epoch
    }
    wandb.log(metrics, commit=False)

    return acc
