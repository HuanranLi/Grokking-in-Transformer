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


import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def define_gradient_norm_metrics(model):
    for name, _ in model.named_parameters():
        # Create a hierarchical name for the metric
        wandb.define_metric(f"grad_norm/{name}", step_metric="step")
        wandb.define_metric(f"weight_norm/{name}", step_metric="step")
        wandb.define_metric(f"grad_changes_cossim/{name}", step_metric="step")
        wandb.define_metric(f"grad_changes_norm/{name}", step_metric="step")
        wandb.define_metric(f"weight_changes_norm/{name}", step_metric="step")
        wandb.define_metric(f"weight_changes_cossim/{name}", step_metric="step")


        if 'in_proj_weight' in name:
            # Adding the norms to the weight_norms dictionary
            wandb.define_metric(f"in_proj_weight_norm/{name}_q_norm", step_metric="step")
            wandb.define_metric(f"in_proj_weight_norm/{name}_k_norm", step_metric="step")
            wandb.define_metric(f"in_proj_weight_norm/{name}_v_norm", step_metric="step")

            wandb.define_metric(f"weight_changes_norm/{name}_q_norm", step_metric="step")
            wandb.define_metric(f"weight_changes_norm/{name}_k_norm", step_metric="step")
            wandb.define_metric(f"weight_changes_norm/{name}_v_norm", step_metric="step")

            wandb.define_metric(f"weight_changes_cossim/{name}_q_norm", step_metric="step")
            wandb.define_metric(f"weight_changes_cossim/{name}_k_norm", step_metric="step")
            wandb.define_metric(f"weight_changes_cossim/{name}_v_norm", step_metric="step")



def save_checkpoint(model, optimizer, filename="final_model_checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    return filename


def get_optimizer(config, model):
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    return optimizer

def get_scheduler(config, optimizer):
    if config.scheduler == 'linear':
        scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=config.total_iters
        )
    elif config.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=config.learning_rate, gamma=config.gamma
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")
    return scheduler
import os
import matplotlib.pyplot as plt

def save_top_predictions(loader, model, device, epoch, phase, run_id, top_k=5):
    model.eval()
    results = []
    difference_counts = []

    with torch.no_grad():
        for batch in loader:
            inputs, labels = tuple(t.to(device) for t in batch)
            outputs = model(inputs)[-1, :, :]
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_preds = probabilities.topk(top_k, dim=1)

            for i in range(inputs.size(0)):
                correct = top_preds[i, 0].item() == labels[i].item()
                differences = [(pred.item() - labels[i].item()) for pred in top_preds[i]]
                difference_counts.extend(differences)

                results.append({
                    'correct': correct,
                    'input': inputs[i].cpu().numpy().tolist(),
                    'label': labels[i].cpu().item(),
                    'top_predictions': top_preds[i].cpu().numpy().tolist(),
                    'top_probabilities': top_probs[i].cpu().numpy().tolist()
                })

    # Create the directory if it doesn't exist
    predictions_dir = os.path.join('predictions', run_id)
    os.makedirs(predictions_dir, exist_ok=True)

    # Save results to a text file
    filename = os.path.join(predictions_dir, f"{phase}_top_predictions_epoch_{epoch}.txt")
    with open(filename, 'w') as f:
        for result in results:
            f.write(f"Correct: {result['correct']}\n")
            f.write(f"Input: {result['input']}\n")
            f.write(f"Label: {result['label']}\n")
            f.write(f"Top Predictions: {result['top_predictions']}\n")
            f.write(f"Top Probabilities: {result['top_probabilities']}\n")
            f.write("\n")

    # Plot histogram of differences
    plt.figure(figsize=(10, 6))
    plt.hist(difference_counts, bins=range(min(difference_counts), max(difference_counts) + 1), alpha=0.75, edgecolor='black')
    plt.title(f'Histogram of Differences Between Top-5 Predictions and Labels ({phase} set)')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.grid(True)

    histogram_filename = os.path.join(predictions_dir, f"{phase}_histogram_epoch_{epoch}.png")
    plt.savefig(histogram_filename)
    plt.close()


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
    if args.tag:
        wandb.init(project="grokking-v2", config=args, dir = current_path, tags=[args.tag], name = args.run_name)
    else:
        wandb.init(project="grokking-v2", config=args, dir = current_path, name = args.run_name)

    config = wandb.config
    run_id = wandb.run.id
    device = torch.device(config.device)

    # Define time scales
    wandb.define_metric("step")
    wandb.define_metric("epoch")

    # Define metrics
    wandb.define_metric("training/top5_accuracy", step_metric='step')
    wandb.define_metric("training/top10_accuracy", step_metric='step')
    wandb.define_metric("training/accuracy", step_metric='step')
    wandb.define_metric("training/loss", step_metric='step')
    wandb.define_metric("training/real_mse", step_metric='step')

    wandb.define_metric("validation/top5_accuracy", step_metric='epoch')
    wandb.define_metric("validation/top10_accuracy", step_metric='epoch')
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

    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    # num_epochs = ceil(config.num_steps / len(train_loader))
    num_epochs = config.num_epochs


    first_95_train = None
    first_95_eval = None

    # first_checkpoint_filename = save_checkpoint(model, optimizer, filename="first_model_checkpoint.pth")
    # wandb.save(first_checkpoint_filename)

    for epoch in tqdm(range(num_epochs)):
        train_acc = train(model, train_loader, optimizer, scheduler, device, config.noise_level)
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

        if epoch % config.save_interval == 0:
            save_top_predictions(train_loader, model, device, epoch, 'train', run_id)
            save_top_predictions(val_loader, model, device, epoch, 'val', run_id)


    if first_95_eval is not None and first_95_train is not None:
        wandb.log({"grokking/epoch_delay": first_95_eval - first_95_train})
        wandb.log({"grokking/step_delay": (first_95_eval - first_95_train) * len(train_loader)})

    # final_checkpoint_filename = save_checkpoint(model, optimizer)
    # wandb.save(final_checkpoint_filename)
    wandb.finish()



def train(model, train_loader, optimizer, scheduler, device, noise_level):
    # Set model to training mode
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    total_acc = 0.0
    total_count = 0
    old_gradients = None

    # Loop over each batch from the training set
    previous_weights = {}

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

        top5_probs, top5_preds = output.topk(5, dim=1)
        top10_probs, top10_preds = output.topk(10, dim=1)
        top5_correct = sum([labels[i].item() in top5_preds[i] for i in range(len(labels))])
        top10_correct = sum([labels[i].item() in top10_preds[i] for i in range(len(labels))])


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
        gradient_changes[f"grad_changes_cossim/total"] = 0
        gradient_changes[f"grad_changes_norm/total"] = 0
        # Calculate cosine similarity and L2 difference between old and new gradients
        for name in old_gradients.keys():
            old_grad = old_gradients[name].view(-1)
            new_grad = new_gradients[name].view(-1)

            cos_sim = cosine_similarity(old_grad.unsqueeze(0), new_grad.unsqueeze(0)).item()
            l2_diff = torch.norm(new_grad - old_grad, p=2).item()

            gradient_changes[f"grad_changes_cossim/{name}"] = cos_sim
            gradient_changes[f"grad_changes_norm/{name}"] = l2_diff

            gradient_changes[f"grad_changes_cossim/total"] += cos_sim
            gradient_changes[f"grad_changes_norm/total"] += l2_diff


        # Collect and log gradient norms
        gradient_norms = {}
        weight_norms = {}

        # Assuming you have a dictionary to store old weights
        l2_differences = {}
        cosine_differences = {}

        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                # Save the current gradient and weight norms
                gradient_norms[f"grad_norm/{name}"] = parameter.grad.norm(2).item()
                weight_norms[f"weight_norm/{name}"] = parameter.norm(2).item()

                # Save the current parameters for next iteration comparison
                if name in previous_weights:
                    # Calculate L2 difference
                    l2_differences[f"weight_changes_norm/{name}"] = torch.norm(parameter - previous_weights[name]).item()
                    # Calculate cosine similarity (1 - cosine similarity gives cosine distance)
                    cosine_differences[f"weight_changes_cossim/{name}"] = cosine_similarity(parameter.view(1, -1), previous_weights[name].view(1, -1)).item()

                # Update the previous weights with the current weights
                previous_weights[name] = parameter.clone().detach()


            # Check if the parameter is the in_proj_weight in a MultiheadAttention layer
            if 'in_proj_weight' in name:
                # Assuming `embed_dim` can be inferred from the size of in_proj_weight
                embed_dim = parameter.size(1)

                # Separating in_proj_weight into q, k, and v weights
                q_weight, k_weight, v_weight = parameter.chunk(3, dim=0)

                # Computing the norms
                q_norm = q_weight.norm().item()
                k_norm = k_weight.norm().item()
                v_norm = v_weight.norm().item()

                # Adding the norms to the weight_norms dictionary
                weight_norms[f"in_proj_weight_norm/{name}_q_norm"] = q_norm
                weight_norms[f"in_proj_weight_norm/{name}_k_norm"] = k_norm
                weight_norms[f"in_proj_weight_norm/{name}_v_norm"] = v_norm

                # Save the current parameters for next iteration comparison
                name = f'{name}_q_norm'
                parameter = q_weight
                if name in previous_weights:
                    l2_differences[f"weight_changes_norm/{name}"] = torch.norm(parameter - previous_weights[name]).item()
                    cosine_differences[f"weight_changes_cossim/{name}"] = cosine_similarity(parameter.view(1, -1), previous_weights[name].view(1, -1)).item()
                previous_weights[name] = parameter.clone().detach()

                # Save the current parameters for next iteration comparison
                name = f'{name}_k_norm'
                parameter = k_weight
                if name in previous_weights:
                    l2_differences[f"weight_changes_norm/{name}"] = torch.norm(parameter - previous_weights[name]).item()
                    cosine_differences[f"weight_changes_cossim/{name}"] = cosine_similarity(parameter.view(1, -1), previous_weights[name].view(1, -1)).item()
                previous_weights[name] = parameter.clone().detach()

                # Save the current parameters for next iteration comparison
                name = f'{name}_v_norm'
                parameter = v_weight
                if name in previous_weights:
                    l2_differences[f"weight_changes_norm/{name}"] = torch.norm(parameter - previous_weights[name]).item()
                    cosine_differences[f"weight_changes_cossim/{name}"] = cosine_similarity(parameter.view(1, -1), previous_weights[name].view(1, -1)).item()
                previous_weights[name] = parameter.clone().detach()


        # Update weights
        optimizer.step()
        scheduler.step()

        if l2_differences:
            total_l2_diff = sum(l2_differences.values()) / len(l2_differences.values())
            l2_differences["weight_changes_norm/total"] = total_l2_diff

        if cosine_differences:
            total_cos_diff = sum(cosine_differences.values()) / len(cosine_differences.values())
            cosine_differences["weight_changes_cossim/total"] = total_cos_diff

        metrics = {
            "training/accuracy": acc / len(labels),
            "training/top5_accuracy": top5_correct / len(labels),
            "training/top10_accuracy": top10_correct / len(labels),
            "training/loss": loss,
            "training/real_mse": mse,
            "step": wandb.run.step,
            **gradient_norms,
            **weight_norms,
            **gradient_changes,
            **l2_differences,
            **cosine_differences,
        }
        wandb.log(metrics)

        # Accumulate total accuracy
        total_acc += acc
        total_count += labels.size(0)


        # # Finish training at maximum gradient updates
        # if wandb.run.step >= num_steps:
        #     break

    # Calculate overall training accuracy
    overall_training_accuracy = total_acc / total_count

    return overall_training_accuracy

# You should replace placeholders like wandb.run.step with actual control logic for step counting and termination if not using wandb.
def evaluate(model, val_loader, device, epoch):
    # Set model to evaluation mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    top5_correct = 0
    top10_correct = 0
    total_count = 0
    loss = 0.0
    mse = 0.0

    # Loop over each batch from the validation set
    for batch in val_loader:
        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Forward pass
        with torch.no_grad():
            output = model(inputs)[-1, :, :]

            predictions = torch.argmax(output, dim=1)
            top5_probs, top5_preds = output.topk(5, dim=1)
            top10_probs, top10_preds = output.topk(10, dim=1)

            correct += (predictions == labels).float().sum().item()
            top5_correct += sum([labels[i].item() in top5_preds[i] for i in range(len(labels))])
            top10_correct += sum([labels[i].item() in top10_preds[i] for i in range(len(labels))])

            loss += criterion(output, labels).item() * len(labels)
            mse += F.mse_loss(predictions.float(), labels.float()).item() * len(labels)
            total_count += len(labels)

    acc = correct / total_count
    top5_accuracy = top5_correct / total_count
    top10_accuracy = top10_correct / total_count
    loss = loss / total_count
    mse = mse / total_count

    metrics = {
        "validation/accuracy": acc,
        "validation/top5_accuracy": top5_accuracy,
        "validation/top10_accuracy": top10_accuracy,
        "validation/loss": loss,
        "validation/real_mse": mse,
        "epoch": epoch
    }
    wandb.log(metrics, commit=False)

    return acc
