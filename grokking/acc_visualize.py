import wandb
import matplotlib.pyplot as plt
import numpy as np


def fetch_run_data(project_path, run_id):
    api = wandb.Api()
    run = api.run(f"{project_path}/{run_id}")
    history = run.scan_history(keys=["training/accuracy", "validation/accuracy"])
    return history

def extract_accuracies(history):
    training_accuracies = []
    validation_accuracies = []
    steps = []

    for data in history:
        # print(data)
        if "training/accuracy" in data and "validation/accuracy" in data:
            training_accuracies.append(data["training/accuracy"])
            validation_accuracies.append(data["validation/accuracy"])
            # steps.append(data["step"])

    return steps, training_accuracies, validation_accuracies

def plot_accuracies(steps, training_accuracies, validation_accuracies, filename):
    plt.figure(figsize=(7, 5))
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.title('Training and Validation Accuracy over Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, format = 'pdf', bbox_inches = 'tight')
    # plt.show()

def main(run_id):
    # Configuration
    project_path = 'huanran-research/grokking/'
    filename = f"visualized_runs/{run_id}/accuracy_plot.pdf"

    # Execution
    history = fetch_run_data(project_path, run_id)
    steps, training_accuracies, validation_accuracies = extract_accuracies(history)
    plot_accuracies(steps, training_accuracies, validation_accuracies, filename)

if __name__ == "__main__":
    best_runs_id = ['1tmpr0ij', 'i11vuz8c', 'f7o6q7rv', 'i2td5wj8', 'u704bg05', 'fyj64sg3', 't8wb6zay', 'aksnn8hb', 'ikhv2x9n', 'efnutu2l']
    worst_runs_id = ['770wt2tg', '742kf75w', 'u64kpwd0', 'h25z2x6o', 'yddb7b23', '5o4jsmt1', 'm2tegcvn', '1vyham6z', '4wi88gom', '10u74oy8']

    runs_id = best_runs_id + worst_runs_id

    for run_id in runs_id:
        print(run_id)
        main(run_id)
