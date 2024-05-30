import wandb
import matplotlib.pyplot as plt
import numpy as np
import os


def fetch_run_data(project_path, run_id, metric):
    api = wandb.Api()
    run = api.run(f"{project_path}/{run_id}")
    history = run.scan_history(keys=["training/"+metric, "validation/"+metric])
    return history

def extract_accuracies(history, metric):
    training_accuracies = []
    validation_accuracies = []
    steps = []

    for data in history:
        # print(data)
        if "training/"+metric in data and "validation/"+metric in data:
            training_accuracies.append(data["training/"+metric])
            validation_accuracies.append(data["validation/"+metric])
            # steps.append(data["step"])

    return steps, training_accuracies, validation_accuracies

def plot_accuracies(steps, training_accuracies, validation_accuracies, filename, metric):
    plt.figure(figsize=(4, 3))
    plt.plot(training_accuracies, label='Training '+metric)
    plt.plot(validation_accuracies, label='Validation '+metric)
    plt.xlabel('Steps')
    plt.ylabel(metric)
    plt.xscale('log')
    plt.title(f'{metric} over steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, format = 'pdf', bbox_inches = 'tight')
    # plt.show()

def main(run_id, metric):
    # Configuration
    project_path = 'huanran-research/grokking/'
    run_dir = f"visualized_runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    filename = f"visualized_runs/{run_id}/{metric}_plot.pdf"

    # Execution
    history = fetch_run_data(project_path, run_id, metric)
    steps, training_accuracies, validation_accuracies = extract_accuracies(history, metric)
    plot_accuracies(steps, training_accuracies, validation_accuracies, filename, metric)

if __name__ == "__main__":
    # best_runs_id = ['1tmpr0ij', 'i11vuz8c', 'f7o6q7rv', 'i2td5wj8', 'u704bg05', 'fyj64sg3', 't8wb6zay', 'aksnn8hb', 'ikhv2x9n', 'efnutu2l']
    # worst_runs_id = ['770wt2tg', '742kf75w', 'u64kpwd0', 'h25z2x6o', 'yddb7b23', '5o4jsmt1', 'm2tegcvn', '1vyham6z', '4wi88gom', '10u74oy8']
    #
    # runs_id = best_runs_id + worst_runs_id
    #
    # for run_id in runs_id:
    #     print(run_id)
    #     main(run_id)
    id = '1vyham6z'
    main(id, 'accuracy')
    main(id, 'loss')
