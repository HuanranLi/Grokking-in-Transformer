import wandb

import numpy as np
import matplotlib.pyplot as plt

def filter_and_process_runs(entity, project_name, filters, max_runs=5):
    api = wandb.Api()
    project_path = f"{entity}/{project_name}"
    runs = api.runs(path=project_path, filters=filters)

    artifact_paths = []
    configs = []
    summaries = []

    for run in runs:
        if len(configs) >= max_runs:
            break

        print(f"Processing Run: {run.id}")
        summaries.append(run.summary)
        configs.append(run.config)

    return summaries, configs



def analyze_runs(summaries, configs):

    grouped_runs = {}
    # Group by 'data/train_size' and 'batch_size'
    for summary, config in zip(summaries, configs):
        group_key = (summary.get('data/train_size'), config.get('batch_size'))
        print(group_key)
        if not all(group_key):
            print('skipped')
            continue

        if group_key not in grouped_runs:
            grouped_runs[group_key] = []
        grouped_runs[group_key].append(summary)

    categorized_results = {}
    step_delays = {}
    for key, summaries in grouped_runs.items():
        categories = {'confusion': 0, 'memorization': 0, 'grokking': 0}
        delays = []
        for summary in summaries:
            training_accuracy = summary.get('training/accuracy', 0)
            validation_accuracy = summary.get('validation/accuracy', 0)
            step_delay = summary.get('step_delay', 0)
            delays.append(step_delay)

            if training_accuracy < 95:
                categories['confusion'] += 1
            elif training_accuracy >= 95 and validation_accuracy < 95:
                categories['memorization'] += 1
            else:
                categories['grokking'] += 1

        # Determine majority category for this group
        majority_category = max(categories, key=categories.get)
        categorized_results[key] = majority_category
        if majority_category == 'grokking':
            step_delays[key] = np.mean(delays)

    return categorized_results, step_delays

def plot_results(results, step_delays):
    # Prepare plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()  # Get the current Axes instance on the current figure matching the given keyword args, or create one.

    colors = {'confusion': 'black', 'memorization': 'blue'}

    for key, category in results.items():
        train_size = key[0]
        batch_size = key[1]
        if category in colors:
            ax.scatter(train_size, batch_size, color=colors[category], label=category, s=100)
        elif category == 'grokking':
            delay = step_delays.get(key, 0)
            ax.scatter(train_size, batch_size, c=[delay], cmap='plasma_r', vmin=0, vmax=max(step_delays.values(), default=1), s=100)

    # Setting log scale for x and y axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Setting labels and title
    ax.set_xlabel('Train Size')
    ax.set_ylabel('Batch Size')
    ax.set_title('Run Categories by Train Size and Batch Size')

    # Adding color bar and legend
    plt.colorbar(ax.collections[-1], label='Average Step Delay for Grokking')
    plt.grid(True)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if legend_labels:  # If there are labels to display in legend.
        unique_labels = dict(zip(legend_labels, legend_handles))  # This removes duplicate labels.
        ax.legend(unique_labels.values(), unique_labels.keys())

    plt.show()


# Example usage
entity = "huanran-research"
project_name = "grokking"
max_runs = 10000
filters = {
    "state": {"$eq": "finished"},  # Correct: Filters runs that have finished
    "sweep": {"$eq": "cir87vhd"},  # Correct: Filters runs belonging to a specific sweep
}

summaries, configs = filter_and_process_runs(entity = entity, project_name = project_name, filters = filters, max_runs = max_runs)
results, step_delays = analyze_runs(summaries, configs)
print(results)
plot_results(results, step_delays)
