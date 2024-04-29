import wandb
import json
import ast
from loss_contour_calculate import *

def load_results_from_file(filename):
    """Loads results from a JSON file and converts string keys back to tuple keys."""
    with open(filename, 'r') as f:
        results_str_keys = json.load(f)
    results = {ast.literal_eval(key): value for key, value in results_str_keys.items()}
    return results


def plot_loss_contours(results, save_path, title='Loss Contours'):
    # Prepare the grid data
    step_sizes = sorted(set(key[0] for key in results.keys()))  # Unique step sizes
    grid_train = np.zeros((len(step_sizes), len(step_sizes)))
    grid_test = np.zeros((len(step_sizes), len(step_sizes)))

    for (step1, step2), (train_loss, test_loss) in results.items():
        i = step_sizes.index(step1)
        j = step_sizes.index(step2)
        grid_train[i, j] = train_loss
        grid_test[i, j] = test_loss

    # Create a meshgrid for plotting
    step1, step2 = np.meshgrid(step_sizes, step_sizes)

    # Plotting
    formatted_steps = [f"{step:.2f}" for step in step_sizes]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title)

    # Training loss contour
    sns.heatmap(grid_train, ax=ax[0], xticklabels=formatted_steps, yticklabels=formatted_steps, annot=False)
    ax[0].set_title('Training Loss')
    ax[0].set_xlabel('Step Size Direction 1')
    ax[0].set_ylabel('Step Size Direction 2')

    # Testing loss contour
    sns.heatmap(grid_test, ax=ax[1], xticklabels=formatted_steps, yticklabels=formatted_steps, annot=False)
    ax[1].set_title('Testing Loss')
    ax[1].set_xlabel('Step Size Direction 1')
    ax[1].set_ylabel('Step Size Direction 2')

    plt.savefig(save_path + ".pdf", format = 'pdf', bbox_inches = 'tight')
    plt.show()



def download_wandb_run_files(run_path, file_names, base_dir='visualized_runs'):
    api = wandb.Api()

    # Split the run path to get the run_id as the folder name
    run_id = run_path.split('/')[-1]
    run_dir = os.path.join(base_dir, run_id)

    # Create the run directory if it doesn't exist
    os.makedirs(run_dir, exist_ok=True)

    downloaded_files = {'run_dir': run_dir}

    for file_name in file_names:
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

# Example usage:
def main():
    run_path = 'huanran-research/grokking/742kf75w'  # Replace with your actual run path
    filename = 'results_steps_5_range_1.0.json'  # Filename you want to download

    file_names_downloaded = download_wandb_run_files(run_path, [filename])

    results = load_results_from_file(file_names_downloaded[filename])

    plot_loss_contours(results, file_names_downloaded['run_dir'], title='Loss Contours')


if __name__ == "__main__":
    main()
