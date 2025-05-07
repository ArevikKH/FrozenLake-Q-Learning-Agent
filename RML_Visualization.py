import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results from the JSON file
with open('training_results.json', 'r') as f:
    results = json.load(f)

# Extract unique epsilon and num_epochs values
epsilon_values = sorted(set(result["epsilon"] for result in results))
num_epochs_values = sorted(set(result["num_epochs"] for result in results))

# Create matrices to store training times and rendering times
training_times_matrix = np.zeros((len(epsilon_values), len(num_epochs_values)))
rendering_times_matrix = np.zeros((len(epsilon_values), len(num_epochs_values)))

# Fill the matrices with training times and rendering times
for result in results:
    epsilon_index = epsilon_values.index(result["epsilon"])
    num_epochs_index = num_epochs_values.index(result["num_epochs"])
    training_times_matrix[epsilon_index, num_epochs_index] = result["execution_time"]
    rendering_times_matrix[epsilon_index, num_epochs_index] = result["render_time"]

# Create a figure with two subplots for training times and rendering times
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot training times
sns.heatmap(training_times_matrix, ax=axes[0], annot=True, fmt=".2f", cmap="plasma",
            xticklabels=num_epochs_values, yticklabels=epsilon_values)
axes[0].set_xlabel('Number of Epochs')
axes[0].set_ylabel('Epsilon')
axes[0].set_title('Training Times (seconds)')

# Plot rendering times
sns.heatmap(rendering_times_matrix, ax=axes[1], annot=True, fmt=".2f", cmap="plasma",
            xticklabels=num_epochs_values, yticklabels=epsilon_values)
axes[1].set_xlabel('Number of Epochs')
axes[1].set_ylabel('Epsilon')
axes[1].set_title('Rendering Times (seconds)')

plt.tight_layout()
plt.show()
