import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# --- Step 1: Collect all txt files (each file is a seed) ---
file_list = glob.glob("exps/*.txt")
if not file_list:
    raise FileNotFoundError("No txt files found in the specified directory (exps/).")

# --- Step 2: Extract metrics from each file ---
# Expected log format per line:
#   | <key>                | <value>       |
# We'll extract keys: total_timesteps, loss, policy_gradient_loss, and value_loss.
all_timesteps = []
all_loss = []
all_policy_loss = []
all_value_loss = []

for filename in file_list:
    timesteps = []
    loss_values = []
    policy_loss_values = []
    value_loss_values = []
    
    with open(filename, "r") as f:
        for line in f:
            # Skip lines that don't look like log entries
            if "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue

            key = parts[1].strip()
            value_str = parts[2].strip()

            if key == "total_timesteps":
                try:
                    ts = int(value_str)
                    timesteps.append(ts)
                except ValueError:
                    continue
            elif key == "loss":
                try:
                    value = float(value_str)
                    loss_values.append(value)
                except ValueError:
                    continue
            elif key == "policy_gradient_loss":
                try:
                    value = float(value_str)
                    policy_loss_values.append(value)
                except ValueError:
                    continue
            elif key == "value_loss":
                try:
                    value = float(value_str)
                    value_loss_values.append(value)
                except ValueError:
                    continue

    # --- Handle extra total_timesteps log at the beginning ---
    # Sometimes the first total_timesteps appears before any loss is computed.
    # In that case, there will be one more timestep than loss entries.
    if (len(timesteps) == len(loss_values) + 1 and 
        len(timesteps) == len(policy_loss_values) + 1 and 
        len(timesteps) == len(value_loss_values) + 1):
        timesteps = timesteps[1:]
    
    if (len(timesteps) == len(loss_values) == len(policy_loss_values) == len(value_loss_values)
            and len(timesteps) > 0):
        all_timesteps.append(timesteps)
        all_loss.append(loss_values)
        all_policy_loss.append(policy_loss_values)
        all_value_loss.append(value_loss_values)
    else:
        print(f"Warning: Mismatched or missing data in file {filename}. Skipping this file.")

if not all_timesteps:
    raise ValueError("No valid data extracted from any file.")

# --- Step 3: Align data across seeds (trim all arrays to the minimum length) ---
min_length = min(len(ts) for ts in all_timesteps)
all_timesteps = [ts[:min_length] for ts in all_timesteps]
all_loss = [l[:min_length] for l in all_loss]
all_policy_loss = [pl[:min_length] for pl in all_policy_loss]
all_value_loss = [vl[:min_length] for vl in all_value_loss]

# Convert lists to NumPy arrays for computation.
timesteps_data = np.array(all_timesteps)    # shape: (num_seeds, min_length)
loss_data = np.array(all_loss)
policy_loss_data = np.array(all_policy_loss)
value_loss_data = np.array(all_value_loss)

# --- Step 4: Compute statistics (mean and 90% confidence intervals) for each metric ---
num_seeds = timesteps_data.shape[0]
mean_timesteps = np.mean(timesteps_data, axis=0)

def compute_stats(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)
    se = std / np.sqrt(num_seeds)
    # For a two-sided 90% confidence interval, use the 95th percentile of the t-distribution.
    t_value = st.t.ppf(0.95, df=num_seeds - 1)
    ci = t_value * se
    return mean, mean - ci, mean + ci

loss_mean, loss_lower, loss_upper = compute_stats(loss_data)
policy_loss_mean, policy_loss_lower, policy_loss_upper = compute_stats(policy_loss_data)
value_loss_mean, value_loss_lower, value_loss_upper = compute_stats(value_loss_data)

# --- Step 5: Plot the metrics with total_timesteps as x-axis ---
plt.figure(figsize=(10, 6))

# Plot generic loss
plt.plot(mean_timesteps, loss_mean, label='Loss', color='red')
plt.fill_between(mean_timesteps, loss_lower, loss_upper, color='red', alpha=0.3)

# Plot policy gradient loss
plt.plot(mean_timesteps, policy_loss_mean, label='Policy Gradient Loss', color='green')
plt.fill_between(mean_timesteps, policy_loss_lower, policy_loss_upper, color='green', alpha=0.3)

# Plot value loss
plt.plot(mean_timesteps, value_loss_mean, label='Value Loss', color='blue')
plt.fill_between(mean_timesteps, value_loss_lower, value_loss_upper, color='blue', alpha=0.3)

plt.xlabel("Total Timesteps")
plt.ylabel("Loss")
plt.title("Loss, Policy Gradient Loss, and Value Loss over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Step 6: Save the plot as a PDF ---
pdf_filename = "exps/loss_metrics.pdf"
plt.savefig(pdf_filename, format='pdf')
print(f"Plot saved as {pdf_filename}")

plt.show()
