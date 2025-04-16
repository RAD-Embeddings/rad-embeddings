import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib.ticker as mticker

# Increase default font size for the entire plot
plt.rcParams.update({'font.size': 24})

# ======================
# 1. File Collection
# ======================
file_list = glob.glob("exps/*.txt")
if not file_list:
    raise FileNotFoundError("No log files found in the directory 'exps/'.")

# ======================
# 2. Data Containers
# ======================
# Learning curve data (rollout):
all_lc_ts = []  # total_timesteps for learning curve
all_lc_ep = []  # ep_rew_disc_mean values

# Loss metrics data (train):
all_loss_ts = []     # total_timesteps for losses
all_loss     = []    # generic loss
all_policy_loss = [] # policy gradient loss
all_value_loss  = [] # value loss

# ======================
# 3. Parse Each File
# ======================
# Assumes log entries in the format:
#   | key                   | value         |
for filename in file_list:
    lc_ts = []   # x-values for learning curve
    lc_ep = []   # ep_rew_disc_mean values

    loss_ts = []  # x-values for losses
    loss_vals = []         # generic loss
    policy_loss_vals = []  # policy gradient loss
    value_loss_vals = []   # value loss

    pending_lc_ep = None  # store ep_rew_disc_mean until next total_timesteps
    last_ts = None        # updated on each total_timesteps line

    with open(filename, "r") as f:
        for line in f:
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
                    last_ts = ts
                    if pending_lc_ep is not None:
                        lc_ts.append(ts)
                        lc_ep.append(pending_lc_ep)
                        pending_lc_ep = None
                except ValueError:
                    continue

            elif key == "ep_rew_disc_mean":
                try:
                    ep_val = float(value_str)
                    pending_lc_ep = ep_val
                except ValueError:
                    continue

            # Record loss metrics using the most recent total_timesteps.
            elif key == "loss":
                try:
                    loss_val = float(value_str)
                    if last_ts is not None:
                        loss_ts.append(last_ts)
                        loss_vals.append(loss_val)
                except ValueError:
                    continue
            elif key == "policy_gradient_loss":
                try:
                    pl_val = float(value_str)
                    policy_loss_vals.append(pl_val)
                except ValueError:
                    continue
            elif key == "value_loss":
                try:
                    vl_val = float(value_str)
                    value_loss_vals.append(vl_val)
                except ValueError:
                    continue

    # Sometimes an extra total_timesteps is present.
    if len(lc_ts) == len(lc_ep) + 1:
        lc_ts = lc_ts[1:]

    if (len(loss_ts) == len(loss_vals) == len(policy_loss_vals) == len(value_loss_vals)
            and len(loss_ts) > 0):
        all_loss_ts.append(loss_ts)
        all_loss.append(loss_vals)
        all_policy_loss.append(policy_loss_vals)
        all_value_loss.append(value_loss_vals)
    else:
        print(f"Warning: Mismatched or missing loss data in file {filename}. Skipping loss data for this file.")

    if (len(lc_ts) == len(lc_ep) and len(lc_ts) > 0):
        all_lc_ts.append(lc_ts)
        all_lc_ep.append(lc_ep)
    else:
        print(f"Warning: Mismatched or missing learning curve data in file {filename}. Skipping learning curve data for this file.")

if not all_lc_ts:
    raise ValueError("No valid learning curve data extracted from any file.")
if not all_loss_ts:
    raise ValueError("No valid loss data extracted from any file.")

# ======================
# 4. Align Data Across Seeds
# ======================
min_len_lc = min(len(ts) for ts in all_lc_ts)
all_lc_ts = [ts[:min_len_lc] for ts in all_lc_ts]
all_lc_ep = [ep[:min_len_lc] for ep in all_lc_ep]

min_len_loss = min(len(ts) for ts in all_loss_ts)
all_loss_ts = [ts[:min_len_loss] for ts in all_loss_ts]
all_loss     = [l[:min_len_loss] for l in all_loss]
all_policy_loss = [pl[:min_len_loss] for pl in all_policy_loss]
all_value_loss  = [vl[:min_len_loss] for vl in all_value_loss]

# Use common length between learning and loss data.
common_len = min(min_len_lc, min_len_loss)
all_lc_ts = [ts[:common_len] for ts in all_lc_ts]
all_lc_ep = [ep[:common_len] for ep in all_lc_ep]
all_loss_ts = [ts[:common_len] for ts in all_loss_ts]
all_loss     = [l[:common_len] for l in all_loss]
all_policy_loss = [pl[:common_len] for pl in all_policy_loss]
all_value_loss  = [vl[:common_len] for vl in all_value_loss]

lc_ts_data = np.array(all_lc_ts)       # (num_seeds, common_len)
lc_ep_data = np.array(all_lc_ep)
loss_ts_data = np.array(all_loss_ts)   # (num_seeds, common_len)
loss_data = np.array(all_loss)
policy_loss_data = np.array(all_policy_loss)
value_loss_data = np.array(all_value_loss)

# ======================
# 5. Compute Statistics (Mean & Full Variance)
# ======================
# Instead of a 90% CI, we compute the full variance (min and max) for each timestep.
def compute_stats(data, num_seeds):
    mean = np.mean(data, axis=0)
    lower_bound = np.min(data, axis=0)
    upper_bound = np.max(data, axis=0)
    return mean, lower_bound, upper_bound

num_seeds_lc = lc_ts_data.shape[0]
lc_x = np.mean(lc_ts_data, axis=0)
lc_ep_mean, lc_ep_lower, lc_ep_upper = compute_stats(lc_ep_data, num_seeds_lc)

num_seeds_loss = loss_ts_data.shape[0]
loss_x = np.mean(loss_ts_data, axis=0)
loss_mean, loss_lower, loss_upper = compute_stats(loss_data, num_seeds_loss)
policy_loss_mean, policy_loss_lower, policy_loss_upper = compute_stats(policy_loss_data, num_seeds_loss)
value_loss_mean, value_loss_lower, value_loss_upper = compute_stats(value_loss_data, num_seeds_loss)

# Assume the x-axes are similar.
combined_x = lc_x

# ======================
# 6. Create a Combined Plot with Twin Y-Axes and Legend Inside
# ======================
fig, ax1 = plt.subplots(figsize=(10, 8))
ax2 = ax1.twinx()

# Plot learning curve on left y-axis.
ax1.plot(combined_x, lc_ep_mean, label="Ep Rew Disc Mean", color="blue", linewidth=2)
ax1.fill_between(combined_x, lc_ep_lower, lc_ep_upper, color="blue", alpha=0.3)
ax1.set_xlabel("Total Timesteps")
ax1.set_ylabel("Discounted Reward Mean", color="black")
ax1.tick_params(axis='y', labelcolor="black")
ax1.grid(True)

# Plot loss metrics on right y-axis.
ax2.plot(loss_x, loss_mean, label="Loss", color="red", linewidth=2)
ax2.fill_between(loss_x, loss_lower, loss_upper, color="red", alpha=0.3)
ax2.plot(loss_x, policy_loss_mean, label="Policy Gradient Loss", color="green", linewidth=2)
ax2.fill_between(loss_x, policy_loss_lower, policy_loss_upper, color="green", alpha=0.3)
ax2.plot(loss_x, value_loss_mean, label="Value Loss", color="purple", linewidth=2)
ax2.fill_between(loss_x, value_loss_lower, value_loss_upper, color="purple", alpha=0.3)
ax2.set_ylabel("Loss Metrics", color="black")
ax2.tick_params(axis='y', labelcolor="black")

# Set right y-axis ticks: spacing at 0.05 with two decimal formatting.
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# Combine legends and place them inside the plot at center right with larger font.
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.9)

plt.title("Pretraining with Bisimulation Metrics")
plt.tight_layout()

# ======================
# 7. Save the Plot as a PDF with Reduced Margins
# ======================
pdf_filename = "combined_learning_and_losses_shaped.pdf"
plt.savefig(pdf_filename, format="pdf", bbox_inches="tight", pad_inches=0.1)
print(f"Plot saved as {pdf_filename}")

plt.show()
