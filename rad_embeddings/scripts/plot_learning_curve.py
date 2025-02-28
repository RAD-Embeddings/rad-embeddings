import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# --- Step 1: Collect all txt files (each file is a seed) ---
file_list = glob.glob("exps/*.txt")
if not file_list:
    raise FileNotFoundError("No txt files found in the current directory.")

# --- Step 2: Extract 'ep_rew_disc_mean' and 'total_timesteps' values from each file ---
all_rewards = []
all_timesteps = []

for filename in file_list:
    rewards = []    # Discounted rewards for this seed
    timesteps = []  # Timesteps for this seed

    with open(filename, "r") as f:
        for line in f:
            if "total_timesteps" in line:
                # Expected line format: "|    total_timesteps     | 1000       |"
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        timestep = int(parts[2].strip())
                        timesteps.append(timestep)
                    except ValueError:
                        continue
            elif "ep_rew_disc_mean" in line:
                # Expected line format: "|    ep_rew_disc_mean     | 0.0769       |"
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        reward = float(parts[2].strip())
                        rewards.append(reward)
                    except ValueError:
                        continue

    if rewards and timesteps and len(rewards) == len(timesteps):
        all_rewards.append(rewards)
        all_timesteps.append(timesteps)
    else:
        print(f"Warning: Mismatch or missing data in file {filename}")

if not all_rewards:
    raise ValueError("No valid rewards and timesteps were extracted from any file.")

# --- Step 3: Align the data ---
# Use the minimum length across all seeds
min_length = min(len(r) for r in all_rewards)
all_rewards = [r[:min_length] for r in all_rewards]
all_timesteps = [t[:min_length] for t in all_timesteps]

# Convert to NumPy arrays
rewards_data = np.array(all_rewards)   # Shape: (num_seeds, num_points)
timesteps_data = np.array(all_timesteps) # Shape: (num_seeds, num_points)

# --- Step 4: Compute statistics (mean and 90% confidence intervals) ---
mean_rewards = np.mean(rewards_data, axis=0)
std_rewards = np.std(rewards_data, axis=0, ddof=1)
num_seeds = rewards_data.shape[0]
se_rewards = std_rewards / np.sqrt(num_seeds)

# 90% confidence interval using the t-distribution
t_value = st.t.ppf(0.95, df=num_seeds - 1)
ci = t_value * se_rewards

# Compute mean timesteps across seeds (they should be similar across seeds)
mean_timesteps = np.mean(timesteps_data, axis=0)

# Lower and upper bounds of the confidence interval
lower_bound = mean_rewards - ci
upper_bound = mean_rewards + ci

# --- Step 5: Plot the learning curve with a shaded confidence region ---
plt.figure(figsize=(8, 5))
plt.plot(mean_timesteps, mean_rewards, label='Mean Reward', color='blue')
plt.fill_between(mean_timesteps, lower_bound, upper_bound, color='blue', alpha=0.3,
                 label='90% Confidence Interval')
plt.xlabel("Total Timesteps")
plt.ylabel("Discounted Reward Mean")
plt.title("Learning Curve with 90% Confidence Interval")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Step 6: Save the plot as a PDF ---
pdf_filename = "exps/learning_curve.pdf"
plt.savefig(pdf_filename, format='pdf')
print(f"Plot saved as {pdf_filename}")

plt.show()
