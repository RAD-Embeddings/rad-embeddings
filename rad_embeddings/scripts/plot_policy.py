import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def read_experiment_data(directory):
    # --- Step 1: Collect all txt files from the given directory ---
    file_list = glob.glob(f"{directory}/*.txt")
    if not file_list:
        raise FileNotFoundError(f"No txt files found in directory: {directory}")
    
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
        raise ValueError(f"No valid rewards and timesteps were extracted from directory: {directory}")
    
    # --- Step 3: Align the data across all seeds ---
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
    
    mean_timesteps = np.mean(timesteps_data, axis=0)
    lower_bound = mean_rewards - ci
    upper_bound = mean_rewards + ci
    
    return mean_timesteps, mean_rewards, lower_bound, upper_bound

# Increase font sizes for all plot elements
plt.rcParams.update({'font.size': 24})

# Directories for the two experiments
dir_exp = "exps/token_env"
dir_baseline = "exps_baseline/token_env"

# Read and process data from both directories
timesteps_exp, rewards_exp, lower_exp, upper_exp = read_experiment_data(dir_exp)
timesteps_base, rewards_base, lower_base, upper_base = read_experiment_data(dir_baseline)

# --- Step 5: Plot the learning curves for both experiments ---
plt.figure(figsize=(10, 8))
# Plot for the first experiment
plt.plot(timesteps_exp, rewards_exp, label='Bisimulation Metrics', color='blue')
plt.fill_between(timesteps_exp, lower_exp, upper_exp, color='blue', alpha=0.3,
                 label='Bisimulation Metrics 90% CI')
# Plot for the baseline
plt.plot(timesteps_base, rewards_base, label='DFA Solving', color='red')
plt.fill_between(timesteps_base, lower_base, upper_base, color='red', alpha=0.3,
                 label='DFA Solving 90% CI')

plt.xlabel("Total Timesteps")
plt.ylabel("Discounted Reward Mean")
plt.title("Policy Learning with DFA Embeddings")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Step 6: Save the plot as a PDF with reduced margins ---
pdf_filename = "policy_learning_curve_comparison.pdf"
plt.savefig(pdf_filename, format='pdf', bbox_inches="tight", pad_inches=0.1)
print(f"Plot saved as {pdf_filename}")

plt.show()
