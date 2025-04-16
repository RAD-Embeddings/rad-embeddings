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
                    parts = line.split("|")
                    if len(parts) >= 3:
                        try:
                            timestep = int(parts[2].strip())
                            timesteps.append(timestep)
                        except ValueError:
                            continue
                elif "ep_rew_disc_mean" in line:
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
    
    rewards_data = np.array(all_rewards)   # (num_seeds, num_points)
    timesteps_data = np.array(all_timesteps) # (num_seeds, num_points)
    
    mean_rewards = np.mean(rewards_data, axis=0)
    # Compute full range (min-max across seeds) for each time point.
    min_rewards = np.min(rewards_data, axis=0)
    max_rewards = np.max(rewards_data, axis=0)
    
    mean_timesteps = np.mean(timesteps_data, axis=0)
    
    return mean_timesteps, mean_rewards, min_rewards, max_rewards

# Increase font sizes for all plot elements
plt.rcParams.update({'font.size': 24})

# Directories for the experiments
dir_exp = "exps/token_env"
dir_baseline = "exps_baseline/token_env"
dir_no_embed = "exps_no_embed"  # Updated directory name

# Read and process data from all directories
timesteps_exp, rewards_exp, min_exp, max_exp = read_experiment_data(dir_exp)
timesteps_base, rewards_base, min_base, max_base = read_experiment_data(dir_baseline)
timesteps_no_embed, rewards_no_embed, min_no_embed, max_no_embed = read_experiment_data(dir_no_embed)

# --- Step 5: Plot the learning curves for all experiments ---
plt.figure(figsize=(10, 8))
# Plot for Bisimulation Metrics
plt.plot(timesteps_exp, rewards_exp, label='Bisimulation Metrics', color='blue')
plt.fill_between(timesteps_exp, min_exp, max_exp, color='blue', alpha=0.2)
# Plot for DFA Solving baseline
plt.plot(timesteps_base, rewards_base, label='DFA Solving', color='red')
plt.fill_between(timesteps_base, min_base, max_base, color='red', alpha=0.2)
# Plot for No Embedding
plt.plot(timesteps_no_embed, rewards_no_embed, label='No Pretraining', color='green')
plt.fill_between(timesteps_no_embed, min_no_embed, max_no_embed, color='green', alpha=0.2)

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
