import gymnasium as gym
import numpy as np
import sympy  # For prime checking
import matplotlib.pyplot as plt
import matplotlib.patches as patches  
# from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap  
from IPython.display import clear_output
import time
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

class PrimeExplorerEnv(gym.Env):
    """
    A custom environment for exploring prime numbers.
    The agent starts at a given number and moves along the number line.
    Reward: +1 for landing on a prime, -0.1 otherwise.
    """
    metadata = {'render_modes': ['human', 'visualization']}
    
    def __init__(self, start=2, max_num=10000, max_steps=200, render_mode="human"):
        super().__init__()
        self.start = start
        self.current = start
        self.max_num = max_num
        self.max_steps = max_steps
        self.steps = 0
        self.prime_count = 0
        self.total_count = 0
        self.prime_percentage = 0
        
        # Define the action space:
        # 0: step +1, 1: step +2, 2: step +5, 3: step +10
        self.action_space = gym.spaces.Discrete(4)
        
        # The state is the current number; we use a Box with shape (1,)
        self.observation_space = gym.spaces.Box(
            low=np.array([self.start], dtype=np.int32),
            high=np.array([self.max_num], dtype=np.int32),
            shape=(1,),
            dtype=np.int32
        )
        
        self.render_mode = render_mode
        
        self.history = {
            'positions': [self.current],
            'actions': [],
            'rewards': [],
            'is_prime': [sympy.isprime(self.current)]
        }
        
        self.prime_cmap = LinearSegmentedColormap.from_list('prime_colors', ['#f5f5f5', '#ff6b6b'])
        self.action_colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']
        
        if self.render_mode == "visualization":
            plt.style.use('ggplot') 
            self.fig = plt.figure(figsize=(15, 10))
            self.fig.suptitle("Prime Number Explorer", fontsize=20, fontweight='bold')
            
            
            self.gs = self.fig.add_gridspec(3, 3)
            
            
            self.ax1 = self.fig.add_subplot(self.gs[0, :])  # Position vs Step (full width, top row)
            self.ax2 = self.fig.add_subplot(self.gs[1, :2])  # Cumulative Reward (2/3 width, middle row)
            self.ax3 = self.fig.add_subplot(self.gs[1, 2])   # Action Distribution (1/3 width, middle row)
            
            self.ax4 = self.fig.add_subplot(self.gs[2, 0])  # Prime density heatmap
            self.ax5 = self.fig.add_subplot(self.gs[2, 1:])  # Recent actions timeline
            
            plt.ion()  
            plt.tight_layout(rect=[0, 0, 1, 0.95])  
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current = self.start
        self.steps = 0
        self.prime_count = 0
        self.total_count = 0
        self.prime_percentage = 0
        
        
        self.history = {
            'positions': [self.current],
            'actions': [],
            'rewards': [],
            'is_prime': [sympy.isprime(self.current)]
        }
        
        return np.array([self.current], dtype=np.int32), {} 
    
    def step(self, action):
        self.steps += 1
        
        # Map action to step size
        step_sizes = [1, 2, 5, 10]
        step_size = step_sizes[action] if action < len(step_sizes) else 1
        
        self.current += step_size
        
        # Check if the new number is prime
        is_prime = sympy.isprime(self.current)
        
        # Update prime statistics
        self.total_count += 1
        if is_prime:
            self.prime_count += 1
            reward = 1.0
        else:
            reward = -0.1
        
        self.prime_percentage = (self.prime_count / self.total_count) * 100
        
        # Update history
        self.history['positions'].append(self.current)
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        self.history['is_prime'].append(is_prime)
        
        # Define termination conditions
        terminated = bool(self.current >= self.max_num)
        truncated = bool(self.steps >= self.max_steps)
        
        info = {
            "is_prime": is_prime,
            "step_size": step_size,
            "prime_percentage": self.prime_percentage
        }
        
        return np.array([self.current], dtype=np.int32), reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.steps} | Current number: {self.current} | Prime: {sympy.isprime(self.current)} | Success Rate: {self.prime_percentage:.2f}%")
        
        elif self.render_mode == "visualization":
            
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            self.ax5.clear()
            
            # Get history data
            positions = np.array(self.history['positions'])
            is_prime = np.array(self.history['is_prime'])
            steps = np.arange(len(positions))
            
            if len(self.history['rewards']):
                rewards = np.array(self.history['rewards'])
                cumulative_rewards = np.cumsum(rewards)
            else:
                rewards = np.array([])
                cumulative_rewards = np.array([])
            
            if len(self.history['actions']):
                actions = np.array(self.history['actions'])
                action_counts = np.bincount(actions, minlength=4)
            else:
                actions = np.array([])
                action_counts = np.zeros(4)
            
            # Plot 1: Position vs Step with prime highlights
            self.ax1.plot(steps, positions, '-', color='#3498db', alpha=0.7, linewidth=1.5, label='Position')
            
            # Color points based on whether they're prime
            self.ax1.scatter(steps[~is_prime], positions[~is_prime], color='#3498db', s=30, alpha=0.5)
            self.ax1.scatter(steps[is_prime], positions[is_prime], color='#e74c3c', s=80, marker='*', label='Prime')
            
            self.ax1.scatter([self.steps], [self.current], color='#2ecc71', s=150, marker='o', 
                            edgecolor='black', linewidth=2, label='Current')
        
            primes_nearby = []
            for i in range(max(1, self.current - 20), self.current + 21):
                if sympy.isprime(i):
                    primes_nearby.append(i)
            
            for prime in primes_nearby:
                self.ax1.axhline(y=prime, color='#e74c3c', linestyle='--', alpha=0.3, linewidth=0.8)
                
                
                if abs(prime - self.current) <= 10:
                    self.ax1.text(self.steps + 1, prime, f"{prime}", fontsize=9, color='#e74c3c', 
                                fontweight='bold', ha='left', va='center')
            
            
            self.ax1.set_xlabel('Step', fontsize=12)
            self.ax1.set_ylabel('Position', fontsize=12)
            current_status = "Prime" if sympy.isprime(self.current) else "Not Prime"
            self.ax1.set_title(f'Current: {self.current} ({current_status})', fontsize=14, fontweight='bold')
            
            
            prime_count = np.sum(is_prime)
            prime_density = (prime_count / len(positions)) * 100 if len(positions) > 0 else 0
            
            
            prime_percentage_text = f"Success Rate: {self.prime_percentage:.2f}%"
            self.ax1.text(0.98, 0.95, prime_percentage_text, transform=self.ax1.transAxes,
                         fontsize=12, fontweight='bold', ha='right', va='top',
                         bbox=dict(boxstyle="round,pad=0.3", fc="#f0e68c", ec="black", alpha=0.8))
            
            
            status_text = f"Position #{len(positions)} | Prime count: {prime_count} | Density: {prime_density:.1f}% | Success Rate: {self.prime_percentage:.2f}%"
            self.ax1.text(0.02, 0.95, status_text, transform=self.ax1.transAxes,
                         fontsize=10, ha='left', va='top',
                         bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="gray", alpha=0.8))
            
            
            self.ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)
            
            # Plot 2: Cumulative Reward with enhanced styling
            if len(cumulative_rewards) > 0:
                # Fill area under curve
                self.ax2.fill_between(np.arange(1, len(cumulative_rewards) + 1), 0, cumulative_rewards, 
                                     alpha=0.2, color='#27ae60')
                self.ax2.plot(np.arange(1, len(cumulative_rewards) + 1), cumulative_rewards, 
                             color='#27ae60', linewidth=2, label='Cumulative')
                
                
                if len(rewards) > 5:
                    window_size = min(10, len(rewards))
                    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    self.ax2.plot(np.arange(window_size, len(rewards) + 1), moving_avg, 
                                 color='#e67e22', linestyle='--', linewidth=2, label=f'{window_size}-step Avg')
                
               
                if len(rewards) > 0:
                    latest_reward = rewards[-1]
                    color = '#27ae60' if latest_reward > 0 else '#e74c3c'
                    self.ax2.plot(len(rewards), cumulative_rewards[-1], 'o', color=color, 
                                 markersize=10, markeredgecolor='black', markeredgewidth=1)
                
                
                if len(cumulative_rewards) > 1:
                    max_reward = np.max(cumulative_rewards)
                    max_idx = np.argmax(cumulative_rewards) + 1
                    self.ax2.plot(max_idx, max_reward, 'o', color='gold', markersize=8,
                                 markeredgecolor='black', markeredgewidth=1)
                    self.ax2.text(max_idx, max_reward, f"Max: {max_reward:.1f}", fontsize=9, 
                                 va='bottom', ha='center', color='#333')
                
                self.ax2.set_xlabel('Step', fontsize=12)
                self.ax2.set_ylabel('Reward', fontsize=12)
                self.ax2.set_title(f'Total Reward: {cumulative_rewards[-1]:.2f}', fontsize=14, fontweight='bold')
                
                
                if len(steps) > 1:  
                
                    ax2_twin = self.ax2.twinx()
                    
                    
                    prime_counts_cumulative = np.cumsum(is_prime)
                    success_rates = (prime_counts_cumulative / np.arange(1, len(prime_counts_cumulative) + 1)) * 100
                    
                    
                    ax2_twin.plot(steps, success_rates, color='purple', linewidth=2, linestyle='-.', label='Success Rate %')
                    ax2_twin.set_ylabel('Success Rate %', color='purple', fontsize=12)
                    ax2_twin.tick_params(axis='y', labelcolor='purple')
                    ax2_twin.set_ylim(0, 100)  # Set y-limit from 0 to 100%
                    
                    
                    lines, labels = self.ax2.get_legend_handles_labels()
                    lines2, labels2 = ax2_twin.get_legend_handles_labels()
                    ax2_twin.legend(lines + lines2, labels + labels2, loc='upper left')
                else:
                    self.ax2.legend(loc='upper left')
                
                
                self.ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Plot 3: Action Distribution with enhanced styling
            if len(actions) > 0:
                action_labels = ['Step +1', 'Step +2', 'Step +5', 'Step +10']
                bars = self.ax3.bar(action_labels, action_counts, color=self.action_colors, alpha=0.8)
                
               
                for bar, count in zip(bars, action_counts):
                    if count > 0:
                        self.ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                     str(int(count)), ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                action_pcts = action_counts / np.sum(action_counts) * 100
                for i, (bar, pct) in enumerate(zip(bars, action_pcts)):
                    if pct > 0:
                        self.ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                                     f"{pct:.1f}%", ha='center', va='center', fontsize=9, 
                                     fontweight='bold', color='white')
                
                if len(actions) > 0:
                    latest_action = actions[-1]
                    bars[latest_action].set_edgecolor('black')
                    bars[latest_action].set_linewidth(2)
                    bars[latest_action].set_alpha(1.0)
                
                self.ax3.set_xlabel('Action', fontsize=12)
                self.ax3.set_ylabel('Count', fontsize=12)
                self.ax3.set_title('Action Distribution', fontsize=14, fontweight='bold')
            
            # Plot 4: Prime Density Heatmap
            if len(positions) > 0:
                
                num_bins = min(20, len(positions) // 5 + 1)
                if num_bins > 1:
                    bin_edges = np.linspace(min(positions), max(positions), num_bins + 1)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    digitized = np.digitize(positions, bin_edges) - 1
                    digitized[digitized >= num_bins] = num_bins - 1  
                    
                    
                    prime_in_bin = np.zeros(num_bins)
                    count_in_bin = np.zeros(num_bins)
                    
                    for i, bin_idx in enumerate(digitized):
                        if bin_idx >= 0 and bin_idx < num_bins:
                            count_in_bin[bin_idx] += 1
                            if is_prime[i]:
                                prime_in_bin[bin_idx] += 1
                    
                   
                    with np.errstate(divide='ignore', invalid='ignore'):
                        density = np.divide(prime_in_bin, count_in_bin)
                        density = np.nan_to_num(density)
                    
                    
                    self.ax4.bar(bin_centers, density, width=(bin_edges[1]-bin_edges[0])*0.8,
                                color=plt.cm.YlOrRd(density), edgecolor='gray', linewidth=0.5)
                    
                    
                    self.ax4.set_xlabel('Number Range', fontsize=12)
                    self.ax4.set_ylabel('Prime Density', fontsize=12)
                    self.ax4.set_title('Prime Number Density', fontsize=14, fontweight='bold')
                    
                    if max(positions) > 10:
                        # Approximate prime density using Prime Number Theorem: 1/ln(n)
                        x_range = np.linspace(max(2, min(positions)), max(positions), 100)
                        prime_density_approx = 1 / np.log(x_range)
                        self.ax4.plot(x_range, prime_density_approx, 'b--', linewidth=1.5, 
                                     label='1/ln(n)', alpha=0.7)
                        self.ax4.legend(loc='upper right', fontsize=9)
                else:
                    self.ax4.text(0.5, 0.5, "Insufficient data\nfor density analysis", 
                                 ha='center', va='center', fontsize=12)
                    
            # Plot 5: Recent Actions Timeline
            if len(actions) > 0:
                num_recent = min(20, len(actions))
                recent_actions = actions[-num_recent:]
                recent_rewards = rewards[-num_recent:]
                recent_is_prime = is_prime[-num_recent:]
                recent_steps = steps[-num_recent:]
                recent_positions = positions[-num_recent:]

                if len(recent_positions) > 1:  
                    for i in range(len(recent_actions)):
                        idx = num_recent - 1 - i  
                        
                        
                        if idx > 0 and idx < len(recent_positions) and (idx+1) < len(recent_positions):
                            x_pos = num_recent - 1 - idx
                            
                            
                            self.ax5.plot([x_pos, x_pos+1], 
                                         [recent_positions[idx], recent_positions[idx+1]], 
                                         '-', color=self.action_colors[recent_actions[i]], 
                                         linewidth=2, alpha=0.7)
                            
                            
                            arrow_len = 0.4
                            mid_x = x_pos + 0.5
                            mid_y = (recent_positions[idx] + recent_positions[idx+1]) / 2
                            dx = arrow_len
                            self.ax5.arrow(mid_x - dx/2, mid_y, dx, 0, 
                                          head_width=abs(recent_positions[idx+1] - recent_positions[idx])*0.1 + 0.5, 
                                          head_length=0.1, fc=self.action_colors[recent_actions[i]], 
                                          ec=self.action_colors[recent_actions[i]])
                
                
                for i, (pos, is_p) in enumerate(zip(recent_positions, recent_is_prime)):
                    idx = min(num_recent - 1 - i, len(recent_positions) - 1) 
                    x_pos = num_recent - 1 - i
                    marker = '*' if is_p else 'o'
                    size = 120 if is_p else 80
                    color = '#e74c3c' if is_p else '#3498db'
                    self.ax5.scatter(x_pos, pos, marker=marker, s=size, color=color, 
                                    edgecolor='black', linewidth=1, zorder=10)
                    
                    
                    self.ax5.text(x_pos, pos, str(pos), fontsize=8, ha='center', va='bottom',
                                 color='black', fontweight='bold' if is_p else 'normal')
                
                
                action_names = ['Step +1', 'Step +2', 'Step +5', 'Step +10']
                legend_elements = []
                for i, (name, color) in enumerate(zip(action_names, self.action_colors)):
                    legend_elements.append(patches.Patch(facecolor=color, edgecolor='gray',
                                                      label=name))
                self.ax5.legend(handles=legend_elements, loc='upper left', fontsize=9)
                
               
                self.ax5.set_xticks(range(min(num_recent, len(recent_steps))))
                step_labels = [str(step) for step in recent_steps]
                self.ax5.set_xticklabels(step_labels[::-1][:num_recent], rotation=45, fontsize=8)
                self.ax5.set_xlabel('Step', fontsize=12)
                self.ax5.set_ylabel('Position', fontsize=12)
                self.ax5.set_title('Recent Actions Timeline', fontsize=14, fontweight='bold')
                
           
                if len(actions) > 0:
                    last_action = actions[-1]
                    last_reward = rewards[-1]
                    result = "Prime!" if is_prime[-1] else "Not prime"
                    last_move_text = (f"Last move: {action_names[last_action]} → {positions[-1]} ({result})\n"
                                    f"Reward: {last_reward:.1f}")
                    self.ax5.text(0.5, -0.15, last_move_text, transform=self.ax5.transAxes,
                                 ha='center', va='top', fontsize=11,
                                 bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="gray", alpha=0.8))
            else:
                
                self.ax5.text(0.5, 0.5, "Waiting for actions...", ha='center', va='center', fontsize=12)
            
           
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.draw()
            plt.pause(0.01)
    
    def close(self):
        if self.render_mode == "visualization":
            plt.close(self.fig)
            plt.ioff()

if __name__ == '__main__':
    print("Starting script...")
    from stable_baselines3 import DQN
    
    print("Creating environment...")
    env = PrimeExplorerEnv(start=2, max_num=1000, max_steps=100, render_mode="visualization")
    
    print("Initializing DQN agent...")
    model = DQN("MlpPolicy", env, verbose=1, device="cuda" if torch.cuda.is_available() else "cpu", 
                buffer_size=1000, learning_starts=100)
    
    def visualize_training_episode(env, model, episode_num):
        """Run a single episode with the current model and visualize it"""
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.05)  
            done = terminated or truncated
        plt.pause(0.5)  
    
    print("Training agent...")
    TOTAL_TIMESTEPS = 10000
    VISUALIZE_EVERY = 1000
    
    for i in range(0, TOTAL_TIMESTEPS, VISUALIZE_EVERY):
        model.learn(total_timesteps=VISUALIZE_EVERY, reset_num_timesteps=False)
        print(f"Trained for {i + VISUALIZE_EVERY} steps. Visualizing current policy...")
        visualize_training_episode(env, model, i // VISUALIZE_EVERY)
    
    print("Training complete! Testing final agent...")
    

    test_env = PrimeExplorerEnv(start=2, max_num=1000, max_steps=100, render_mode="visualization")
    obs, _ = test_env.reset()
    total_reward = 0
    
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        total_reward += reward
        time.sleep(0.1) 
        
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps!")
            break
    
    print(f"Final score: {total_reward:.2f}")
    
    # Keep the plot open at the end
    if test_env.render_mode == "visualization":
        plt.ioff()
        plt.show()
