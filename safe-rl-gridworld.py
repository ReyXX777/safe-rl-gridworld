import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import trange

class SafeExplorationEnv:
    def __init__(self, size=5, n_messes=3, max_steps=50, danger_zones=None, seed=None):
        self.size = size
        self.n_messes = n_messes
        self.max_steps = max_steps
        self.danger_zones = danger_zones if danger_zones is not None else set()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.reset() 

    def is_unsafe(self, pos, buffer=1):
        return any(max(abs(pos[0]-dz[0]), abs(pos[1]-dz[1])) <= buffer for dz in self.danger_zones)

    def reset(self):
        self.agent_pos = (0, 0)
        while self.is_unsafe(self.agent_pos, buffer=0):
            self.agent_pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        self.messes = set()
        while len(self.messes) < self.n_messes:
            pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if pos != self.agent_pos and not self.is_unsafe(pos, buffer=0):
                self.messes.add(pos)
        self.steps = 0
        return self.get_state()  # return initial state

    def get_state(self):
        return (self.agent_pos, frozenset(self.messes))  # return hashable state

    def step(self, action):
        self.steps += 1
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        dx, dy = moves[action]
        new_pos = (int(np.clip(self.agent_pos[0]+dx,0,self.size-1)),
                   int(np.clip(self.agent_pos[1]+dy,0,self.size-1)))
        if self.is_unsafe(new_pos):
            return self.get_state(), -25.0, True, True  # unsafe penalty
        self.agent_pos = new_pos
        reward = -0.15
        unsafe = False
        if self.agent_pos in self.messes:
            reward += 12.0
            self.messes.remove(self.agent_pos)
        done = (len(self.messes)==0) or (self.steps>=self.max_steps)
        return self.get_state(), reward, unsafe, done  # return next state

def safe_q_train(n_episodes=12000, alpha=0.08, gamma=0.98, danger_zones=None, seed=42):
    env = SafeExplorationEnv(size=5, n_messes=2, max_steps=40, danger_zones=danger_zones, seed=seed)
    Q = defaultdict(float)
    epsilon_start, epsilon_end = 1.0, 0.015
    decay_episodes = n_episodes * 0.6
    rewards_hist, violations, cleaned_per_episode = [], [], []

    for ep in trange(n_episodes, desc="Safe Q-Learning"):
        state = env.reset()
        done = False
        ep_reward, ep_violations, messes_cleaned = 0.0, 0, 0
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start-epsilon_end)*(ep/decay_episodes))

        while not done:
            action = random.randint(0,3) if random.random()<epsilon else np.argmax([Q[(state,a)] for a in range(4)])
            next_state, reward, unsafe, done = env.step(action)
            if unsafe: ep_violations += 1
            if reward>=10: messes_cleaned += 1
            best_next = 0 if done else max(Q[(next_state,a)] for a in range(4))
            Q[(state,action)] += alpha*(reward + gamma*best_next*(0 if unsafe else 1)-Q[(state,action)])
            state = next_state
            ep_reward += reward

        rewards_hist.append(ep_reward)
        violations.append(ep_violations)
        cleaned_per_episode.append(messes_cleaned)

    return rewards_hist, violations, cleaned_per_episode

if __name__ == "__main__":
    danger_zones = {(2,2),(4,0)}
    rewards, violations, cleaned = safe_q_train(n_episodes=12000, danger_zones=danger_zones, seed=42)

    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(11,10),sharex=True)
    ax1.plot(np.convolve(rewards,np.ones(150)/150,mode='valid'), color='teal', lw=1.4, label='Reward (150-ep MA)')
    ax1.set_title("Safe Q-Learning Progress")
    ax1.set_ylabel("Episode Reward")
    ax1.grid(alpha=0.25)
    ax1.legend()
    ax2.plot(np.convolve(violations,np.ones(150)/150,mode='valid'), color='darkred', lw=1.3, label='Safety Violations (150-ep MA)')
    ax2.set_ylabel("Violations")
    ax2.grid(alpha=0.25)
    ax2.legend()
    ax3.plot(np.convolve(cleaned,np.ones(150)/150,mode='valid'), color='darkgreen', lw=1.3, label='Messes Cleaned (150-ep MA)')
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Cleaned")
    ax3.grid(alpha=0.25)
    ax3.legend()
    plt.tight_layout()
    plt.show()
