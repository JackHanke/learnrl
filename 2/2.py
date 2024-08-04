import numpy as np
import matplotlib.pyplot as plt
from time import time

def better_argmax(arr, return_arr=False):
    maxval = -float('inf')
    best_indices_arr =[]
    for index, val in enumerate(arr):
        if val > maxval:
            maxval = val
            best_indices_arr = [index]
        elif val == maxval:
            best_indices_arr.append(index)
    max_index = np.random.randint(low=0,high=len(best_indices_arr))
    if return_arr: return best_indices_arr
    return best_indices_arr[max_index]

class StationaryKArmedBandit:
    def __init__(self, k):
        self.bandit_list = [np.random.normal(loc=0,scale=1) for _ in range(k)] # initialize equal starting point

    def interact(self, action): # return reward taken by specific action
        q_star_a = self.bandit_list[action]
        return np.random.normal(loc=q_star_a, scale=1)

class NonStationaryKArmedBandit:
    def __init__(self, k):
        self.k = k
        self.bandit_list = [0 for _ in range(k)] # initialize equal starting point

    def time_step(self):
        for action in range(self.k):
            self.bandit_list[action] += np.random.normal(loc=0,scale=0.001)

    def interact(self, action): # return reward taken by specific action
        q_star_a = self.bandit_list[action]
        return np.random.normal(loc=q_star_a, scale=1)

class SampleAverageAgent:
    def __init__(self, k, epsilon):
        self.n = 0 # time agent experiences
        self.k = k
        self.epsilon = epsilon
        self.Q_list = [0 for _ in range(k)]

    def update(self, action, reward):
        self.n += 1 
        Q_n = self.Q_list[action] 
        self.Q_list[action] += Q_n + (reward - Q_n)/(self.n)

    def choose(self):
        # print(f"self.Q_list = {self.Q_list}")
        greedy_action = better_argmax(self.Q_list)
        if np.random.uniform(0,1) < self.epsilon:
            random_action = greedy_action
            while random_action == greedy_action:
                random_action = np.random.randint(low=0,high=((self.k)))
            return random_action
        else:
            return greedy_action

class IncrementalComputeAgent:
    def __init__(self, k, epsilon, alpha):
        self.n = 0 # time agent experiences
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q_list = [0 for _ in range(k)]

    def update(self, action, reward):
        self.n += 1 
        Q_n = self.Q_list[action] 
        self.Q_list[action] += Q_n + (reward - Q_n)*(self.alpha)

    def choose(self):
        greedy_action = better_argmax(self.Q_list)
        if np.random.uniform(0,1) < self.epsilon:
            random_action = greedy_action
            while random_action == greedy_action:
                random_action = np.random.randint(low=0,high=self.k)
            return random_action
        else:
            return greedy_action

def excersize_2dot5_experiment(global_k, global_epsilon, global_alpha, reward_plot=False):
    trials = 100
    time_window = 10000
    # global monitoring structs
    global_optimal_action_lst = [[0 for _ in range(time_window)] for _ in range(2)]
    global_reward_lst = [[0 for _ in range(time_window)] for _ in range(2)]
    start_time = time()
    for trial in range(trials):
        # agent 1 bandit problem
        bandit = NonStationaryKArmedBandit(k=global_k)
        agent_lst = [
            SampleAverageAgent(k=global_k, epsilon=global_epsilon), 
            IncrementalComputeAgent(k=global_k, epsilon=global_epsilon, alpha=global_alpha)
        ]
        for agent_index, agent in enumerate(agent_lst):
            # trial monitoring structs
            for trial_time_step in range(time_window):
                # agents choose actions
                agent_action = agent.choose()
                # agents receives reward
                agent_reward = bandit.interact(agent_action)
                # agent updates its internal understanding of environment
                agent.update(action=agent_action, reward=agent_reward)
                # global monitoring
                global_optimal_action_lst[agent_index][trial_time_step] += (agent_action in better_argmax(bandit.bandit_list, return_arr=True))
                global_reward_lst[agent_index][trial_time_step] += agent_reward
                # environment updates it's own state
                bandit.time_step()
            if (trial % 100) == (100 - 1): print(f'Trial {trial} for Agent {agent_index} completed after {(time()-start_time):.2f}s.')

    # plot final global monitoring structs
    if not reward_plot:
        for action_agent_lst in global_optimal_action_lst:
            plt.scatter([t for t in range(time_window)], [val/trials for val in action_agent_lst])
            plt.xlabel('Time Step')
            plt.ylabel('Percent Optimal Action')
            plt.ylim(0,1)
    else:
        for action_agent_lst in global_reward_lst:
            plt.scatter([t for t in range(time_window)], [val/trials for val in action_agent_lst])
            plt.xlabel('Time Step')
            plt.ylabel('Average Reward')
    plt.show()

excersize_2dot5_experiment(global_k=10, global_epsilon=0.1, global_alpha=0.1)
