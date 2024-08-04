import numpy as np
import matplotlib.pyplot as plt
from time import time

def better_argmax(arr):
    maxval = -float('inf')
    best_indices_arr =[]
    for index, val in enumerate(arr):
        if val > maxval:
            maxval = val
            best_indices_arr = [index]
        elif val == maxval:
            best_indices_arr.append(index)
    max_index = np.random.randint(low=0,high=len(best_indices_arr))
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
        self.Q_list[action] += Q_n + (reward - Q_n)/self.n

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
                random_action = random.randint(low=0,high=k+1)
            return random_action
        else:
            return greedy_action

def excersize_2dot5_experiment(global_k, global_epsilon, global_alpha):
    trials = 5
    time_window = 10000
    # global monitoring structs
    global_optimal_action_lst_agent_1 = []
    global_reward_lst_agent1 = []
    global_optimal_action_lst_agent_2 = []
    global_reward_lst_agent2 = []
    start_time = time()
    for trial in range(trials):

        # agent 1 bandit problem
        bandit = NonStationaryKArmedBandit(k=global_k)
        agent_1 = SampleAverageAgent(k=global_k, epsilon=global_epsilon)
        # trial monitoring structs
        trial_optimal_action_lst_agent_1 = [] # True at index t if optimal action was chosen at time t, False otherwise
        trial_reward_lst_agent_1 = [] # record of reward received at time t


        for global_time_step in range(time_window):
            # agents choose actions
            agent_1_action = agent_1.choose()
            # monitor chosen action
            trial_optimal_action_lst_agent_1.append(agent_1_action == better_argmax(bandit.bandit_list))
            # agents receives reward
            agent_1_reward = bandit.interact(agent_1_action)
            # monitor received reward
            trial_reward_lst_agent_1.append(agent_1_reward)

            # agent updates its internal understanding of environment
            agent_1.update(action=agent_1_action, reward=agent_1_reward)

            # environment updates it's own state
            bandit.time_step()

        if (trial % 100) == 0: print(f'Trial {trial} completed after {(time()-start_time):.2f}s.')

        # agent 1 bandit problem
        # bandit = NonStationaryKArmedBandit(k=global_k)
        # agent_2 = IncrementalComputeAgent(k=global_k, epsilon=global_epsilon, alpha=global_alpha)

        # trial_optimal_action_lst_agent_2 = [] # True at index t if optimal action was chosen at time t, False otherwise
        # trial_reward_lst_agent_2 = [] # record of reward received at time t
        # for global_time_step in range(time_window):
        #     bandit = NonStationaryKArmedBandit(k=global_k)
        #     # agents choose actions
        #     agent_2_action = agent_2.choose()
        #     # monitor chosen action
        #     trial_optimal_action_lst_agent_2.append(agent_2_action == better_argmax(bandit.bandit_list))
        #     # agents receives reward
        #     agent_2_reward = bandit.interact(agent_2_action)
        #     # monitor received reward
        #     trial_reward_lst_agent_2.append(agent_2_reward)
        #     # agent updates its internal understanding of environment
        #     agent_2.update(action=agent_1_action, reward=agent_2_reward)
        #     # environment updates it's own state
        #     bandit.time_step()

        # update global monitoring lists

    # plot final global monitoring structs
    plt.scatter([t for t in range(time_window)], [val/trials for val in global_optimal_action_lst_agent_1])
    # plt.scatter([t for t in range(time_window)], [val/trials for val in global_optimal_action_list_agent2])
    # plt.scatter([t for t in range(time_window)], [val/trials for val in global_reward_lst_agent1])
    # plt.scatter([t for t in range(time_window)], [val/trials for val in global_reward_lst_agent2])
    plt.xlabel('Time Step')
    plt.ylabel('Percent Optimal Action')
    # plt.ylabel('Running Average of Reward over time')
    plt.ylim(0,1)
    plt.show()

excersize_2dot5_experiment(global_k=10, global_epsilon=0.1, global_alpha=0.1)
