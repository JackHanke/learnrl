import numpy as np
import matplotlib.pyplot as plt

def better_argmax(arr):
    maxval = -float('inf')
    best_indices_arr =[]
    for index,val_lst in enumerate(arr):
        val = val_lst[0]
        if val > maxval:
            maxval = val
            best_indices_arr = [index]
        elif val == maxval:
            best_indices_arr.append(index)
    max_index = np.random.randint(low=0,high=len(best_indices_arr))
    return best_indices_arr[max_index]

def k_armed_bandit(k):
    bandit_list = []
    for bandit in range(k):
        q_star = np.random.normal(loc=0,scale=1)
        bandit_list.append(q_star)
    return bandit_list

def epsilon_greedy_run(k, bandit_list, T, epsilon):
    derived_q_a_lst = [[q_a,0] for q_a in range(k)] # estimated value, n(a) the number of times an action has been chosen over a run
    reward_lst = [] # reward over time
    for t in range(1,T+1):
        epsilon_rv = np.random.uniform(low=0,high=1)
        best_move = better_argmax(derived_q_a_lst)
        if epsilon_rv > epsilon:
            action = best_move
        elif epsilon_rv < epsilon:
            action = np.random.randint(low=0,high=k)

        reward = np.random.normal(loc=bandit_list[action],scale=1)

        derived_q_a_lst[action][1] += 1
        derived_q_a_lst[action][0] += (reward - derived_q_a_lst[action][0])/derived_q_a_lst[action][1]

        reward_lst.append(reward)

    return reward_lst

K = 10
bandit_list = k_armed_bandit(K)

trials = 2000
time_window = 500
for epsilon in [0,0.01,0.1]:
    average_reward_over_trials = [0 for _ in range(time_window)]
    for trial in range(1,trials+1):
        results = epsilon_greedy_run(k=K, bandit_list=bandit_list, T=time_window, epsilon=epsilon)
        # running_average = [sum(results[1][:window])/window for window in range(1,len(results[1])+1)]

        for index, reward_val in enumerate(results):
            average_reward_over_trials[index] += reward_val

        # plt.scatter(results[0], results[1])
    plt.scatter([t for t in range(1, time_window+1)], [val/trials for val in average_reward_over_trials])
        

plt.xlabel('Time Step')
plt.ylabel('Running Average of Reward over time')
plt.ylim(0,2.5)
plt.show()
