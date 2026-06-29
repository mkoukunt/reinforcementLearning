import numpy as np

# 1. Setup Environment parameters
states = [0, 1, 2]  # 0: s1, 1: s2, 2: s3 (Terminal)
actions = ['Left', 'Right']
gamma = 0.9
epsilon = 1e-6  # Convergence threshold

# Initialize Value Function Table
V = np.zeros(len(states))


# Define Transition Dynamics & Rewards: P(s_next | s, a) and R(s, a, s_next)
def get_dynamics(s, a):
    # Returns a list of tuples: (probability, next_state, reward)
    if s == 2:  # Terminal state has no transitions
        return [(1.0, 2, 0)]

    transitions = []
    if s == 0:
        if a == 'Right':
            transitions.append((0.8, 1, -1))  # Success
            transitions.append((0.2, 0, -1))  # Slip (bounce)
        elif a == 'Left':
            transitions.append((1.0, 0, -1))  # Bounce off wall

    elif s == 1:
        if a == 'Right':
            transitions.append((0.8, 2, 10))  # Success (Goal!)
            transitions.append((0.2, 0, -1))  # Slip backward
        elif a == 'Left':
            transitions.append((0.8, 0, -1))  # Success moving left
            transitions.append((0.2, 2, 10))  # Slip forward to Goal

    return transitions


# 2. Value Iteration (Dynamic Programming)
iteration = 0
while True:
    delta = 0
    V_old = V.copy()

    for s in states:
        if s == 2:
            continue  # Skip terminal state

        action_values = []
        for a in actions:
            q_value = 0
            for prob, next_s, reward in get_dynamics(s, a):
                q_value += prob * (reward + gamma * V_old[next_s])
            action_values.append(q_value)

        V[s] = max(action_values)
        delta = max(delta, abs(V[s] - V_old[s]))

    iteration += 1
    if delta < epsilon:
        break

# 3. Policy Extraction
optimal_policy = {}
for s in states:
    if s == 2:
        optimal_policy[f"s{s + 1}"] = "Goal"
        continue

    best_action = None
    best_value = float('-inf')
    for a in actions:
        q_value = 0
        for prob, next_s, reward in get_dynamics(s, a):
            q_value += prob * (reward + gamma * V[next_s])
        if q_value > best_value:
            best_value = q_value
            best_action = a
    optimal_policy[f"s{s + 1}"] = best_action

# Output Results
print(f"Converged in {iteration} iterations.")
print(f"Optimal Values: State 1 = {V[0]:.2f}, State 2 = {V[1]:.2f}, State 3 = {V[2]:.2f}")
print("Optimal Policy:", optimal_policy)
