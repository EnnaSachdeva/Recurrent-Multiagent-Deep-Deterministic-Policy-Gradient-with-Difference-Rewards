"""

# Pseudocode for history buffer for RNN

history_buffer = []
obs = env.reset()
obs_tminus_0 = obs
obs_tminus_1 = obs
obs_tminus_2 = obs

for t in range(timesteps):               # main.py loop
    
    obs_history = obs_tminus_0        # Just to give obs_history the same list shape.
    
    # Make obs_history to feed into the policy
    for a in range(num_agents):          # for each agent
        for n in range(3):               # for previous time steps
            obs_history[a][n] = obs_tminus_0[a][n] + obs_tminus_1[a][n] + obs_tminus_2[a][n]

    action = policy(obs_history)
    next_obs = step(action)
    
    # Make next_obs_history to feed into replay buffer
    next_obs_history = obs_history       # Just to give the same shape
    for a in range(num_agents):          # for each agent
        for n in range(3):               # for previous time steps
            next_obs_history[a][n] = next_obs[a][n] + obs_tminus_0[a][n] + obs_tminus_1[a][n]
    
    
    replay_buffer.push(obs, next_obs_history)
    
    obs_tminus_2 = copy.copy(obs_tminus_1)
    obs_tminus_1 = copy.copy(obs_tminus_0)
    obs_tminus_0 = copy.copy(next_obs)
    
    obs = next_obs
    


"""



"""
At t = T
# Agent 0's observations = obs[0][0][0]
# Agent 1's observations = obs[0][0][1]
# Agent 2's observations = obs[0][0][2]

At t = T-1
# Agent 0's observations = obs[0][1][0]
# Agent 1's observations = obs[0][1][1]
# Agent 2's observations = obs[0][1][2]

At t = T-2
# Agent 0's observations = obs[0][2][0]
# Agent 1's observations = obs[0][2][1]
# Agent 2's observations = obs[0][2][2]

"""
