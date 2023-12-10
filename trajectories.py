import torch
import gym
import numpy as np

class Trajectories:
    def __init__(self, batch_size, env, policy_network, seed):
        self.batch_size = batch_size
        self.env = env
        self.policy_network = policy_network
        self.seed = seed
        torch.manual_seed(self.seed)
        
    def get_trajectory(self, seed):
        state, info = self.env.reset()
        state = np.concatenate(list(state.values()))
        
        states, actions, rewards, next_states = [], [], [], []
        
        done = False
        while not done:
            mean, std = self.policy_network(torch.tensor(state))
            action = torch.normal(mean, std)
            
            next_state, reward, terminated, trunc, info = self.env.step(action.detach().numpy())
            next_state = np.concatenate(list(next_state.values()))
            done = terminated or trunc
            
            states.append(state.tolist())
            actions.append(action.tolist())
            rewards.append(reward)
            next_states.append(next_state.tolist())
            
            state = next_state
        
        return states, actions, rewards, next_states
    
    def get_batch(self):
        states_batch, actions_batch, rewards_batch, next_states_batch = [], [], [], []
        
        for index in range(self.batch_size):
            states, actions, rewards, next_states = self.get_trajectory(seed=self.seed+index)
            states_batch.append(torch.tensor(states))
            actions_batch.append(torch.tensor(actions))
            rewards_batch.append(torch.tensor(rewards))
            next_states_batch.append(torch.tensor(next_states))

        return states_batch, actions_batch, rewards_batch, next_states_batch