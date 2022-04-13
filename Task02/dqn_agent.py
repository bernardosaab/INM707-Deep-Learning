import numpy as np
import random
from collections import namedtuple, deque

from model import DQNetwork, ConvDQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

REPLAY_MEMORY_SIZE = 10000  # replay buffer size
MIN_REPLAY_MEMORY_SIZE = 1000
BATCH_SIZE = 64         # minibatch size
#GAMMA = 0.99            # discount factor
TAU = 10
#1e-3                    # for soft update of target parameters
#LR = 5e-4               # learning rate 
UPDATE_TARGET_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, fc1_dim, fc2_dim, fc3_dim, action_size, seed, learning_rate, gamma, with_conv=True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        
        # self.state_size = state_size
        # self.fc1_dim = fc1_dim
        # self.fc2_dim = fc2_dim
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.with_conv = with_conv
        
        # Q-Network (With or without convolution layers)
        if self.with_conv:
            # Local Network
            self.qnetwork_local = ConvDQNetwork(state_size, fc1_dim, fc2_dim, fc3_dim, action_size, seed).to(device)
            # Target Network
            self.qnetwork_target = ConvDQNetwork(state_size, fc1_dim, fc2_dim, fc3_dim ,action_size, seed).to(device)
        else:
            # Local Network
            self.qnetwork_local = DQNetwork(state_size, fc1_dim, fc2_dim, fc3_dim, action_size, seed).to(device)
            # Target Network
            self.qnetwork_target = DQNetwork(state_size, fc1_dim, fc2_dim, fc3_dim ,action_size, seed).to(device)

        
        # Adam optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Replay memory (Create a new instance of Experience Replay Buffer)
        self.memory = ExperienceBuffer(action_size, REPLAY_MEMORY_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.mem_count = 0

        # Initialize time step (for updating the Target Network)
        self.qtarget_update_count = 0

    def step(self, state, action, reward, next_state, terminal):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, terminal)       
        
        # Learn every UPDATE_EVERY time steps.
        #self.mem_count = (self.mem_count + 1) % UPDATE_EVERY
        self.mem_count = (self.mem_count + 1)
        #if self.mem_count == 0:
        

        if len(self.memory) < MIN_REPLAY_MEMORY_SIZE:
            return
    
        # If enough samples are available in memory, get random subset and learn	
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def select_action(self, observation, valid_actions, epsilon=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            epsilon (float): epsilon, for epsilon-greedy action selection
        """
        #state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        #self.qnetwork_local.eval()
        #with torch.no_grad():
        #    action_values = self.qnetwork_local(state)
        

        # Epsilon-greedy action selection
        if np.random.random() > epsilon:
            state = torch.from_numpy(np.array(observation)).float().unsqueeze(0).to(device)
            #self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)               
                action_values_valid = torch.tensor([-np.inf,-np.inf,-np.inf,-np.inf-np.inf,-np.inf,-np.inf,-np.inf]).to(device)
                for action,value in enumerate(np.asarray(action_values.view(-1,1).squeeze(1).to(device).cpu())):
                    if action in valid_actions:
                        action_values_valid[action] = torch.from_numpy(np.array(value)).to(device)
                #action_values = torch.from_numpy(np.array([action_values[0][index] for index in valid_actions])).float().unsqueeze(0).to(device)
                action = np.argmax(action_values_valid.cpu().data.numpy())
                #self.qnetwork_local.train()
        else:
            action_values = [action for action in valid_actions]
            #print(action_values)
            action = np.random.choice(action_values)
            #action = np.random.choice(np.arange(self.action_size))
        
        return action

    def update_target_network(self):
        # Update parameters to frozen network (Target Network) from Local Network
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def learn(self, experiences, gamma, ddqn=False):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', terminal) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, terminals = experiences
        
        #print('1',states.size())
        #print('2',states)
        #print(actions)
        # Get expected Q values from local network
        #Q_expected = self.qnetwork_local(states).gather(0, actions.unsqueeze(1)).squeeze(1)
        Q_expected = self.qnetwork_local(states).gather(0, actions)
  
        # Get max predicted Q values (for next states) from target network
        Q_targets_next = self.qnetwork_target(next_states).detach().squeeze(1).max(1)[0].unsqueeze(1).float()

        #Q_targets_next = torch.max(self.qnetwork_target(next_states), dim=1).values.view(-1, 1).detach()
        #print('2',Q_targets_next.size())
        
        # Compute Q targets for current states 
        terminals_mask = (1 - terminals).float()
        Q_targets = rewards + terminals_mask * gamma * Q_targets_next
        
        #print('1',rewards.size(),gamma,Q_targets_next.size(),terminals.size(),self.qnetwork_target(next_states))
        #print('3',Q_targets)
    
        #print('2',Q_expected.size(),Q_targets.size())

        # Compute Loss function
        # MSE Loss
        loss = F.huber_loss(Q_expected, Q_targets)
        # MSE Loss
        #loss = F.mse_loss(Q_expected, Q_targets.squeeze())
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        # ----------- Gradient clipping and flipping -------------- #
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
        # --------------------------------------------------------- #
        
        self.optimizer.step()

        # ------------------- Update target network --------------- #

        # ------------------- Soft Update ------------------------- #
        #self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     
        # --------------------------------------------------------- #

        # ------------------- Hard Update ------------------------- #
         
#         if terminals:
#             self.qtarget_update_count += 1

#         if self.qtarget_update_count > UPDATE_TARGET_EVERY:
#             self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
#             self.qtarget_update_count = 0
        # --------------------------------------------------------- #
    

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ExperienceBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, memory_size, batch_size, seed):
        """Initialize a ExperienceBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            memory_size (int): maximum size of memory buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "terminal"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, terminal):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, terminal)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)


        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        terminals = torch.from_numpy(np.vstack([e.terminal for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, terminals)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)