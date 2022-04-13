import numpy as np
import random
from collections import namedtuple, deque

from dqn_model import DeepQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

MEMORY_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
#GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
#LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, fc1_dim, fc2_dim, action_size, seed, lr, gamma):
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
        
        # Q-Network
        # Local Network
        self.q_network = DeepQNetwork(state_size, fc1_dim, fc2_dim, action_size, seed, lr=lr).to(device)
        #.to(device)
        # Target Network
        self.q_network_target = DeepQNetwork(state_size, fc1_dim, fc2_dim, action_size, seed, lr=lr).to(device)
        #.to(device)
        # Adam optimizer
        self.optimizer = optim.Adam(self.q_network.parameters())

        # Replay memory (Create a new instance of Experience Replay Buffer)
        self.memory = ExperienceBuffer(action_size, MEMORY_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.mem_count = 0

	
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)       
        
        # Learn every UPDATE_EVERY time steps.
        #self.mem_count = (self.mem_count + 1) % UPDATE_EVERY
        self.mem_count = (self.mem_count + 1)
        #if self.mem_count == 0:
        
        # If enough samples are available in memory, get random subset and learn
        
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def select_action(self, observation, epsilon=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            epsilon (float): epsilon, for epsilon-greedy action selection
        """
        #state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        #self.q_network.eval()
        #with torch.no_grad():
        #    action_values = self.q_network(state)
        #self.q_network.train()

        # Epsilon-greedy action selection
        if np.random.random() > epsilon:
            state = torch.from_numpy(np.array(observation)).float().unsqueeze(0).to(device)
            #self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
                action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = np.random.choice(np.arange(self.action_size))
        
        return action
        
    def learn(self, experiences, gamma, ddqn=False):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get expected Q values from local model (Policy network)
        Q_values = self.q_network(states).gather(1, actions)
        #print('Q_values',Q_values.shape)
        
        # Get selected actions from local network
        selected_actions = self.q_network(next_states).max(1)[1].unsqueeze(1).detach()
        #print('Selected',selected_actions.shape)

        # Get max predicted Q values (for next states) from target model
        #Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # use current network to evaluate action argmax_a' Q_current(s', a')_
        Q_targets_next = self.q_network_target(next_states).gather(1, selected_actions).view(-1,).unsqueeze(1).detach()
        #print('Q_targets_next',Q_targets_next.shape)
        
        # use target network to evaluate value y = r + discount_factor * Q_tar(s', a')
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        

        # Compute loss (MSE Loss)
        loss = F.mse_loss(Q_values, Q_targets)
        
        # Optimize and minimize the loss (Backpropagation)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Update weights and biases
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.q_network, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        q_network_theta = self.q_network.get_weights()
        target_network_theta = self.q_network_target.get_weights()
        
        for target_weight, local_weight in zip(q_network_theta, local_model.parameters()):
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)


        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)