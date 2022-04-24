import numpy as np
import random
from random import shuffle
from collections import namedtuple, deque
import copy

from model import DQNetwork, ConvDQNetwork, DuelingDQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

REPLAY_MEMORY_SIZE = 1000  # replay memory size
MIN_REPLAY_MEMORY_SIZE = 1000
BATCH_SIZE = 200            # minibatch size (we set batch as the number of max timesteps)
#GAMMA = 0.99               # discount factor
TAU = 10 
#1e-3                       # for soft update of target parameters
#LR = 5e-4                  # learning rate 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, fc1_dim, fc2_dim, fc3_dim, action_size, seed, learning_rate, gamma, 
                 loss_fun='mse', conv=True, ddqn=False, dueling=False,per=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        
        self.state_size = state_size
        # self.fc1_dim = fc1_dim
        # self.fc2_dim = fc2_dim
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.loss_fun = loss_fun
        self.with_conv = conv
        self.with_dueling = dueling
        self.with_ddqn = ddqn
        self.with_per = per
        
        self.losses = []
        self.qvalues_estimates = []
        #self.batch_size = BATCH_SIZE
    
        self.per_alpha = 0.6
        self.per_beta = 0.4
        
        # Q-Network (With or without convolution layers)
        if self.with_conv:
            # Local Network
            self.q_network = ConvDQNetwork(state_size, fc1_dim, fc2_dim, fc3_dim, action_size, seed).to(device)
            # Target Network
            self.q_network_target = ConvDQNetwork(state_size, fc1_dim, fc2_dim, fc3_dim ,action_size, seed).to(device)
        else:
            if self.with_dueling:
                # Local Network
                self.q_network = DuelingDQNetwork(state_size, fc1_dim, fc2_dim, fc3_dim, action_size, seed)
                # Target Network
                self.q_network_target = DuelingDQNetwork(state_size, fc1_dim, fc2_dim, fc3_dim ,action_size, seed)
            else:
                # Local Network
                self.q_network = DQNetwork(state_size, fc1_dim, fc2_dim, fc3_dim, action_size, seed)
                # Target Network
                self.q_network_target = DQNetwork(state_size, fc1_dim, fc2_dim, fc3_dim ,action_size, seed)
                

        # Copy network parameters for initialised instance of Target Q Network 
        self.q_network_target = copy.deepcopy(self.q_network) 
        self.q_network_target.load_state_dict(self.q_network.state_dict())  
        
        # Adam optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        

        # Replay memory (Create a new instance of Experience Replay Buffer)
        if self.with_per: # Prioritized Experience Replay
            self.memory = PrioritizedExperienceReplay(state_size, action_size, REPLAY_MEMORY_SIZE)
        else:             # Experience Replay
            self.memory = ExperienceReplay(action_size, REPLAY_MEMORY_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every number of steps)
        self.mem_count = 0

        # Initialize time step (for updating the Target Network)
        self.qtarget_update_count = 0

    def step(self, state, action, reward, next_state, terminal):
        # Save experience in replay memory
        #print('Adding',state,next_state,action)
        self.memory.add(state, action, reward, next_state, terminal)       
        
        # Learn every update_steps time steps.
        #self.mem_count = (self.mem_count + 1) % update_steps
        self.mem_count = (self.mem_count + 1)

        # Warmp up phase
        #if len(self.memory) < MIN_REPLAY_MEMORY_SIZE:
        #    return
    
        # If enough samples are available in memory, get random subset and learn	
        if len(self.memory) > BATCH_SIZE:
            if isinstance(self.memory, ExperienceReplay):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
            elif isinstance(self.memory, PrioritizedExperienceReplay):
                experiences, weights, tree_idx = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, self.gamma, weights)
                
    def select_action(self, state, valid_actions, epsilon=0.):
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
        
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            #action_values = [action for action in valid_actions]
            #action = np.random.choice(action_values)
            
            #action = np.random.choice(np.arange(self.action_size))
            action = np.random.randint(0,self.action_size)
        else:
            if self.with_dueling:
                action = self.q_network.select_action(state)
            else:
                #state = torch.from_numpy(np.array(observation)).float().unsqueeze(0).to(device)
                #self.q_network.eval() # Only if the objective is to avoid issues with batch normalisation due to have 1 training sample 
                with torch.no_grad():
                    action_values = self.q_network(state)               
                    action = np.argmax(action_values.data.cpu().numpy())
                #self.q_network.train() # Resume training mode
            
        return action
    
    def greedy_action(self, state, valid_actions):
        """Returns actions for given state as per greedy policy.
        
        Params
        ======
            state (array_like): current state
            epsilon (float): epsilon, for epsilon-greedy action selection
        """

        if self.with_dueling:
            action = self.q_network.select_action(state)
        else:
            #state = torch.from_numpy(np.array(observation)).float().unsqueeze(0).to(device)
            #self.q_network.eval() # Only if the objective is to avoid issues with batch normalisation due to have 1 training sample 
            with torch.no_grad():
                action_values = self.q_network(state)               
                action = np.argmax(action_values.data.cpu().numpy())
            #self.q_network.train() # Resume training mode
            
        return action
        
    def update_target_network(self):
        # Update parameters to frozen network (Target Network) from Local Network
        self.q_network_target.load_state_dict(self.q_network.state_dict())

    def learn(self, experiences, gamma, weights=None):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', terminal) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, terminals = experiences

        # Get expected Q values from local network
        #Q_expected = self.q_network(states).gather(0, actions.unsqueeze(1)).squeeze(1)

        #Q_expected = self.q_network(states).gather(0, actions.unsqueeze(1))
        
        Q_expected = self.q_network(states).gather(dim=1,index=actions.long().unsqueeze(dim=1)).squeeze()

        
        if weights is None:
            weights = torch.ones_like(Q_expected).to(device)
        


        if self.with_ddqn: #-----------------Double DQN-----------------#
            # We decouple the action selection from the target Q-value generation and reduce the overestimation bias effect
            # We use DQN to select the best action of the next action (1) then evaluate the same action using the target DQN (2)
            
            # 2nd check
            #next_q_values = self.q_network(next_states) # Or torch.max(self.q_network(next_states), 1)[1].unsqueeze(1)
            #next_q_state_values = self.q_network_target(next_states) 
            #next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
            #expected_q_value = rewards + gamma * next_q_value * (1 - terminals)
            ###########
            
            # (1) Get selected action from local network for  
            selected_action = self.q_network(next_states).detach().max(1)[1].unsqueeze(1)
            
            
            # (2) Use target network to evaluate action argmax_a' from Q_current(s', a')
            Q_targets_next = self.q_network_target(next_states).gather(1, selected_action).view(-1,).unsqueeze(1)
            
        else:              #-----------------Vanilla DQN-----------------#
            # Get max predicted Q values (for next states) from target network (max Q values for (s',a'))
            #Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1).float()
            with torch.no_grad():
                Q_targets_next = self.q_network_target(next_states)

        # Compute Q targets for states 
        terminals_mask = (1 - terminals)  # if current state is in the end of episode there is no next Q values
        Q_targets = rewards + gamma * (terminals_mask * torch.max(Q_targets_next,dim=1)[0])

        #print(selected_action,Q_targets,expected_q_value)
        
        
        # Compute Loss function
        if self.loss_fun == 'huber': # Huber Loss function
            loss = F.huber_loss(Q_expected, Q_targets.detach()) * weights
        elif self.loss_fun == 'mse': # MSE Loss function
            if self.with_per: # Loss function if Prioritized Experience Replay required
                loss = self.mse_loss_weighted(Q_expected, Q_targets.detach(), weights)
            else: # Loss function if Prioritized Experience Replay not required
                loss = F.mse_loss(Q_expected, Q_targets.detach())

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        # --------- Save Q-values estimates of the policy network for every training step ----------- #
        self.losses.append(loss.item())
        # ------------------------------------------------------------------------------------------- #
        
        
        # ----------- Gradient clipping and no flipping ----------- #
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        # --------------------------------------------------------- #
        
        self.optimizer.step()
        
        
        # --------- Save Q-values estimates of the policy network for every training step ----------- #
        n = 4 # 4 classes
        one_hot = torch.nn.functional.one_hot(actions.to(torch.int64), n)
        
        self.qvalues_estimates.append(torch.sum(self.q_network(states).detach()*one_hot).cpu().numpy())
        # ------------------------------------------------------------------------------------------- #
        
        # ------------------- Update target network --------------- #

        # ------------------- Soft Update ------------------------- #
        #self.soft_update(self.q_network, self.q_network_target, TAU)                     
        # --------------------------------------------------------- #

        # ------------------- Hard Update ------------------------- #
         
#         if terminals:
#             self.qtarget_update_count += 1

#         if self.qtarget_update_count > UPDATE_TARGET_EVERY:
#             self.q_network_target.load_state_dict(self.q_network.state_dict())
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

    def mse_loss_weighted(self, input, output, weight): 
        #print(input.get_device())
        #print(output.get_device())
        #print(weight.get_device())
        return torch.sum(weight * (input - output) ** 2)

class ExperienceReplay:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, memory_size, batch_size, seed):
        """Initialize a ExperienceReplay object.
        Params
        ======
            action_size (int): dimension of each action
            memory_size (int): maximum size of memory buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)  
        #self.memory = []
        self.memory_size = memory_size
        self.batch_size = batch_size
        #self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "terminal"])
        self.seed = random.seed(seed)
        self.counter = 0
        
    def shuffle_memory(self):
        shuffle(self.memory)
    
    def add(self, state, action, reward, next_state, terminal):
        """Add a new experience to memory."""
        #self.counter = self.counter + 1
        
        #if self.counter % 500 == 0:
        #    self.shuffle_memory()
        
        if len(self.memory) <= self.memory_size:
            experience = (state, action, reward, next_state, terminal)
            self.memory.append(experience)
        else:
            rand_index = np.random.randint(0, self.memory_size - 1)
            self.memory[rand_index] = (state, action, reward, next_state, terminal)
            print(rand_index, self.memory_size)
            
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        #experiences = random.sample(self.memory, k=self.batch_size)

        if len(self.memory) < self.batch_size:
            batch_size = len(self.memory)
        else:
            batch_size = self.batch_size
    
        if len(self.memory) < 1:
            print("Error: No data in memory.")
            return None
        
        minibatch = random.sample(self.memory, batch_size)
        state_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]).to(device)
        action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(device)
        reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(device)
        next_state_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch]).to(device)
        terminal_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(device)

        #ind = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)
        
        
        #batch = [self.memory[i] for i in ind] # sample batch as a list of tuples
        
        #print('1',[x[0] for x in batch])
        #print('2',[x[3] for x in batch])
        

        #state_batch = torch.Tensor(np.stack([x[0] for x in batch], axis=0)).squeeze(1).float().to(device)
        #action_batch = torch.Tensor(np.stack([x[1] for x in batch], axis=0)).long().to(device)
        #reward_batch = torch.Tensor(np.stack([x[2] for x in batch], axis=0)).unsqueeze(1).float().to(device)
        #next_state_batch = torch.Tensor(np.stack([x[3] for x in batch], axis=0)).squeeze(1).float().to(device)
        #terminal_batch = torch.Tensor(np.stack([x[4] for x in batch], axis=0)).unsqueeze(1).float().to(device)

        
        #states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        #actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        #rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        #next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        #terminals = torch.from_numpy(np.vstack([e.terminal for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
 
        
        return (state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedExperienceReplay:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, state_size, action_size, memory_size, eps=1e-2, alpha=0.6, beta=0.4):
        self.tree = SumTree(memory_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, next_state, terminal
        self.state = torch.empty(memory_size, state_size, dtype=torch.float)
        self.action = torch.empty(memory_size, 1, dtype=torch.long)
        self.reward = torch.empty(memory_size, dtype=torch.float)
        self.next_state = torch.empty(memory_size, state_size, dtype=torch.float)
        self.terminal = torch.empty(memory_size, dtype=torch.float)

        self.count = 0
        self.real_size = 0
        #self.batch_size = batch_size
        self.size = memory_size

    def add(self, state, action, reward, next_state, terminal):
        
        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.terminal[self.count] = torch.as_tensor(terminal)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "check if buffer contains more samples than batch size and proceed if positive"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)
        
              

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total()
        

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs].to(device),
            self.action[sample_idxs].squeeze(1).to(device),
            self.reward[sample_idxs].unsqueeze(1).to(device),
            self.next_state[sample_idxs].to(device),
            self.terminal[sample_idxs].unsqueeze(1).to(device)
        )
        return batch, torch.as_tensor(weights).to(device), tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.real_size  
    
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
