import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNetwork(nn.Module):
    """Fully connected deep neural network."""
    
    # Network initialisation
    def __init__(self, state_size, fc1_dim, fc2_dim, fc3_dim, action_size, seed):
        """
        Args:
            state_size: observation/state size of the environment
            hidden_size: size of hidden layers
            action_size: number of discrete actions available in the environment
            
        """
        super(DQNetwork, self).__init__()
        
        self.input_dim = state_size
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.fc3_dim = fc3_dim
        self.output_dim = action_size
        #self.seed = torch.manual_seed(seed)
        
        # Embedding Layer (Disabled)
        #self.emb = nn.Embedding(self.input_dim, 4)
        
        # Fully connected layers and Batch Normalisation
        #self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
        #self.bn1 = nn.BatchNorm1d(self.fc1_dim)
        #self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        #self.bn2 = nn.BatchNorm1d(self.fc2_dim)
        #self.fc3 = nn.Linear(self.fc3_dim, self.fc3_dim)
        #self.out = nn.Linear(self.fc2_dim, self.output_dim)
        
                
        #self.loss = nn.MSELoss()
        #self.loss = nn.SmoothL1Loss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.q_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.fc1_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc1_dim, self.fc2_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc2_dim, self.output_dim)
        ).to(device)

        
    # Forward propagation 
    def forward(self, state):
        #state = state.type(torch.FloatTensor).to(self.device)
        #x = F.relu(self.fc1(self.emb(state)))
        
        actions = self.q_net(state)
        
        #x = F.relu(self.fc1(state))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #actions = self.out(x)
        
        return actions 


class ConvDQNetwork(nn.Module):
    """Convolutional + FCL deep neural network."""
    
    # Network initialisation
    def __init__(self, state_size, fc1_dim, fc2_dim, fc3_dim, action_size, seed):
        super(ConvDQNetwork, self).__init__()
        self.input_dim = (216,)
        self.output_dim = action_size
        #self.fc_input_dim = self.feature_size()
        
        self.features = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 25, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(25, 50, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, stride=2),
            nn.ReLU()
        ).to(device)

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        ).to(device)
        
        self.to(self.device)
    
    # Forward propagation 
    def forward(self, state):
        x = self.features(state)
        x = features.view(x.size(0), -1)
        qvals = self.fc(x)
        return qvals

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


class DuelingDQNetwork(nn.Module):
    """Dueling Deep Q network."""
    
    # Network initialisation
    def __init__(self, state_size, fc1_dim, fc2_dim, fc3_dim, action_size, seed):
        """
        Args:
            state_size: observation/state size of the environment
            fc1_dim, fc2_dim: size of hidden layers
            action_size: number of discrete actions available in the environment
            
        """
        super(DuelingDQNetwork, self).__init__()
        
        self.input_dim = state_size
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.output_dim = action_size
        #self.seed = torch.manual_seed(seed)
        
        
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        
        
        # Embedding Layer (Disabled)
        #self.emb = nn.Embedding(self.input_dim, 10)
        
        # Fully connected layer
        #self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
        
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim, self.fc1_dim),
            nn.ReLU()
        ).to(device)
        
        # ----------------------------- Dueling Network ------------------------------------- #
        # Value approximator - V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(self.fc1_dim, self.fc2_dim),
            nn.ReLU(),
            nn.Linear(self.fc2_dim, 1)
        ).to(device)
        
        # Advantage approximator - Q(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc1_dim, self.fc2_dim),
            nn.ReLU(),
            nn.Linear(self.fc2_dim, self.output_dim)
        ).to(device)
        # ----------------------------------------------------------------------------------- #   
 

        # Weight initialization (Xavier Uniform distribution)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu')) 
        
        #self.optimizer = optim.Adam(self.parameters(),lr=lr)
        #self.loss = nn.MSELoss()
        

    # Forward propagation 
    def forward(self, state):
        #state = torch.flatten(obs,start_dim=1)
        #state = state.type(torch.FloatTensor).to(self.device)
        #x = F.relu(self.fc1(self.emb(state)))
        
        #fc1 = F.relu(self.fc1(state))
        fc1 = self.fc1(state)
        
        values = self.value_stream(fc1)
        advantages = self.advantage_stream(fc1)
        advantages_average = torch.mean(advantages, dim=1, keepdim=True)
        #print('values',values)
        #print('advantages',advantages)
        #print('advantages_mean',advantages.mean()) #Deprecated
        #print('advantages_average',advantages_average)
        Q_values = values + (advantages - advantages_average)

        return Q_values

    # Select best action based on output of Q_values
    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state) # forward pass
            action_idx = torch.argmax(Q, dim=1) # select best action
        return action_idx.item()

      
class ICM(nn.Module):
    """Intrinsic curiosity network."""
    
    # Network initialisation
    def __init__(self, state_size, fc1_dim, fc2_dim, action_size, seed, alpha=1, beta=0.2):
        """
        Args:
            state_size: observation/state size of the environment
            fc1_dim, fc2_dim: size of hidden layers
            action_size: number of discrete actions available in the environment
            
        """
        super(ICM, self).__init__()

        self.input_dim = state_size
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.output_dim = action_size
        self.seed = torch.manual_seed(seed)
        
        self.alpha = alpha
        self.beta = beta
        
        self.inverse = nn.Linear(4*2, 256)
        self.pi_logits = nn.Linear(256, self.output_dim)

        self.fc1 = nn.Linear(4 + 1,256)
        self.new_state = nn.Linear(256, 4)
        
        self.to.device(device)
        
    # Forward propagation 
    def forward(self, state, new_state, action):
        state = state.type(torch.FloatTensor).to(self.device)
        inverse = F.elu(self.inverse(torch.cat([state,new_state], dim=1)))
        
        action = action.reshape((action.size()[0], 1))
        forward_input = torch.cat([state, action], dim=1)
        dense = F.elu(self.fc1(forward_input))
        state_ = self.new_state(dense)
        
        return pi_logits, state_

    # Calculate loss 
    def calc_loss(self, state, new_state, action):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        
        pi_logits, state_ = self.forward(state, new_state, action)
        
        inverse_loss = nn.CrossEntropyLoss()
        L_I = (1 - self.beta) * inverse_loss(pi_logits, action.to(torch.Long)) 
        
        forward_loss = nn.MSELoss()
        L_F = self.beta * forward_loss(state_, new_state)
        
        intrinsic_reward = self.alpha * ((state_, new_state).pow(2)).mean(dim=1) # Get mean accross the first dimension (number of states)
        
        return intrinsic_reward, L_I, L_F