import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd



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
        self.seed = torch.manual_seed(seed)
    
        #self.emb = nn.Embedding(self.input_dim, 4)
        self.fc1 = nn.Linear(1, self.fc1_dim)   
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc3_dim, self.fc3_dim)
        self.out = nn.Linear(self.fc3_dim, self.output_dim)
        

        #self.optimizer = optim.Adam(self.parameters(),lr=lr)
        #self.loss = nn.MSELoss()
        #self.loss = nn.SmoothL1Loss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        
    # Forward propagation 
    def forward(self, state):
        #state = torch.flatten(obs,start_dim=1)
        #torch.tensor(state).to(torch.int64)
        state = state.type(torch.FloatTensor).to(self.device)
        #x = F.relu(self.fc1(self.emb(state)))
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.out(x)
        return actions 


class ConvDQNetwork(nn.Module):
    """Convolutional + FCL deep neural network."""
    
    # Network initialisation
    def __init__(self, state_size, fc1_dim, fc2_dim, fc3_dim, action_size, seed):
        super(ConvDQNetwork, self).__init__()
        self.input_dim = state_size
        self.output_dim = action_size
        #self.fc_input_dim = self.feature_size()
        
        self.features = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            #nn.Linear(self.fc_input_dim, 128),
            nn.Linear(self.feature_size(), 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )
        
        self.to(self.device)
        
    def forward(self, state):
        x = self.features(state)
        x = features.view(x.size(0), -1)
        qvals = self.fc(x)
        return qvals

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


class DuelingDeepQNetwork(nn.Module):
    """Simple MLP network."""
    
    # Network initialisation
    def __init__(self, state_size, fc1_dim, fc2_dim, action_size, seed, lr):
        """
        Args:
            state_size: observation/state size of the environment
            hidden_size: size of hidden layers
            action_size: number of discrete actions available in the environment
            
        """
        super(DeepQNetwork, self).__init__()
        
        self.input_dim = state_size
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.output_dim = action_size
        self.seed = torch.manual_seed(seed)

        #self.emb = nn.Embedding(self.input_dim, 10)
        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)   
        self.fc_value = nn.Linear(self.fc1_dim, self.fc2_dim)   
        self.fc_advantage = nn.Linear(self.fc1_dim, self.fc2_dim)

        self.value = nn.Linear(self.fc2_dim, 1)   
        self.advantage = nn.Linear(self.fc2_dim, output_dim)


        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    # Forward propagation 
    def forward(self, state):
        #state = torch.flatten(obs,start_dim=1)
        #print(obs.shape, state.shape)
        x = F.relu(self.fc1(state))

        advantage = self.fc_advantage(h1)
        value = self.fc_value(h1).expand(-1, ACTION_SIZE)
        advantage_average = torch.mean(advantage, dim=1, keepdim=True)
        Q_output = value + advantage - advantage_average

        return Q_output