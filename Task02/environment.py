import numpy as np
from discrete_space import Discrete

class MapWalkingEnv:

    def __init__(self, N, M):
        # Numpy array that holds the information about the environment
        self.territory = np.zeros((N, M), dtype = np.int8)
        self.height = N
        self.width = M
        self.size = N * M 
        self.average_size = round((N+M)/2,0)
        self.hostage = 1 # 1 hostage zones
        self.exit = 2 # 1 safe zone and 1 unsafe
        
        # Number of random objects
        self.number_keys = 1
        self.number_hostages = 1
        self.number_mines = 1
        self.exit_steps = None 

        # Define Walls (Static coordinates)
        wall_states = [(0,1),(1,5),(2,5),(3,0),(3,1),(3,3),(3,4),(3,6),(4,2),(4,4),(4,6),(4,7),(4,8),
                       (5,2),(5,4),(7,6),(7,7),(7,8),(7,9),(8,1),(8,2),(8,3),(8,4),(8,5),(9,7)]
        
        for i,j in wall_states:
            self.territory[i,j] = 1
        
        # Pick coordinates for the mine
        self.mine_coord = mine_coord = (0,9) # Static coordinates
        self.territory[mine_coord[0], mine_coord[1]] = 2

        # Pick 2 random coordinates to the safe zone and unsafe zone respectively
        #self.exit_coord = exit_coord = self.get_exit_cells(self.exit) # Dynamic coordinates
        self.exit_coord = exit_coord = ((0,8),(9,9))  # Static coordinates
        self.territory[exit_coord[0][0], exit_coord[0][1]] = 5 # Safe
        self.territory[exit_coord[1][0], exit_coord[1][1]] = 6 # Unsafe
        
        # Pick 1 random coordinate for the key
        #self.key_coord = key_coord = self.get_empty_cells(self.number_keys) # Dynamic coordinates
        self.key_coord = key_coord = (5,7) # Static coordinates
        self.territory[key_coord[0],key_coord[1]] = 3
         
        # Pick 1 random coordinate for the hostage zone
        #self.hostage_coord = hostage_coord = self.get_empty_cells(self.number_hostages) # Dynamic coordinates
        self.hostage_coord = hostage_coord = (9,8) # Static coordinates
        self.territory[hostage_coord[0],hostage_coord[1]] = 4
        
        # initial position of the agent will be decided by resetting the environment.
        self.position_agent = None
        self.start_agent = None

        # run time
        self.time_elapsed = 0
        self.time_limit = 2*self.size #* 8

        # display help
        self.dict_map_display={ 0:'.', #0 : empty cell
                                1:'X', #1 : walls obstacle that are non-traversable
                                2:'M', #2 : mine field location
                                3:'K', #3 : key location
                                4:'H', #3 : hostage zone    
                                5:'S', #4 : safe zone
                                6:'U', #6 : unsafe zone
                                7:'A' #7 : agent location
                              }
       

        num_states = self.size * 8
        num_rows = N
        num_columns = M
        max_row = num_rows - 1
        max_col = num_columns - 1
        self.initial_state_distrib = np.zeros(num_states)
        num_actions = 7
        
        # Set available actions
        self.actions = [0,1,2,3,4,5,6]
        #self.actions = ['up','down','left','right','pickup','dropoff','open']
        
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }
        self.states_array = {
            state   : {}
            for state in range(num_states)
        }
        self.observations_array = {
            state   : {}
            for state in range(num_states)
        }
        
        # Create matrix of possible transitions, essentially like R matrix
        for row in range(num_rows): #7
            for col in range(num_columns): #7
                for key_idx in range(2): # +1 for key being with agent
                    for host_idx in range(2): # +1 for host being with agent
                        for exit_idx in range(2): # for tracking if door was opened or not
                            state = self.encode(row, col, key_idx, host_idx, exit_idx)

                            if  (self.territory[row, col] != 1) \
                                and self.territory[row, col] != 2 \
                                and (key_idx < 1 and host_idx < 1 and exit_idx < 1):
                                self.initial_state_distrib[state] += 1
                                self.states_array[state] = (row, col)
                            if  self.territory[row, col] != 1: 
                                self.observations_array[state] = (row, col)
                            
                            valid_actions = self.valid_actions(row, col)
                            
                            for action in valid_actions:
                                # defaults
                                new_row, new_col, new_key_idx, new_host_idx, new_exit_idx=row, col, key_idx, host_idx, exit_idx
                                reward = (
                                    0 #-0.1
                                )  # Default reward when there is no pickup/dropoff/open and agent moves to an open cell (not a wall or a mine)
                                
                                done = False
                                agent_loc = (row, col)
                                

                                if action == 0: #'up' 
                                    new_row = max(row - 1, 0)
                                elif action == 1: #'down' 
                                    new_row = min(row + 1, max_row)
                                elif action == 2: #'right' 
                                    new_col = min(col + 1, max_col)
                                elif action == 3: #'left':
                                    new_col = max(col - 1, 0)
                                elif action == 4: #'pickup' 
                                    if new_host_idx == 0 and new_key_idx == 1 and new_exit_idx == 1 and \
                                    (agent_loc == (hostage_coord[0],hostage_coord[1])):
                                        new_host_idx = 1
                                        reward = 20
                                    elif new_key_idx == 0 and (agent_loc == (key_coord[0],key_coord[1])):
                                        new_key_idx = 1
                                        reward = 20
                                    elif self.territory[new_row, new_col] != 2: # key not at location or agent did not open a door yet
                                        reward = -20
                                elif action == 5: #'dropoff' 
                                    if (agent_loc == (exit_coord[0][0],exit_coord[0][1]) and \
                                        new_key_idx == 1 and new_host_idx == 1):
                                        done = True
                                        reward = 100
                                    elif agent_loc == (exit_coord[1][0],exit_coord[1][1]) : #(agent_loc == (exit_coord[0][0],exit_coord[0][1]) and (new_key_idx == 0)) or \
                                        done = True
                                        reward = -100
                                    elif self.territory[new_row, new_col] != 2:  # dropoff at wrong location
                                        reward = -20
                                elif action == 6: #'open' 
                                    if (agent_loc == (exit_coord[0][0],exit_coord[0][1])) and (new_key_idx == 1 and new_exit_idx == 0):
                                        new_exit_idx = 1
                                        #reward = 20
                                    elif self.territory[new_row, new_col] != 2:  # open at wrong location
                                        reward = -20

                                if (new_host_idx == 1 and new_key_idx == 1) or \
                                    new_exit_idx == 1:
                                    new_exit_idx=1
                                else:
                                    new_exit_idx=0
                                    
                                # If the agent bumps into a wall, it doesn't move
                                if self.territory[new_row, new_col] == 1:
                                    new_row = row
                                    new_col = col
                                    reward = -10
                                
                                # If the agent bumps into a mine it reaches a terminal state
                                if self.territory[new_row, new_col] == 2:
                                    new_row = row
                                    new_col = col
                                    done = True
                                    reward = -100 

                                if self.territory[row, col] != 1 and self.territory[row, col] != 2:
                                    new_state = self.encode(
                                        new_row, new_col, new_key_idx, new_host_idx, new_exit_idx
                                    )
                                    
                                    self.P[state][action].append((1.0, new_state, reward, done))
                       
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = Discrete(num_actions) #int(num_actions)
        self.observation_space = Discrete(num_states) # Encoded states shape
        self.observation_space_decoded = len(list(self.decode(0))) # Decoded states shape
        
    def get_empty_cells(self, n_cells):
        empty_cells_coord = np.where( self.territory == 0 )
        selected_indices = np.random.choice( np.arange(len(empty_cells_coord[0])), n_cells )
        selected_coordinates = empty_cells_coord[0][selected_indices], empty_cells_coord[1][selected_indices]
        
        if n_cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        
        return selected_coordinates
        
    def get_exit_cells(self, n_cells):
        y=np.arange(self.territory.shape[0] * self.territory.shape[1]).reshape(self.territory.shape[0], self.territory.shape[1])        
        border_coord_indices = list(np.unique(np.array(sorted(np.concatenate([y[0,:],y[-1,:],y[:,0], y[:,-1]]))).reshape(1,-1)))
        border_coord=np.zeros((2,len(border_coord_indices)),dtype = np.int8)
        border_coord[0] = [int(i) for i,ival in enumerate(y) for j,jval in enumerate(ival) if y[i,j] in border_coord_indices]
        border_coord[1] = [j for i,ival in enumerate(y) for j,jval in enumerate(ival) if y[i,j] in border_coord_indices]
        #print(border_coord_indices)
        #print(border_coord)
        selected_indices = np.random.choice( np.arange(len(border_coord[0])), n_cells )
        selected_coordinates = border_coord[0][selected_indices], border_coord[1][selected_indices]
        
        if n_cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        
        return selected_coordinates
    
    
    # a key function for the q learning update rule
    def step(self, action):
        transitions = self.P[self.s][action]
        i = self.categorical_sample([t[0] for t in transitions], np.random)
        p, state, reward, done = transitions[i]
        self.s = state
        self.lastaction = action
        
        if done == False:
            self.position_agent = (self.observations_array[self.s][0],self.observations_array[self.s][1])

        # Update time elapsed
        self.time_elapsed+=1
        
        _, _, _, _, self.exit_steps = self.decode(self.s)

        if self.time_elapsed == self.time_limit:
            done = True
        
        return (int(state), reward, done, {"prob": p})
   

    def agent_on_map(self):
        #prints out current location of agent on the grid
        grid = np.zeros((self.N, self.M))
        grid[ self.current_location[0], self.current_location[1]] = 1
        return grid
    
    
    def encode(self, agent_row, agent_col, key_idx, host_idx, exit_idx):
        i = agent_row
        i *= 10
        i += agent_col
        i *= 2
        i += key_idx
        i *= 2
        i += host_idx
        i *= 2
        i += exit_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % 10)
        i = i // 10
        out.append(i)
        assert 0 <= i < 10
        return reversed(out)
    
    def categorical_sample(self, prob_n, np_random):
        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)
        return (csprob_n > np_random.random()).argmax()
    
    def display(self, mode='human'):
 
        envir_with_agent = self.territory.copy()
        envir_with_agent[self.position_agent[0], self.position_agent[1]] = 7
        
        agent_row, agent_col, key_idx, host_idx, exit_idx = self.decode(self.s)

        full_repr = ""

        for r in range(self.height):
            
            line = ""
            
            for c in range(self.width):
                
                string_repr = self.dict_map_display[ envir_with_agent[r,c] ]
                line += "{0:2}".format(string_repr)

            full_repr += line + "\n"
        
        if key_idx == 1 and exit_idx == 0: # Agent found the key but did not open the door yet
            full_repr = full_repr.replace('A',colorize('A', "yellow", highlight=True))
        elif exit_idx == 1 and host_idx == 0: # Agent opened the door but did not find hostage
            full_repr = full_repr.replace('A',colorize('A', "blue", highlight=True))
        elif host_idx == 1: # Agent found the hostage
            full_repr = full_repr.replace('A',colorize('A', "green", highlight=True))

        print('Key (Yellow):',key_idx,'/ Door (Blue):',exit_idx,'/ Hostage (Green):',host_idx,'- Agent start',self.start_agent)
        print(full_repr)

        
    def calculate_observations(self):
        
        relative_coordinates = self.position_exit - self.position_agent
                
        surroundings = self.territory[ self.position_agent[0] -1: self.position_agent[0] +2,
                                     self.position_agent[1] -1: self.position_agent[1] +2]
        
        obs = {'relative_coordinates':relative_coordinates,
               'surroundings': surroundings}
        
        return obs
        
    def reset(self, return_info: bool = False):
        #This function resets the environment for the agent
        self.time_elapsed = 0
         
        invalid_position = True                                    


        while invalid_position == True:
            self.s = self.categorical_sample(self.initial_state_distrib, np.random)
            new_coord = self.states_array[self.s]
            #print(new_coord)
            if      ((new_coord[0],new_coord[1]) != (self.exit_coord[0][0],self.exit_coord[0][1])) \
                and ((new_coord[0],new_coord[1]) != (self.exit_coord[1][0],self.exit_coord[1][1])) \
                and ((new_coord[0],new_coord[1]) != (self.hostage_coord[0],self.hostage_coord[1])) \
                and ((new_coord[0],new_coord[1]) != (self.key_coord[0],self.key_coord[1])):
                invalid_position = False
                self.position_agent = (new_coord[0],new_coord[1])
        
        #self.position_agent = (6,9)
        self.start_agent = self.position_agent
        

        self.lastaction = None
        if not return_info:
            return int(self.s)
        else:
            return int(self.s), {"prob": 1}
        #return observations
        
        
    def valid_actions(self, row=None, col=None, cell=None):
        actions = [0, 1, 2, 3, 4, 5, 6]
        if (row == None and col == None):
            row = self.observations_array[self.s][0]
            col = self.observations_array[self.s][1]

        nrows, ncols = self.territory.shape
        if row == 0:
            actions.remove(0)
        elif row == nrows-1:
            actions.remove(1)

        if col == 0:
            actions.remove(3)
        elif col == ncols-1:
            actions.remove(2)
            
        return actions