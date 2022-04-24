import numpy as np
from discrete_space import Discrete

#These are the key actions you can apply at states
action_dict = {
    0: 'up',
    1: 'down',
    2: 'right',
    3: 'left',
}

#Colours for environment
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)

#Function to surround string with certain colour codes
def colorise(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    attrs = ";".join(attr)
    return f"\x1b[{attrs}m{string}\x1b[0m"

# A function that counts the number of true values when the cumulative sum of n probabilities
#In an array is greater than a randomly generated number used in the step function



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
        wall_states = [(0,1),(1,5),(2,5),(2,8),(3,0),(3,1),(3,3),(3,4),(3,6),(3,8),(4,2),(4,4),(4,6),
                       (4,7),(4,8),(5,2),(5,4),(7,6),(7,7),(7,8),(7,9),(8,1),(8,2),(8,3),(8,4),(8,5),(9,7)]
        #(2,8),(3,8)
        #wall_states = [(0,1),(1,5),(2,5),(3,0),(3,1),(3,3),(4,4),(5,4)]
        
        for i,j in wall_states:
            self.territory[i,j] = 1
        
        # Pick coordinates for the mine
        #((0,2),(1,8),(5,2),(5,8),(8,9)) 
        self.mine_coord = mine_coord = ((0,2),(1,8),(5,2),(5,8),(8,9)) # Static coordinates
        self.territory[mine_coord[0][0], mine_coord[0][1]] = 2
        self.territory[mine_coord[1][0], mine_coord[1][1]] = 2
        self.territory[mine_coord[2][0], mine_coord[2][1]] = 2
        self.territory[mine_coord[3][0], mine_coord[3][1]] = 2
        self.territory[mine_coord[4][0], mine_coord[4][1]] = 2
    
        # Pick 2 random coordinates to the safe zone and unsafe zone respectively
        #self.exit_coord = exit_coord = self.get_exit_cells(self.exit) # Dynamic coordinates
        self.exit_coord = exit_coord = (0,0)  # Static coordinates
        self.territory[exit_coord[0], exit_coord[1]] = 5 # Safe
        #self.territory[exit_coord[1][0], exit_coord[1][1]] = 6 # Unsafe
        
        # Pick 1 random coordinate for the key
        #self.key_coord = key_coord = self.get_empty_cells(self.number_keys) # Dynamic coordinates
        #self.key_coord = key_coord = (4,1) # Static coordinates
        #self.territory[key_coord[0],key_coord[1]] = 3
         
        # Pick 1 random coordinate for the hostage zone
        #self.hostage_coord = hostage_coord = self.get_empty_cells(self.number_hostages) # Dynamic coordinates
        #self.hostage_coord = hostage_coord = (2,1) # Static coordinates
        #self.territory[hostage_coord[0],hostage_coord[1]] = 4
        
        self.displ_board = np.zeros((4, self.height, self.width), dtype=np.uint8)
        self.add_encoded_states()
        
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
                                #3:'K', #3 : key location
                                #4:'H', #3 : hostage zone    
                                5:'S', #4 : safe zone
                                #6:'U', #6 : unsafe zone
                                7:'A' #7 : agent location
                              }
       

        num_states = self.size * 8
        num_rows = N
        num_columns = M
        max_row = num_rows - 1
        max_col = num_columns - 1
        self.initial_state_distrib = np.zeros(num_states)
        num_actions = 4
        
        # Set available actions
        self.actions = [0,1,2,3]
        #self.actions = ['up','down','left','right']
        
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(self.size)
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
                #for key_idx in range(2): # for tracking if door was opened or not                
                #    for host_idx in range(2): # for tracking if door was opened or not
                        #for exit_idx in range(2): # for tracking if door was opened or not
                            #state = self.encode(row, col, key_idx, host_idx, exit_idx)
                            state = self.encode(row, col)
                            
                            if  (self.territory[row, col] != 1) \
                                and self.territory[row, col] != 2:
                                self.initial_state_distrib[state] += 1
                                self.states_array[state] = (row, col)
                            if  self.territory[row, col] != 1: 
                                self.observations_array[state] = (row, col)
                            
                            valid_actions = [0,1,2,3]
                            
                            #valid_actions = self.valid_actions(row, col)
                            
                            for action in valid_actions:
                                # defaults
                                #new_row, new_col, new_key_idx, new_host_idx, new_exit_idx=row, col, key_idx, host_idx, exit_idx
                                new_row, new_col = row, col
                                
                                reward = (
                                    -1 #-0.1
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
                                
                                # Invalid actions
                                if action == 0 and row == 0:
                                    reward = -2
                                elif action == 1 and row == max_row:
                                    reward = -2
                                elif action == 2 and col == max_col:
                                    reward = -2
                                if action == 3 and col == 0:
                                    reward = -2
                                
                                
                                #if new_key_idx == 0 and ((new_row, new_col) == (self.key_coord[0],self.key_coord[1])):
                                #    reward = 20
                                #    new_key_idx = 1
 
                                #if (new_key_idx == 1 and new_exit_idx == 0) and \
                                #   ((new_row, new_col) == (exit_coord[0][0],exit_coord[0][1])):
                                #    new_exit_idx = 1
                                
                                #if new_host_idx == 1 and ((new_row, new_col) == (self.hostage_coord[0],self.hostage_coord[1])):
                                #    reward = 20
                                #    new_host_idx = 1
                                
                                if ((new_row, new_col) == (self.exit_coord[0],self.exit_coord[1])):
                                    done = True
                                    reward = 100
                                    
                                #if (new_host_idx == 1 and new_key_idx == 1) or \
                                #    new_exit_idx == 1:
                                #    new_exit_idx=1
                                #else:
                                #    new_exit_idx=0
                                    
                                # If the agent bumps into a wall, it doesn't move and receives a negative rewards
                                if self.territory[new_row, new_col] == 1:
                                    new_row = row
                                    new_col = col
                                    reward = -10
                                
                                # If the agent bumps into a mine it reaches a terminal state and receives a negative rewards
                                if self.territory[new_row, new_col] == 2:
                                    new_row = row
                                    new_col = col
                                    done = True
                                    reward = -100 
                                
                                # If the agent has hostage and bumps into unsafe state it reaches a terminal state and receives a negative reward
                                #if new_host_idx == 1 and ((new_row, new_col) == (exit_coord[1][0],exit_coord[1][1])):
                                #    new_row = row
                                #    new_col = col
                                #    done = True
                                #    reward = -100 
                                
                                if self.territory[row, col] != 1 and self.territory[row, col] != 2:
                                    new_state = self.encode(
                                        new_row, new_col
                                    )
                                    
                                    self.P[state][action].append((1.0, new_state, reward, done))
                       
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = Discrete(num_actions) #int(num_actions)
        self.observation_space = Discrete(num_states) # Encoded states shape
        self.observation_space_decoded = self.displ_board.reshape(1,-1).shape[1] # Decoded states shape

    def get_empty_cells(self, n_cells):
        empty_cells_coord = np.where( self.territory == 0 )
        selected_indices = np.random.choice( np.arange(len(empty_cells_coord[0])), n_cells )
        selected_coordinates = empty_cells_coord[0][selected_indices], empty_cells_coord[1][selected_indices]
        
        if n_cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        
        return selected_coordinates
    
    def add_encoded_states(self):

        components = {'wall': [(0,1),(1,5),(2,5),(2,8),(3,0),(3,1),(3,3),(3,4),(3,6),(3,8),(4,2),(4,4),(4,6),
                       (4,7),(4,8),(5,2),(5,4),(7,6),(7,7),(7,8),(7,9),(8,1),(8,2),(8,3),(8,4),(8,5),(9,7)],
                      'mine': [(0,2),(1,8),(5,2),(5,8),(8,9)],
                      'safe':[(0,0)]
                      }
    
        #components['agent'] = [(4,5)]
        layer = 0
        for name, piece in components.items():
            #pos = (layer,) + piece.pos
            for index in piece:
                self.displ_board[layer][index] = 1
            layer += 1

    def decode_grid(self, state):
        row, col = np.array([v for _,v in enumerate(self.decode(state))])
        #row, col, key_idx, host_idx, exit_idx = np.array([v for _,v in enumerate(self.decode(state))])

        return self.displ_board.reshape(1,-1)


    def get_exit_cells(self, n_cells):
        y=np.arange(self.territory.shape[0] * self.territory.shape[1]).reshape(self.territory.shape[0], self.territory.shape[1])        
        border_coord_indices = list(np.unique(np.array(sorted(np.concatenate([y[0,:],y[-1,:],y[:,0], y[:,-1]]))).reshape(1,-1)))
        border_coord=np.zeros((2,len(border_coord_indices)),dtype = np.int8)
        border_coord[0] = [int(i) for i,ival in enumerate(y) for j,jval in enumerate(ival) if y[i,j] in border_coord_indices]
        border_coord[1] = [j for i,ival in enumerate(y) for j,jval in enumerate(ival) if y[i,j] in border_coord_indices]

        selected_indices = np.random.choice( np.arange(len(border_coord[0])), n_cells )
        selected_coordinates = border_coord[0][selected_indices], border_coord[1][selected_indices]
        
        if n_cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        
        return selected_coordinates

    
    # a key function for the q learning update rule
    def step(self, action):
        #print('Before',self.observations_array[self.s][0],self.observations_array[self.s][1])
        transitions = self.P[self.s][action]
        i = self.categorical_sample([t[0] for t in transitions], np.random)
        p, state, reward, done = transitions[i]
        self.s = state
        self.lastaction = action
        #print('After',self.observations_array[self.s][0],self.observations_array[self.s][1])

        if done == False:
            self.position_agent = (self.observations_array[self.s][0],self.observations_array[self.s][1])
            # Reset agent 
            layer = 3
            self.displ_board[layer][:] = 0
            # Move agent to new location
            #print(self.observations_array[self.s][0],self.observations_array[self.s][1])
            self.displ_board[layer][self.observations_array[self.s][0],self.observations_array[self.s][1]] = 1

        # Update time elapsed
        self.time_elapsed+=1
        
        #_, _, _, _, self.exit_steps = self.decode(self.s)

        if self.time_elapsed == self.time_limit:
            done = True
        
        return (int(state), reward, done, {"prob": p})
   

    def eval_reward(self):
        if (self.territory[self.position_agent[0],self.position_agent[1]]) == 2:
            return -100
        elif (self.position_agent == (self.exit_coord[0], self.exit_coord[1])):
            return 100
        else:
            return -1
    
    
    #def encode(self, agent_row, agent_col, key_idx, host_idx, exit_idx):
    def encode(self, agent_row, agent_col):
        i = agent_row
        i *= 10
        i += agent_col
        #i *= 2
        #i += key_idx
        #i *= 2
        #i += host_idx
        #i *= 2
        #i += exit_idx
        return i

    def decode(self, i):
        out = []
        #out.append(i % 2)
        #i = i // 2
        #out.append(i % 2)
        #i = i // 2
        #out.append(i % 2)
        #i = i // 2
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
        
        #agent_row, agent_col, key_idx, host_idx, exit_idx = self.decode(self.s)
        agent_row, agent_col = self.decode(self.s)
        
        full_repr = ""

        for r in range(self.height):
            
            line = ""
            
            for c in range(self.width):
                
                string_repr = self.dict_map_display[ envir_with_agent[r,c] ]
                line += "{0:2}".format(string_repr)

            full_repr += line + "\n"
        

        dist_agent_safe_row = abs(agent_row - self.exit_coord[0])
        dist_agent_safe_col = abs(agent_col - self.exit_coord[1])
        
        if dist_agent_safe_row <= 2 and dist_agent_safe_col <= 2 : # Agent at least 2 rows and 2 columns columns close to the safe zone state
            full_repr = full_repr.replace('A',colorise('A', "green", highlight=True))
        elif dist_agent_safe_row <= 4 and dist_agent_safe_col <= 4: # Agent at least 4 rows and 4 columns columns close to the safe zone state
            full_repr = full_repr.replace('A',colorise('A', "yellow", highlight=True))
        else: # Agent opened the door but did not find hostage
            full_repr = full_repr.replace('A',colorise('A', "red", highlight=True))
            exit_state = 1
        
        full_repr = full_repr.replace('S',colorise('S', "blue", highlight=True))
        
        #print('Key (Yellow):',key_idx,'/ Door (Blue):',exit_idx,'/ Hostage (Green):',host_idx,'- Agent start',self.start_agent)
        print('Safe zone(S):',self.exit_coord,' - Agent start:',self.start_agent,'Rows to S:',dist_agent_safe_row,'Columns to S:',dist_agent_safe_col)
        print('Green: Agent at least 2 rows and 2 columns close to safe zone')
        print('Yellow: Agent at least 4 rows and 4 columns close to safe zone')
        print('Red: Agent more than 4 rows or 4 columns far from safe zone\n')
        print(full_repr)

        
    def reset(self, return_info: bool = False):
        #This function resets the environment for the agent
        self.time_elapsed = 0
         
        invalid_position = True                                    


        while invalid_position == True:
            self.s = self.categorical_sample(self.initial_state_distrib, np.random)
            new_coord = self.states_array[self.s]
            #print(new_coord)
            #and ((new_coord[0],new_coord[1]) != (self.exit_coord[1][0],self.exit_coord[1][1])) \
            #and ((new_coord[0],new_coord[1]) != (self.hostage_coord[0],self.hostage_coord[1])) \
            #and ((new_coord[0],new_coord[1]) != (self.key_coord[0],self.key_coord[1])):

            if      ((new_coord[0],new_coord[1]) != (self.exit_coord[0],self.exit_coord[1])):
                invalid_position = False
                self.position_agent = (new_coord[0],new_coord[1])
        
        #self.position_agent = (6,9)
        # Assign position to agent
        self.start_agent = self.position_agent
        
        
        # Reset agent 
        layer = 3
        self.displ_board[layer][:] = 0
        # Move agent to new location
        self.displ_board[layer][new_coord[0],new_coord[1]] = 1
        
        self.lastaction = None
        if not return_info:
            return int(self.s)
        else:
            return int(self.s), {"prob": 1}
        #return observations

    def valid_actions(self, row=None, col=None, cell=None):
        actions = [0, 1, 2, 3]
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