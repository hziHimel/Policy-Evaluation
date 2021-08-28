import numpy as np
from itertools import product

#declaring essential parameters
OUTSIDE_GRID_REWARD = -1
GAMMA = 1
#THRESHOLD = 0.0001
GRID_SIZE = (4,4)
TERMINAL_STATE1 = (0,0)
TERMINAL_STATE2 = (GRID_SIZE[0]-1, GRID_SIZE[1]-1)
REWARD = -1
ACTION_PROB = 0.25 #Probality of choosing an action. We assume that agent can pick an action out of possible 4 actions with equal probablity.



value = np.zeros(GRID_SIZE)     #Initializing value function
                                #IMPORTANT: There will be reward 0 for being in terminal states, not for transition
actions = ['L', 'R', 'U', 'D']
STATES = list(product([i for i in range(GRID_SIZE[0])], [i for i in range(GRID_SIZE[1])]))


#Helping functions:

def state_change(current_state, action):  #This function takes current state and agent's action as input. It will determine#next state of agent.
                                                                
    global GRID_SIZE   
    if action == 'L':
        next_state = (current_state[0], current_state[1] - 1)
    elif action == 'R':
        next_state = (current_state[0], current_state[1] + 1)
    elif action == 'U':
        next_state = (current_state[0] - 1, current_state[1])
    elif action == 'D':
        next_state = (current_state[0] + 1, current_state[1])
    else:
        next_state = current_state

    #This block checks wether an action will lead to outside of the grid. If that happens, agent will remain at same state
    if next_state[0] < 0 or next_state[0] > (GRID_SIZE[0]-1) or next_state[1] < 0 or next_state[1] > (GRID_SIZE[1]-1):
        next_state = current_state
        
    return next_state

def state_value_update(state):      #This function will update value of a given state
    state_updated_value = 0.0
    x_next, y_next = 0,0
    global REWARD
    global ACTION_PROB
    
    for a in actions:    
        (x_next, y_next) = state_change(state, a)     #Next state due to the particular action
        state_updated_value += ACTION_PROB* (REWARD + GAMMA * value[x_next, y_next]) #Update state value for that action using Bellman's equation

    return state_updated_value
        
def update_all_states():                              #Using previous functions, it will update all states' value
    v_new = np.zeros(GRID_SIZE)
    global TERMINAL_STATE1
    global TERMINAL_STATE2
    for s in STATES:
        if (s ==  TERMINAL_STATE1) or (s == TERMINAL_STATE2):
            None
        else:
            v_new[s[0], s[1]] = state_value_update(s)
    
    return v_new   

def main_loop(num_iters):                           #main loop that colesces all other functions
    i = 0
    while(i < num_iters):
        global value
        v_new = update_all_states()
        value = v_new
        i += 1
    print(f"number of iterations is: {num_iters}\n")
    print("Value fuction after evaluating policy: ")
    print(value)

main_loop(1000)                                    #You can use any integer instead of 1000. I suggest to start with small integer like 1,2,3 etc. to observe changes of value function
