import numpy as np
import matplotlib.pyplot as plt
from itertools import product

#declaring essential parameters
OUTSIDE_GRID_REWARD = -1
GAMMA = 1
#THRESHOLD = 0.0001
GRID_SIZE = (4,4)
TERMINAL_STATE1 = (0,0)
TERMINAL_STATE2 = (GRID_SIZE[0]-1, GRID_SIZE[1]-1)
REWARD = -1
ACTION_PROB = 0.7   # Probability of an action to be done as it ought to be.
                    # For example, if agent chooses 'Right', there will be 70% chance it will go right, and 30% chance it will take
                    # any of other 3 actions. In this case, noise = 0.3
EACH_ACTION_NOISE = 0.1


#Initializing value function
#IMPORTANT: There will be reward 0 for being in terminal states, not for transition

value = np.zeros(GRID_SIZE)

actions = ['L', 'R', 'U', 'D']

STATES = list(product([i for i in range(GRID_SIZE[0])], [i for i in range(GRID_SIZE[1])]))

policy = dict.fromkeys(STATES, 'R') #Initializing random policy. This one will be updated through iterations
policy[TERMINAL_STATE1] = 'X'
policy[TERMINAL_STATE2] = 'X'

print(policy)

def state_change(current_state, action):        #This function will return next state depending on current state and action
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
        
def noisy_actions(state, a):   # This function will take an action and find values for that state for due to other 3 actions
    global actions        
    global EACH_ACTION_NOISE
    global value
    noisy_sum = 0.0        
    for i in actions:   
        if i != a:
            (x, y) = state_change(state, i)
            noisy_sum += EACH_ACTION_NOISE * (REWARD + GAMMA * value[x, y])  

    return noisy_sum

def state_value_update(state):
    next_state_values = []
    state_updated_value = 0.0
    x_next, y_next = 0,0
    global REWARD
    global ACTION_PROB
    global policy    
    a = policy[state]     
    (x_next, y_next) = state_change(state, a)     #Next state due to the particular action
    state_updated_value += ACTION_PROB* (REWARD + GAMMA * value[x_next, y_next]) #Update state value for that action
    state_updated_value += noisy_actions(state, a) #Expected value of other actions due to noise

    return state_updated_value

def update_all_states():
    v_new = np.zeros(GRID_SIZE)
    global TERMINAL_STATE1
    global TERMINAL_STATE2
    global policy
    for s in STATES:
        if (s ==  TERMINAL_STATE1) or (s == TERMINAL_STATE2):
            None
        else:
            v_new[s[0], s[1]] = state_value_update(s)
    
    return v_new

def update_policy(state):               #This function will update policy of a state
    global actions
    adjacent_states = []
    for a in actions:
        next_state = state_change(state, a)
        if next_state != state:
            adjacent_states.append((value[next_state], a))
    sorted_list = sorted(adjacent_states, key = lambda x : x[0], reverse = True)        #Sorting adjacent states' value and finding the maximum
    
    return sorted_list[0][1]

def update_all_policies():              #This function will update all states' policy
    global TERMINAL_STATE1
    global TERMINAL_STATE2
    global policy    
    for s in STATES:
        if (s ==  TERMINAL_STATE1) or (s == TERMINAL_STATE2):
            None
        else:
            policy[s] = update_policy(s)

def print_policy():                     #Helping function to print policies like a matrix
    global policy
    global GRID_SIZE
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            print(policy[(i, j)], end = '\t')
        print('')

def main_loop(num_iters):               # I have created the loop to run num_iters time. This can be done with comparing consecutive value function's diffence and a threshold value
    i = 0
    while(i < num_iters):
        global value
        v_new = update_all_states()
        update_all_policies()
        value = v_new
        i += 1
    print(f"number of iterations is: {num_iters}\n")
    print("Value fuction after evaluating policy: ")
    print(value)
    print_policy()


main_loop(1000)