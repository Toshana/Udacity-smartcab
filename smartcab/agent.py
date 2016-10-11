import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import time
import pandas as pd
import math

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""  

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        
        self.state = None # set state of agent
        self.epsilon = 0 # set exploitation/exploration parameter
        self.alpha = 0.5 # set learning Rate
        self.gamma = 0.25 # set Discount Rate
        self.action = None # set action 
        self.successes = [] # number of successful trips
        self.trial = 0 # current trial
        
        
        waypoints = ["forward", "left", "right"]
        traffic_lights = ["red", "green"]
        oncoming = [None, "forward", "left", "right"]
        left = [None, "forward", "left", "right"]
        right = [None, "forward", "left", "right"]
       
        self.q_table = {} # initialize q-table with set of (state action) pairs and q-values = 0
        for w in waypoints:
            for t in traffic_lights:
                for o in oncoming:
                    for l in left:
                        for r in right:
                            self.q_table[(w, t, o, l, r)] = {None: 0.0, "forward": 0.0, "left": 0.0, "right": 0.0}


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        
        def updatetrial(trial):
            trial += 1
            return trial
            
        self.trial = updatetrial(self.trial) # create counter to keep track of trials
        
        def decayparameter(parameter):
#            parameter = (100 - self.trial)/100.0 ## Linear decay
            parameter = math.exp(-self.trial/2.0) ## exponential decay
            return parameter
        
        
        self.epsilon = decayparameter(self.epsilon)
        self.successes.append(0)

    def getaction(self, state):
        ''' get action based on the state '''
        epsilon = self.epsilon
        q_values = self.q_table[state].values()
        max_q_value = max(q_values)
        count = q_values.count(max_q_value)
        if random.random() < epsilon:            
            action = random.choice(self.q_table[state].keys())
        elif count > 1:
            ties = []
            for x in self.q_table[state].items():
                if x[1] == max_q_value:
                    ties.append(x[0])
            action = random.choice(ties)
        else:
            action = [k for k, v in self.q_table[state].iteritems() if v == max_q_value][0]
        return action

    def UpdateQtable(self, s, s_prime, alpha, gamma, reward):
        """ Q-learning update equation """
        Q_value1 = self.q_table[s][self.action]
        Q_value_prime = max(self.q_table[s_prime].values())
        Q_value = (((1 - alpha) * Q_value1) + (alpha * (reward + (gamma * Q_value_prime))))
        self.q_table[s][self.action] = Q_value
        return Q_value


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
#        self.log = open('log.txt', 'a')    

        # TODO: Update state
        self.state = (self.next_waypoint,) + tuple(inputs.values())
        state = self.state
        
        # TODO: Select action according to your policy
        
        # valid_actions = [None, "forward", "left", "right"]
        # action = random.choice(valid_actions) 
        self.action = self.getaction(self.state)
        
        # Execute action and get reward
        reward = self.env.act(self, self.action)
        
        new_state = (self.next_waypoint,) + tuple(self.env.sense(self).values())        
        
        # TODO: Learn policy based on state, action, reward

        self.UpdateQtable(s = state, s_prime = new_state, alpha = self.alpha, gamma = self.gamma, reward = reward)
               
        location = self.env.agent_states[self]["location"] 
        destination = self.env.agent_states[self]["destination"]
        if location == destination:
                    self.successes[-1] = 1
                    
        if self.trial == 100:
            if location == destination:
                wins = self.successes.count(1)
#            first50_wins = self.successes[0:51].count(1)
#            second50_wins = self.successes[50:101].count(1)
                final_wins = self.successes[90:101].count(1)
#            print >> self.log.write("wins = {}, final wins = {}, alpha = {}, gamma = {}\n".format(wins, final_wins, self.alpha, self.gamma))
#            print "first 50 = {}".format(first50_wins)
#            print "last 50 = {}".format(second50_wins)
                print "wins = {}, final_wins = {}".format(wins, final_wins)


        #print >> self.log.write("{}, {}, {}\n".format(self.alpha, self.gamma, self.trial))
        # [debug]
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, self.action, reward)

def run(Alpha, Gamma):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    a.alpha = Alpha
    a.gamma = Gamma
    
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    
    sim.run(n_trials=100)  # run for a specified number of trials
    
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

params = [x/10.0 for x in xrange(1, 10)]
parameter_comparison = []
for a in params:
    for g in params:
        parameter_comparison.append([a, g])

if __name__ == '__main__':
#    for x in parameter_comparison:
#        a = x[0]
#        g = x[1]
        run(Alpha = 0.5, Gamma = 0.25)
        
    
