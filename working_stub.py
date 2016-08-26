# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        # self.last_state  = None
        # self.last_action = None
        # self.last_reward = None
        self.iteration=0
        self.last_state  = 0
        self.last_action = 0
        self.last_reward = 0
        #self.alpha = 0.000025            # learning rate
        self.alpha = 0.05
        self.discount = .97          # discount for Q-learning
        self.weights = np.ones(6)   # weights for each feature
        self.features = np.zeros(6)
        self.last_Q = 0             # used for difference update in Q-learning
        self.epsilon = .05          # do something random with probabily epsilon

    def reset(self):
        # self.last_state  = None
        # self.last_action = None
        # self.last_reward = None
        self.last_state  = 0
        self.last_action = 0
        self.last_reward = 0
        self.iteration+=1
        print self.iteration
        self.epsilon=self.epsilon*.85
        print self.epsilon
        print self.weights

    def compute_f(self, state, action):
        f1 = 0                      # monkey top
        f2 = 0                      # monkey botom
        f3 = state['tree']['top']   # top of tree
        f4 = state['tree']['bot']   # bottom of tree
        f5 = state['tree']['dist']  # distance from monkey to tree
        f5 = f5-25
        f6 = 0
        if action==0:
            f1 = state['monkey']['top']+state['monkey']['vel']
            f2 = state['monkey']['bot']+state['monkey']['vel']
            f6 = state['monkey']['vel']-4
        else:
            f1 = state['monkey']['top']+16
            f2 = state['monkey']['bot']+16
            f6 = state['monkey']['vel']+20
        #normalize the features
        #f1=416-f1
        tree_midpoint = (f3+f4)/2.

        foo1 =f1
        foo2=f2

        f1= abs(f1-200)
        f2= abs(f2-200)

        f1=f1/200.
        f2=f2/200.

        f3=abs(foo1-tree_midpoint)
        f4=abs(foo2-tree_midpoint)

        f3=f3/200.
        f4=f4/200.

        f5=f5/600.

        f6=f6/20. #how to normalize velocity?
        # f3 = f1>f3
        # f4 = f2>f4
        # f3= f1-f3
        # f4=f2-f4
        # f3 = f3*f1 #interactions???
        # f4=f4*f2

        return np.array([f1]+[f2]+[f3]+[f4]+[f5]+[f6])      # temporary features

    def compute_q(self, state, action):
        return self.weights.dot(self.compute_f(state,action))

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        # update weights
        q_s0 = self.compute_q(state,0)
        q_s1 = self.compute_q(state,1)
        a_prime = np.array([q_s0, q_s1]).argmax()
        max_q = np.array([q_s0, q_s1]).max()
        difference = self.last_reward + self.discount * max_q - self.last_Q
        new_weights = self.weights + self.alpha * difference * self.features
        self.weights = new_weights
        #self.weights = new_weights / new_weights.sum()
        self.features = self.compute_f(state, a_prime)

        # get new action
        new_action = np.array([q_s0, q_s1]).argmax()
        self.last_Q = self.compute_q(state, new_action)
        # new_action = npr.rand() < 0.1
        # new_state  = state

        # do something random with probability epsilon
        if npr.rand() < self.epsilon:
            new_action = npr.choice([0,1])

        self.last_action = new_action
        self.last_state  = state
        #print 'difference = ',difference
        # print 'state = ', self.last_state
        # print 'action = ', self.last_action
        # print 'self.features = ', self.features
        # print 'self.weights = ', self.weights

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        #print 'reward = ', reward, '\n'
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

    print 'main function'
    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games.
    print 'running games'
    run_games(agent, hist, 150, 10)

    # Save history.
    np.save('hist',np.array(hist))


