import numpy as np
import MDP
import random

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        Q = np.zeros([self.mdp.nActions,self.mdp.nStates])
        policy = np.zeros(self.mdp.nStates,int)
        count = np.ones([self.mdp.nActions,self.mdp.nStates])
        cum_reward = np.zeros(nEpisodes)

        Q = initialQ
        for i in range(nEpisodes):
            state = s0
            for j in range(nSteps):
                action = 0
                p = random.uniform(0, 1)
                if p <= epsilon:
                    action = random.randint(0, self.mdp.nActions-1)
                else:
                    action = np.argmax(Q[:,state])
                count[action, state] += 1
                [reward, nextState] = self.sampleRewardAndNextState(state, action)
                cum_reward[i] += reward

                Q[action, state] = Q[action, state] + 1.0 / count[action, state] * (reward + self.mdp.discount * max(Q[:,nextState] - Q[action, state]) )
                state = nextState

        for i in range(self.mdp.nStates):
            policy[i] = np.argmax(Q[:,i])

        return [Q,policy,cum_reward]    