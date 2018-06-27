import numpy as np
import copy
import MDP

class RL2:
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

    def softmax(self, x):
        pi = np.exp(x - np.max(x))
        pi /= pi.sum()
        return pi

    def sampleSoftmaxPolicy(self,policyParams,state):
        '''Procedure to sample an action from stochastic policy
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        This function should be called by reinforce() to selection actions

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs: 
        action -- sampled action
        '''

        pi = self.softmax(policyParams[:,state])
        cumProb = np.cumsum(pi)
        action = np.where(cumProb >= np.random.rand(1))[0][0]
        
        return action

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy 
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs: 
        V -- final value function
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.mdp.nStates)
        policy = np.zeros(self.mdp.nStates,int)

        return [V,policy]    

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)

        return empiricalMeans

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)

        return empiricalMeans

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)

        return empiricalMeans

    def reinforce(self,s0,initialPolicyParams,nEpisodes,nSteps):
        '''reinforce algorithm.  Learn a stochastic policy of the form
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        This function should call the function sampleSoftmaxPolicy(policyParams,state) to select actions

        Inputs:
        s0 -- initial state
        initialPolicyParams -- parameters of the initial policy (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs: 
        policyParams -- parameters of the final policy (array of |A|x|S| entries)
        '''

        # policyParams = np.zeros((self.mdp.nActions,self.mdp.nStates))
        policyParams = initialPolicyParams

        for i in range(nEpisodes):
            states = np.zeros(nSteps).astype(int)
            actions = np.zeros(nSteps).astype(int)
            rewards = np.zeros(nSteps).astype(float)

            # generate episode s, a, r
            nextState = s0
            for j in range(nSteps):
                states[j] = nextState
                actions[j] = self.sampleSoftmaxPolicy(policyParams, states[j])
                rewards[j], nextState = self.sampleRewardAndNextState(states[j], actions[j])
            
            count = np.ones([self.mdp.nActions,self.mdp.nStates])
            # prev_log_pi = np.log(np.ones((self.mdp.nActions, self.mdp.nStates)) / self.mdp.nActions)
            # prev_log_pi = np.zeros((self.mdp.nActions, self.mdp.nStates))
            # loop for each step 
            # for s in range(self.mdp.nStates):
            #     prev_log_pi[:, s] = np.log(self.softmax(policyParams[:, s]))

            for n in range(nSteps):
                # calculate g_n
                g_n = 0.0
                for t in range(nSteps - n):
                    g_n += (self.mdp.discount ** t) * rewards[n+t]

                # to calculate the gradient of log pi
                # cur_log_pi = np.zeros((self.mdp.nActions, self.mdp.nStates))
                # for s in range(self.mdp.nStates):
                #     cur_log_pi[:,s] = np.log(self.softmax(policyParams[:,s]))
                # delta = cur_log_pi[actions[n],states[n]] - prev_log_pi[actions[n],states[n]]

                count[actions[n],states[n]] += 1.0
                # update policyParams by stochastic policy gradient
                # policyParams += alpha[actions[n],states[n]] * (self.mdp.discount ** n) * g_n * delta
                gradient = 1.0 - self.softmax(policyParams[:,states[n]])
                for a in range(self.mdp.nActions):
                    policyParams[a, states[n]] += (1.0 / count[a,states[n]]) * (self.mdp.discount ** n) * g_n * gradient[a]
                # prev_log_pi = copy.deepcopy(cur_log_pi)

        return policyParams    
