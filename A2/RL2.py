import numpy as np
import random
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

        model = MDP.MDP(defaultT, initialR, self.mdp.discount)
        V = np.zeros(model.nStates)
        policy = np.zeros(model.nStates,int)

        count_sa = np.zeros((model.nStates, model.nActions)).astype(float)
        count_sas = np.zeros((model.nStates, model.nActions, model.nStates)).astype(float)
        
        c_reward = np.zeros(nEpisodes)
        for i in range(nEpisodes):
            state = s0
            for j in range(nSteps):
                action = policy[state]
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, model.nActions-1)

                reward, nextState = self.sampleRewardAndNextState(state, action)
                c_reward[i] += reward * (model.discount ** j)

                count_sa[state, action] += 1.0
                count_sas[state, action, nextState] += 1.0

                model.T[action, state, :] = np.divide(count_sas[state, action, :], count_sa[state, action])
                model.R[action, state] = (reward + (count_sa[state, action]-1.0) * model.R[action, state]) / count_sa[state, action]
                policy, V, _ = model.policyIteration(policy)

                state = nextState

        return [V, policy, c_reward] 

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        empiricalMeans = np.zeros(self.mdp.nActions).astype(float)
        count = np.zeros(self.mdp.nActions).astype(float)
        c_reward = np.zeros(nIterations)

        for i in range(nIterations):
            # epsilon = 1.0 / (i + 1.0)
            epsilon = 0.1
            action = np.argmax(empiricalMeans)
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, self.mdp.nActions-1)

            reward = self.sampleReward(self.mdp.R[action])
            empiricalMeans[action] = (empiricalMeans[action] * count[action] + reward) / (count[action]+1.0)
            count[action] += 1.0
            c_reward[i] = reward

        return empiricalMeans, c_reward

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        empiricalMeans = np.zeros(self.mdp.nActions).astype(float)
        count = np.zeros(self.mdp.nActions).astype(float)
        c_reward = np.zeros(nIterations)

        for i in range(nIterations):
            # sample k rewards for each action
            avg_rewards = np.zeros(self.mdp.nActions).astype(float)
            for a in range(self.mdp.nActions):
                avg_rewards[a] = np.average(np.random.beta(prior[a,0], prior[a,1], k))

            # execute best action and receive reward
            action = np.argmax(avg_rewards)
            reward = self.sampleReward(self.mdp.R[action])
            empiricalMeans[action] += reward
            count[action] += 1.0
            c_reward[i] = reward

            # update prior
            if reward == 1:
                prior[action, 0] += 1
            else:
                prior[action, 1] += 1

        return np.divide(empiricalMeans, count), c_reward

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        empiricalMeans = np.zeros(self.mdp.nActions).astype(float)
        count = np.zeros(self.mdp.nActions).astype(float)
        c_reward = np.zeros(nIterations)

        for i in range(nIterations):
            action = np.argmax(empiricalMeans + np.sqrt(2.0 * np.log(i+1.0) / (count+1.0)))
            reward = self.sampleReward(self.mdp.R[action])
            empiricalMeans[action] = (empiricalMeans[action]*count[action] + reward) / (count[action]+1.0)
            count[action] += 1.0
            c_reward[i] = reward

        return empiricalMeans, c_reward

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
        alpha = 0.01
        c_reward = np.zeros(nEpisodes)
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
                c_reward[i] += rewards[j] * (self.mdp.discount ** j)
            
            count = np.ones([self.mdp.nActions,self.mdp.nStates]).astype(float)
            for n in range(nSteps):
                # calculate sum of discounted rewards
                g_n = 0.0
                for t in range(n, nSteps):
                    g_n += (self.mdp.discount ** t) * rewards[t]

                # update learning rate / count n(s,a)
                count[actions[n],states[n]] += 1.0
                # calculate partial derivative for log softmax
                gradient = - self.softmax(policyParams[:,states[n]])
                gradient[actions[n]] += 1.0

                # update policyParams by stochastic policy gradient
                for a in range(self.mdp.nActions):
                    # policyParams[a, states[n]] += (1.0 / count[a,states[n]]) * g_n * gradient[a]
                    policyParams[a, states[n]] += alpha * g_n * gradient[a]

        return policyParams, c_reward

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
        c_reward = np.zeros(nEpisodes)

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
                c_reward[i] += reward * (self.mdp.discount ** j)

                Q[action, state] = Q[action, state] + 1.0 / count[action, state] * (reward + self.mdp.discount * max(Q[:,nextState] - Q[action, state]) )
                state = nextState

        for i in range(self.mdp.nStates):
            policy[i] = np.argmax(Q[:,i])

        return [Q,policy,c_reward]    
