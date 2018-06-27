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

    def extractPolicy(self,V,R,T):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.zeros(self.mdp.nStates)
        res = np.zeros((self.mdp.nActions, self.mdp.nStates))
        for j in range(self.mdp.nActions):
            res[j] = R[j] + self.mdp.discount * np.matmul(T[j], V)
        policy = np.argmax(res, axis=0)
        return policy 

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

        V = np.zeros(self.mdp.nStates)
        policy = np.zeros(self.mdp.nStates,int)

        count_sa = np.zeros((self.mdp.nStates, self.mdp.nActions)).astype(float)
        count_sas = np.zeros((self.mdp.nStates, self.mdp.nActions, self.mdp.nStates)).astype(float)

        c_reward = np.zeros(nEpisodes)
        for i in range(nEpisodes):
            state = s0
            R = initialR
            T = defaultT
            for j in range(nSteps):
                action = np.argmax(R[:,state])
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, self.mdp.nActions-1)

                reward, nextState = self.sampleRewardAndNextState(state, action)
                c_reward[i] += reward * (self.mdp.discount ** j)

                count_sa[state, action] += 1.0
                count_sas[state, action, nextState] += 1.0

                T[action, state, :] = np.divide(count_sas[state, action, :], count_sa[state, action])
                R[action, state] = (reward + (count_sa[state, action] - 1) * R[action, state]) / count_sa[state, action]

                V[state] = np.amax(R[:, state] + self.mdp.discount * np.sum(np.multiply(T[:, state, :], V[nextState]), axis=1))
                state = nextState

            policy = self.extractPolicy(V, R, T)

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
        epsilon = 1.0 / nIterations

        for i in range(nIterations):
            action = np.argmax(empiricalMeans)
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, self.mdp.nActions-1)

            reward = self.sampleReward(self.mdp.R[action])
            empiricalMeans[action] = (empiricalMeans[action] * count[action] + reward) / (count[action] + 1.0)
            count[action] += 1.0

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

        empiricalMeans = np.zeros(self.mdp.nActions).astype(float)
        count = np.zeros(self.mdp.nActions).astype(float)

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

            # update prior
            if reward == 1:
                prior[action, 0] += 1
            else:
                prior[action, 1] += 1

        return np.divide(empiricalMeans, count)

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        empiricalMeans = np.zeros(self.mdp.nActions).astype(float)
        count = np.zeros(self.mdp.nActions).astype(float)

        for i in range(nIterations):
            action = np.argmax(empiricalMeans + np.sqrt(np.divide(2.0 * np.log(self.mdp.nActions), count + 1.0)))
            reward = self.sampleReward(self.mdp.R[action])
            empiricalMeans[action] = (empiricalMeans[action] * count[action] + reward) / (count[action] + 1.0)
            count[action] += 1.0

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
                    policyParams[a, states[n]] += (1.0 / count[a,states[n]]) * g_n * gradient[a]

        return policyParams, c_reward
