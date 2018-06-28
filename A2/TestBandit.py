import numpy as np
import MDP
import RL2


def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.3],[0.5],[0.7]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)
banditProblem = RL2.RL2(mdp,sampleBernoulli)

# Test epsilon greedy strategy
c_reward = np.zeros(200)
e_means = np.zeros(mdp.nActions)
for _ in range(1000):
	empiricalMeans_t, c_reward_t = banditProblem.epsilonGreedyBandit(nIterations=200)
	e_means += empiricalMeans_t
	c_reward += c_reward_t
e_means /= 1000.0
c_reward /= 1000.0

print "\nepsilonGreedyBandit results"
print e_means
print c_reward

# Test Thompson sampling strategy
c_reward = np.zeros(200)
e_means = np.zeros(mdp.nActions)
for _ in range(1000):
	empiricalMeans_t, c_reward_t = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=200)
	e_means += empiricalMeans_t
	c_reward += c_reward_t
e_means /= 1000.0
c_reward /= 1000.0

print "\nthompsonSamplingBandit results"
print e_means
print c_reward

# Test UCB strategy
c_reward = np.zeros(200)
e_means = np.zeros(mdp.nActions)
for _ in range(1000):
	empiricalMeans_t, c_reward_t = banditProblem.UCBbandit(nIterations=200)
	e_means += empiricalMeans_t
	c_reward += c_reward_t
e_means /= 1000.0
c_reward /= 1000.0

print "\nUCBbandit results"
print e_means
print c_reward
