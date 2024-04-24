import gym
from typing import Optional
from collections import defaultdict
import numpy as np
from typing import Callable
from policy import create_epsilon_policy, create_epsilon_schedule_policy, create_epsilon_heuristic_policy
from policy import ExponentialSchedule, DiffSchedule
from policy import wg_heuristic_1
from tqdm import trange


#generate_episode taken from ex4
#slightly edited to stop episodes at a given max time
def generate_episode(env: gym.Env, policy: Callable, es: bool = False, maxTime = None):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state,_ = env.reset()
    while True:
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, _, _ = env.step(action)
        episode.append((state, action, reward))
        if maxTime is not None:
            if done or len(episode) > maxTime:
                break
        else:
            if done:
                break
        state = next_state

    return episode


def sarsa(env: gym.Env, num_steps: int, gamma: float, eps_sched: ExponentialSchedule, step_size: float, returnpol: bool = False):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    #trying to copy from book alg

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_schedule_policy(Q, eps_sched)
    epLengths = [] #we need episode lengths for plotting

    timer = 0
    while timer < num_steps:
        # instead of generating episodes, we just reset the env and run the episodes/update Q live

        S, _ = env.reset() #initialize S...
        A = policy(S, timer) #and A
        done = False

        while not done:
            
            Sp, R, done, _, _ = env.step(A) #step once, if this isn't terminal then step again from there
            
            if done:
                Q[S][A] = Q[S][A] + step_size * (R + 0 - Q[S][A]) #0 represents the value of Q(terminal) for any s', a' pair in the terminal state
            else: #need to step ahead to see what values we need for evaluation
                Ap = policy(Sp, timer+1)
                Q[S][A] = Q[S][A] + step_size * (R + gamma*Q[Sp][Ap] - Q[S][A]) #proper SARSA equation
                S = Sp
                A = Ap

            timer += 1

        epLengths.append(timer) #episode lengths, needed for plotting
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.

    
    if returnpol:
        return policy, Q

    return epLengths

def nstep_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    eps_sched: ExponentialSchedule,
    step_size: float,
    n: int = 4,
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
        n (int): "n" in n-step sarsa
    """
    #trying to copy book algorithm

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_schedule_policy(Q, eps_sched)
    epLengths = [] #we need episode lengths for plotting

    timer = 0
    while timer < num_steps:

        S, _ = env.reset() # initialize S0
        A = policy(S, timer) # and A0
        T = np.inf
        t = 0
        tau = 0
        rewards = [0]
        states = [S] #and store them in arrays to be added to later
        actions = [A]
        done = False
        

        while tau < (T-1):
            
            if t < T: #first if from alg
                Sp, R, done, _, _ = env.step(A) #take action a
                timer += 1 #increment timer since we've stepped once
                rewards.append(R) #append rewards
                states.append(Sp) #append states

                if done:
                    T = t+1
                else:
                    Ap = policy(Sp, timer)
                    actions.append(Ap) #append actions
            
            tau = t - n + 1 #we need to actually update by the back-tracked Tau

            if tau >= 0: #assuming Tau refers to at least the first state in the episode
                G = np.sum([gamma**(i-tau-1)*rewards[i] for i in range(tau+1, min(tau+n+1,T+1))]) #calculate G for the n-step sarsa alg
                if tau+n < T:
                    G = G + gamma**n *Q[states[tau+n]][actions[tau+n]]
                
                Q[states[tau]][actions[tau]] = Q[states[tau]][actions[tau]] + step_size * (G - Q[states[tau]][actions[tau]]) #q update eq from the sarsa algorithm

            t = t+1
            S = Sp
            A = Ap

        epLengths.append(timer) #episode lengths, needed for plotting
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.

    return epLengths


def eps_soft_expect(Q, S, eps): #helper function for expected sarsa, will give the expected value of Q[S] over a given eps
    maxVal = max(Q[S]) #we're going to need 1-eps * the max plus eps/num actions times every value (incl the max itself)
    probRandEach = eps/len(Q[S]) #prob of picking each action randomly
    #prob of picking the maxval is just 1-eps

    sumRand = 0 #sum of all random expected values, i.e. prob of picking it * its value
    for k in range(len(Q[S])):
        sumRand += probRandEach*Q[S][k]

    expVal = (1-eps)*maxVal + sumRand #we get EV by summing prob of each random selection * value of each, plus 1-eps * the value of the greedy selection
    return expVal

def exp_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    eps_sched: ExponentialSchedule,
    step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    #similar to Q-learning, main difference is instead of argmax, we do the expected value of the next state according to our policy
    #our policy here is eps-soft so to get our probabilities we need to consider that

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_schedule_policy(Q, eps_sched)
    epLengths = [] #we need episode lengths for plotting

    timer = 0
    while timer < num_steps:
        # instead of generating episodes, we just reset the env and run the episodes/update Q live

        S, _ = env.reset() #initialize S
        
        done = False

        while not done:

            A = policy(S, timer) #take action A according to the policy
            
            Sp, R, done, _, _ = env.step(A) #step once, if this isn't terminal then step again from there
            
            if done:
                Q[S][A] = Q[S][A] + step_size * (R + 0 - Q[S][A]) #0 represents the value of Q(terminal) for any s', a' pair in the terminal state
            else: #need to step ahead to see what values we need for evaluation
                epsilon = eps_sched.value(timer)
                Q[S][A] = Q[S][A] + step_size * (R + gamma*eps_soft_expect(Q,Sp,epsilon) - Q[S][A]) #expected-sarsa eq
                S = Sp

            timer += 1

        epLengths.append(timer) #episode lengths, needed for plotting
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.

    return epLengths


def q_learning(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    eps_sched: ExponentialSchedule,
    step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    #similar to SARSA, main difference is instead of stepping, then presuming another step, we just take the argmax a over Q in the next state

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_schedule_policy(Q, eps_sched)
    epLengths = [] #we need episode lengths for plotting

    timer = 0
    while timer < num_steps:
        # instead of generating episodes, we just reset the env and run the episodes/update Q live

        S, _ = env.reset() #initialize S
        
        done = False

        while not done:

            A = policy(S, timer) #take action A according to the policy
            
            Sp, R, done, _, _ = env.step(A) #step once, if this isn't terminal then step again from there
            
            if done:
                Q[S][A] = Q[S][A] + step_size * (R + 0 - Q[S][A]) #0 represents the value of Q(terminal) for any s', a' pair in the terminal state
            else: #need to step ahead to see what values we need for evaluation
                qmaxsp = max(Q[Sp]) #argmax over potential A' for Q[S']
                Q[S][A] = Q[S][A] + step_size * (R + gamma*qmaxsp - Q[S][A]) #q-learning equation
                S = Sp

            timer += 1

        epLengths.append(timer) #episode lengths, needed for plotting
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.

    return epLengths

def sarsa_h(env: gym.Env, num_steps: int, gamma: float, eps_sched: DiffSchedule, h_sched: ExponentialSchedule, h_policy: Callable, step_size: float, returnpol: bool = False):
    """SARSA algorithm, with heuristics

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    #trying to copy from book alg

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_heuristic_policy(Q, eps_Sched=eps_sched, h_Sched=h_sched, h_pol=h_policy)
    epLengths = [] #we need episode lengths for plotting

    timer = 0
    while timer < num_steps:
        # instead of generating episodes, we just reset the env and run the episodes/update Q live

        S, _ = env.reset() #initialize S...
        A = policy(S, timer) #and A
        done = False

        while not done:
            
            Sp, R, done, _, _ = env.step(A) #step once, if this isn't terminal then step again from there
            
            if done:
                Q[S][A] = Q[S][A] + step_size * (R + 0 - Q[S][A]) #0 represents the value of Q(terminal) for any s', a' pair in the terminal state
            else: #need to step ahead to see what values we need for evaluation
                Ap = policy(Sp, timer+1)
                Q[S][A] = Q[S][A] + step_size * (R + gamma*Q[Sp][Ap] - Q[S][A]) #proper SARSA equation
                S = Sp
                A = Ap

            timer += 1

        epLengths.append(timer) #episode lengths, needed for plotting
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.

    
    if returnpol:
        return policy, Q

    return epLengths

def nstep_sarsa_h(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    eps_sched: DiffSchedule,
    h_sched: ExponentialSchedule,
    h_policy: Callable,
    step_size: float,
    n: int = 4,
):
    """N-step SARSA with heuristics

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
        n (int): "n" in n-step sarsa
    """
    #trying to copy book algorithm

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_heuristic_policy(Q, eps_Sched=eps_sched, h_Sched=h_sched, h_pol=h_policy)
    epLengths = [] #we need episode lengths for plotting

    timer = 0
    while timer < num_steps:

        S, _ = env.reset() # initialize S0
        A = policy(S, timer) # and A0
        T = np.inf
        t = 0
        tau = 0
        rewards = [0]
        states = [S] #and store them in arrays to be added to later
        actions = [A]
        done = False
        

        while tau < (T-1):
            
            if t < T: #first if from alg
                Sp, R, done, _, _ = env.step(A) #take action a
                timer += 1 #increment timer since we've stepped once
                rewards.append(R) #append rewards
                states.append(Sp) #append states

                if done:
                    T = t+1
                else:
                    Ap = policy(Sp, timer)
                    actions.append(Ap) #append actions
            
            tau = t - n + 1 #we need to actually update by the back-tracked Tau

            if tau >= 0: #assuming Tau refers to at least the first state in the episode
                G = np.sum([gamma**(i-tau-1)*rewards[i] for i in range(tau+1, min(tau+n+1,T+1))]) #calculate G for the n-step sarsa alg
                if tau+n < T:
                    G = G + gamma**n *Q[states[tau+n]][actions[tau+n]]
                
                Q[states[tau]][actions[tau]] = Q[states[tau]][actions[tau]] + step_size * (G - Q[states[tau]][actions[tau]]) #q update eq from the sarsa algorithm

            t = t+1
            S = Sp
            A = Ap

        epLengths.append(timer) #episode lengths, needed for plotting
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.

    return epLengths

def eps_soft_expect_h(Q, S, eps, h_val, h_pol): #helper function for expected sarsa, will give the expected value of Q[S] over a given eps
    maxVal = max(Q[S]) #we're going to need 1-eps * the max plus eps/num actions times every value (incl the max itself)
    probRandEach = eps/len(Q[S]) #prob of picking each action randomly
    #prob of picking the maxval is just 1-eps

    heurAct = h_pol(S) #this is the heuristic action we would take at the state
    heurVal = Q[S][heurAct]
    heurProb = h_val

    sumRand = 0 #sum of all random expected values, i.e. prob of picking it * its value
    for k in range(len(Q[S])):
        sumRand += probRandEach*Q[S][k]

    expVal = (1-(eps+h_val))*maxVal + heurVal*heurProb + sumRand #we get EV by summing prob of each random selection * value of each, plus prob of heuristic * its val, plus (1-prob(anything but optimal))*optimal val
    return expVal

def exp_sarsa_h(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    eps_sched: DiffSchedule,
    h_sched: ExponentialSchedule,
    h_policy: Callable,
    step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    #similar to Q-learning, main difference is instead of argmax, we do the expected value of the next state according to our policy
    #our policy here is eps-soft so to get our probabilities we need to consider that

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_heuristic_policy(Q, eps_Sched=eps_sched, h_Sched=h_sched, h_pol=h_policy)
    epLengths = [] #we need episode lengths for plotting

    timer = 0
    while timer < num_steps:
        # instead of generating episodes, we just reset the env and run the episodes/update Q live

        S, _ = env.reset() #initialize S
        
        done = False

        while not done:

            A = policy(S, timer) #take action A according to the policy
            
            Sp, R, done, _, _ = env.step(A) #step once, if this isn't terminal then step again from there
            
            if done:
                Q[S][A] = Q[S][A] + step_size * (R + 0 - Q[S][A]) #0 represents the value of Q(terminal) for any s', a' pair in the terminal state
            else: #need to step ahead to see what values we need for evaluation
                epsilon = eps_sched.value(timer)
                h_val = h_sched.value(timer)
                Q[S][A] = Q[S][A] + step_size * (R + gamma*eps_soft_expect_h(Q,Sp,epsilon, h_val, h_policy) - Q[S][A]) #expected-sarsa eq
                S = Sp

            timer += 1

        epLengths.append(timer) #episode lengths, needed for plotting
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.

    return epLengths

def q_learning_h(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    eps_sched: DiffSchedule,
    h_sched: ExponentialSchedule,
    h_policy: Callable,
    step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    #similar to SARSA, main difference is instead of stepping, then presuming another step, we just take the argmax a over Q in the next state

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_heuristic_policy(Q, eps_Sched=eps_sched, h_Sched=h_sched, h_pol=h_policy)
    epLengths = [] #we need episode lengths for plotting

    timer = 0
    while timer < num_steps:
        # instead of generating episodes, we just reset the env and run the episodes/update Q live

        S, _ = env.reset() #initialize S
        
        done = False

        while not done:

            A = policy(S, timer) #take action A according to the policy
            
            Sp, R, done, _, _ = env.step(A) #step once, if this isn't terminal then step again from there
            
            if done:
                Q[S][A] = Q[S][A] + step_size * (R + 0 - Q[S][A]) #0 represents the value of Q(terminal) for any s', a' pair in the terminal state
            else: #need to step ahead to see what values we need for evaluation
                qmaxsp = max(Q[Sp]) #argmax over potential A' for Q[S']
                Q[S][A] = Q[S][A] + step_size * (R + gamma*qmaxsp - Q[S][A]) #q-learning equation
                S = Sp

            timer += 1

        epLengths.append(timer) #episode lengths, needed for plotting
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.

    return epLengths


#We also need algorithms that end by #episodes, not #steps, so we make those down below:

def nstep_sarsa_by_episode(
    env: gym.Env,
    num_episodes: int,
    gamma: float,
    eps_sched: ExponentialSchedule,
    step_size: float,
    n: int = 4,
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
        n (int): "n" in n-step sarsa
    """
    #trying to copy book algorithm

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_schedule_policy(Q, eps_sched)
    epLengths = [] #we need episode lengths for plotting
    epEnds = [] #also want episode end times for plotting

    timer = 0
    epCount = 0
    while epCount < num_episodes:

        epStartTime = timer #needed to see how long each episode is for plotting later

        S, _ = env.reset() # initialize S0
        A = policy(S, timer) # and A0
        T = np.inf
        t = 0
        tau = 0
        rewards = [0]
        states = [S] #and store them in arrays to be added to later
        actions = [A]
        done = False
        

        while tau < (T-1):
            
            if t < T: #first if from alg
                Sp, R, done, _, _ = env.step(A) #take action a
                timer += 1 #increment timer since we've stepped once
                rewards.append(R) #append rewards
                states.append(Sp) #append states

                if done:
                    T = t+1
                else:
                    Ap = policy(Sp, timer)
                    actions.append(Ap) #append actions
            
            tau = t - n + 1 #we need to actually update by the back-tracked Tau

            if tau >= 0: #assuming Tau refers to at least the first state in the episode
                G = np.sum([gamma**(i-tau-1)*rewards[i] for i in range(tau+1, min(tau+n+1,T+1))]) #calculate G for the n-step sarsa alg
                if tau+n < T:
                    G = G + gamma**n *Q[states[tau+n]][actions[tau+n]]
                
                Q[states[tau]][actions[tau]] = Q[states[tau]][actions[tau]] + step_size * (G - Q[states[tau]][actions[tau]]) #q update eq from the sarsa algorithm

            t = t+1
            S = Sp
            A = Ap

        epLengths.append(timer-epStartTime) #episode lengths, needed for plotting
        epEnds.append(timer)
        epCount += 1
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.

    return epLengths, epEnds


def nstep_sarsa_h_by_episode(
    env: gym.Env,
    num_episodes: int,
    gamma: float,
    eps_sched: DiffSchedule,
    h_sched: ExponentialSchedule,
    h_policy: Callable,
    step_size: float,
    n: int = 4,
):
    """N-step SARSA with heuristics

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
        n (int): "n" in n-step sarsa
    """
    #trying to copy book algorithm

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_heuristic_policy(Q, eps_Sched=eps_sched, h_Sched=h_sched, h_pol=h_policy)
    epLengths = [] #we need episode lengths for plotting
    epEnds = [] #also want episode end times for plotting

    timer = 0
    epcount = 0
    while epcount < num_episodes:

        epStartTime = timer #needed to see how long each episode is for plotting later

        S, _ = env.reset() # initialize S0
        A = policy(S, timer) # and A0
        T = np.inf
        t = 0
        tau = 0
        rewards = [0]
        states = [S] #and store them in arrays to be added to later
        actions = [A]
        done = False
        

        while tau < (T-1):
            
            if t < T: #first if from alg
                Sp, R, done, _, _ = env.step(A) #take action a
                timer += 1 #increment timer since we've stepped once
                rewards.append(R) #append rewards
                states.append(Sp) #append states

                if done:
                    T = t+1
                else:
                    Ap = policy(Sp, timer)
                    actions.append(Ap) #append actions
            
            tau = t - n + 1 #we need to actually update by the back-tracked Tau

            if tau >= 0: #assuming Tau refers to at least the first state in the episode
                G = np.sum([gamma**(i-tau-1)*rewards[i] for i in range(tau+1, min(tau+n+1,T+1))]) #calculate G for the n-step sarsa alg
                if tau+n < T:
                    G = G + gamma**n *Q[states[tau+n]][actions[tau+n]]
                
                Q[states[tau]][actions[tau]] = Q[states[tau]][actions[tau]] + step_size * (G - Q[states[tau]][actions[tau]]) #q update eq from the sarsa algorithm

            t = t+1
            S = Sp
            A = Ap

        epLengths.append(timer-epStartTime) #episode lengths, needed for plotting
        epEnds.append(timer)
        epcount += 1
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.

    return epLengths, epEnds