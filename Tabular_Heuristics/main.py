#Bennett Brain
#ex 6
#main file used for running code

from algorithms import sarsa, q_learning, exp_sarsa, nstep_sarsa
from algorithms import sarsa_h, q_learning_h, exp_sarsa_h, nstep_sarsa_h
from algorithms import nstep_sarsa_by_episode, nstep_sarsa_h_by_episode
import gym
import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from env import WindyGridworld, FourRoomsEnv
from tqdm import trange
from policy import ExponentialSchedule, DiffSchedule
from policy import wg_heuristic_1, wg_heuristic_2, wg_heuristic_3
from policy import fr_heuristic_1, fr_heuristic_2, fr_heuristic_3

def no_h_helper(alg, env, ns, g, a, e):
    epEnds = alg(env= env, num_steps= ns, gamma= g, step_size = a, eps_sched = e)

    epCount = [0]*8001
    for k in range(1,8001):
        epCount[k] = epCount[k-1]
        if k in epEnds: #if k is the end of an episode, increment our "completed episodes" counter by 1
            epCount[k] += 1 

    return epCount


def no_heuristic(ns, alpha = .5, trials = 10, stoWind = False): #ns is number of steps
    env = WindyGridworld(stochastic_wind=stoWind)

    xind = list(range(8001))

    # epEnds = on_policy_mc_control_epsilon_soft(env= env, num_steps= ns, gamma=1, epsilon=eps)
    
    #omitting mc-control because it always gets stuck-- see explanation in submitted document
    epCountData = np.zeros((4,trials,8001)) #index is which method (sarsa / q / exp_sarsa / nstep_sarsa in that order), then which trial index, then which time index for how many episodes are completed by that point
    epCountAvg = np.zeros((4,8001))

    eps_sched = ExponentialSchedule(1,.02,6000) #epsilon scheduling

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        epCountData[0,t] = no_h_helper(sarsa,env,ns,1,alpha,eps_sched)
        epCountData[1,t] = no_h_helper(q_learning,env,ns,1,alpha,eps_sched)
        epCountData[2,t] = no_h_helper(exp_sarsa,env,ns,1,alpha,eps_sched)
        epCountData[3,t] = no_h_helper(nstep_sarsa,env,ns,1,alpha,eps_sched)
    
    for k in range(8001):
        for agt_idx in range(4):
            epCountAvg[agt_idx,k] = np.mean(epCountData[agt_idx,:,k]) #averages across trials
    
    ecSerr = np.zeros((4,8001))

    for i in range(8001): #serr formula is stdev/sqrt(n)
        for a in range(4):
            ecSerr[a,i] = (np.std(epCountData[a,:,i]))/np.sqrt(trials)

    #with all trials done, time to make plots!
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    eba = .4 #error bar alpha

    ax.plot(xind,epCountAvg[0], label = "SARSA")
    ax.fill_between(range(0,8001), epCountAvg[0] - 1.96*ecSerr[0], epCountAvg[0] + 1.96*ecSerr[0], alpha=eba)
    ax.plot(xind,epCountAvg[1], label = "Q-Learning")
    ax.fill_between(range(0,8001), epCountAvg[1] - 1.96*ecSerr[1], epCountAvg[1] + 1.96*ecSerr[1], alpha=eba)
    ax.plot(xind,epCountAvg[2], label = "Expected SARSA")
    ax.fill_between(range(0,8001), epCountAvg[2] - 1.96*ecSerr[2], epCountAvg[2] + 1.96*ecSerr[2], alpha=eba)
    ax.plot(xind,epCountAvg[3], label = "N-Step SARSA, n = 4")
    ax.fill_between(range(0,8001), epCountAvg[3] - 1.96*ecSerr[3], epCountAvg[3] + 1.96*ecSerr[3], alpha=eba)
    ax.set_title("Average episode count over timesteps - no heuristic")
    ax.legend()
    ax.set_xlabel("time step")
    ax.set_ylabel("number of completed episodes")

    plt.show()


def h_helper(alg, env, ns, g, a, e, h, h_pol):
    epEnds = alg(env= env, num_steps= ns, gamma= g, step_size = a, eps_sched = e, h_sched = h, h_policy = h_pol)

    epCount = [0]*8001
    for k in range(1,8001):
        epCount[k] = epCount[k-1]
        if k in epEnds: #if k is the end of an episode, increment our "completed episodes" counter by 1
            epCount[k] += 1 

    return epCount

def heuristic(ns, alpha = .5, trials = 10, stoWind = False): #ns is number of steps
    env = WindyGridworld(stochastic_wind=stoWind)

    xind = list(range(8001))

    h_pol = wg_heuristic_1
    
    #omitting mc-control because it always gets stuck-- see explanation in submitted document
    epCountData = np.zeros((4,trials,8001)) #index is which method (sarsa / q / exp_sarsa / nstep_sarsa in that order), then which trial index, then which time index for how many episodes are completed by that point
    epCountAvg = np.zeros((4,8001))

    h_sched = ExponentialSchedule(.9,.001, 5000) #schedule for heuristic - follow closely at the start but drop off quickly
    eps_sched = DiffSchedule(1,.02,6000, h_sched) #eps will return as the difference between the h schedule and these values

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        epCountData[0,t] = h_helper(sarsa_h,env,ns,1,alpha,eps_sched, h_sched, h_pol)
        epCountData[1,t] = h_helper(q_learning_h,env,ns,1,alpha,eps_sched, h_sched, h_pol)
        epCountData[2,t] = h_helper(exp_sarsa_h,env,ns,1,alpha,eps_sched, h_sched, h_pol)
        epCountData[3,t] = h_helper(nstep_sarsa_h,env,ns,1,alpha,eps_sched, h_sched, h_pol)
    
    for k in range(8001):
        for agt_idx in range(4):
            epCountAvg[agt_idx,k] = np.mean(epCountData[agt_idx,:,k]) #averages across trials
    
    ecSerr = np.zeros((4,8001))

    for i in range(8001): #serr formula is stdev/sqrt(n)
        for a in range(4):
            ecSerr[a,i] = (np.std(epCountData[a,:,i]))/np.sqrt(trials)

    #with all trials done, time to make plots!
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    eba = .4 #error bar alpha

    ax.plot(xind,epCountAvg[0], label = "SARSA")
    ax.fill_between(range(0,8001), epCountAvg[0] - 1.96*ecSerr[0], epCountAvg[0] + 1.96*ecSerr[0], alpha=eba)
    ax.plot(xind,epCountAvg[1], label = "Q-Learning")
    ax.fill_between(range(0,8001), epCountAvg[1] - 1.96*ecSerr[1], epCountAvg[1] + 1.96*ecSerr[1], alpha=eba)
    ax.plot(xind,epCountAvg[2], label = "Expected SARSA")
    ax.fill_between(range(0,8001), epCountAvg[2] - 1.96*ecSerr[2], epCountAvg[2] + 1.96*ecSerr[2], alpha=eba)
    ax.plot(xind,epCountAvg[3], label = "N-Step SARSA, n = 4")
    ax.fill_between(range(0,8001), epCountAvg[3] - 1.96*ecSerr[3], epCountAvg[3] + 1.96*ecSerr[3], alpha=eba)
    ax.set_title("Average episode count over timesteps - with heuristic")
    ax.legend()
    ax.set_xlabel("time step")
    ax.set_ylabel("number of completed episodes")

    plt.show()

def direct_compare_wg(ne, alpha = .5, trials = 10, stoWind = False):

    nt = 4 #number of things we're testing

    env = WindyGridworld(stochastic_wind=stoWind)

    h_pol_good = wg_heuristic_1
    h_pol_bad = wg_heuristic_2 #this is our bad heuristic
    h_pol_mediocre = wg_heuristic_3

    h_sched = ExponentialSchedule(.9,.001, 6000) #schedule for heuristic - follow closely at the start but drop off quickly
    eps_sched_h = DiffSchedule(1,.02,6000, h_sched) #eps will return as the difference between the h schedule and these values

    eps_sched_noh = ExponentialSchedule(1,.02,6000)

    epLenData = np.zeros((nt,trials,ne)) #index is which method (sarsa / q / exp_sarsa / nstep_sarsa in that order), then which trial index, then which time index for how many episodes are completed by that point
    epLenAvg = np.zeros((nt,ne))

    epEndData = np.zeros((nt,trials,ne))
    epEndAvg = np.zeros((nt,ne))

    for t in trange(trials, desc="Trials"):
        epLenData[0,t], epEndData[0,t] = nstep_sarsa_h_by_episode(env=env,num_episodes=ne,gamma=1, step_size=alpha,eps_sched=eps_sched_h,h_sched=h_sched,h_policy=h_pol_good)
        epLenData[1,t], epEndData[1,t] = nstep_sarsa_by_episode(env=env,num_episodes=ne,gamma=1,eps_sched=eps_sched_noh,step_size=alpha)
        epLenData[2,t], epEndData[2,t] = nstep_sarsa_h_by_episode(env=env,num_episodes=ne,gamma=1, step_size=alpha,eps_sched=eps_sched_h,h_sched=h_sched,h_policy=h_pol_bad)
        epLenData[3,t], epEndData[3,t] = nstep_sarsa_h_by_episode(env=env,num_episodes=ne,gamma=1, step_size=alpha,eps_sched=eps_sched_h,h_sched=h_sched,h_policy=h_pol_mediocre)
    

    for k in range(ne):
        for agt_idx in range(nt):
            epLenAvg[agt_idx,k] = np.mean(epLenData[agt_idx,:,k]) #averages across trials
            epEndAvg[agt_idx,k] = np.mean(epEndData[agt_idx,:,k]) #averages across trials

    elSerr = np.zeros((nt,ne))
    eeSerr = np.zeros((nt,ne))

    for i in range(ne): #serr formula is stdev/sqrt(n)
        for a in range(nt):
            elSerr[a,i] = (np.std(epLenData[a,:,i]))/np.sqrt(trials)
            eeSerr[a,i] = (np.std(epEndData[a,:,i]))/np.sqrt(trials)
    
    xind = list(range(ne))

    #with all trials done, time to make plots!
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)

    eba = .4 #error bar alpha

    ax.plot(xind,epLenAvg[0], label = "N-Step SARSA, n = 4, using good Heuristic")
    ax.fill_between(range(0,ne), epLenAvg[0] - 1.96*elSerr[0], epLenAvg[0] + 1.96*elSerr[0], alpha=eba)
    ax.plot(xind,epLenAvg[1], label = "N-Step SARSA, n = 4, no Heuristic")
    ax.fill_between(range(0,ne), epLenAvg[1] - 1.96*elSerr[1], epLenAvg[1] + 1.96*elSerr[1], alpha=eba)
    ax.plot(xind,epLenAvg[2], label = "N-Step SARSA, n = 4, using adversarial Heuristic")
    ax.fill_between(range(0,ne), epLenAvg[2] - 1.96*elSerr[2], epLenAvg[2] + 1.96*elSerr[2], alpha=eba)
    ax.plot(xind,epLenAvg[3], label = "N-Step SARSA, n = 4, using mediocre Heuristic")
    ax.fill_between(range(0,ne), epLenAvg[3] - 1.96*elSerr[3], epLenAvg[3] + 1.96*elSerr[3], alpha=eba)
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Windy Gridworld Episode Length')
    ax.set_xlabel("Episodes Completed")
    ax.set_ylabel("Episode Length")

    ax2 = fig.add_subplot(1,2,2)

    ax2.plot(xind,epEndAvg[0], label = "N-Step SARSA, n = 4, using Heuristic")
    ax2.fill_between(range(0,ne), epEndAvg[0] - 1.96*elSerr[0], epEndAvg[0] + 1.96*elSerr[0], alpha=eba)
    ax2.plot(xind,epEndAvg[1], label = "N-Step SARSA, n = 4, no Heuristic")
    ax2.fill_between(range(0,ne), epEndAvg[1] - 1.96*elSerr[1], epEndAvg[1] + 1.96*elSerr[1], alpha=eba)
    ax2.plot(xind,epEndAvg[2], label = "N-Step SARSA, n = 4, using adversarial Heuristic")
    ax2.fill_between(range(0,ne), epEndAvg[2] - 1.96*elSerr[2], epEndAvg[2] + 1.96*elSerr[2], alpha=eba)
    ax2.plot(xind,epEndAvg[3], label = "N-Step SARSA, n = 4, using mediocre Heuristic")
    ax2.fill_between(range(0,ne), epEndAvg[3] - 1.96*elSerr[3], epEndAvg[3] + 1.96*elSerr[3], alpha=eba)
    ax2.legend()
    ax2.set_xlabel("Episodes Completed")
    ax2.set_title('Windy Gridworld Total Time')
    ax2.set_ylabel("Total Elapsed Time")

    plt.show()


def direct_compare_4room(ne, alpha = .5, trials = 10):

    nt = 4 #number of things we're testing

    env = FourRoomsEnv()
    gamma = .99

    h_pol_good = fr_heuristic_1 #this is our good heuristic
    h_pol_bad = fr_heuristic_2 #this is our bad heuristic
    h_pol_mediocre = fr_heuristic_3 #this is our mediocre heuristic

    h_sched = ExponentialSchedule(.9,.001, 6000) #schedule for heuristic - follow closely at the start but drop off quickly
    eps_sched_h = DiffSchedule(1,.02,6000, h_sched) #eps will return as the difference between the h schedule and these values

    eps_sched_noh = ExponentialSchedule(1,.02,6000)

    epLenData = np.zeros((nt,trials,ne)) #index is which method (sarsa / q / exp_sarsa / nstep_sarsa in that order), then which trial index, then which time index for how many episodes are completed by that point
    epLenAvg = np.zeros((nt,ne))

    epEndData = np.zeros((nt,trials,ne))
    epEndAvg = np.zeros((nt,ne))

    for t in trange(trials, desc="Trials"):
        epLenData[0,t], epEndData[0,t] = nstep_sarsa_h_by_episode(env=env,num_episodes=ne,gamma=gamma, step_size=alpha,eps_sched=eps_sched_h,h_sched=h_sched,h_policy=h_pol_good)
        epLenData[1,t], epEndData[1,t] = nstep_sarsa_by_episode(env=env,num_episodes=ne,gamma=gamma,eps_sched=eps_sched_noh,step_size=alpha)
        epLenData[2,t], epEndData[2,t] = nstep_sarsa_h_by_episode(env=env,num_episodes=ne,gamma=gamma, step_size=alpha,eps_sched=eps_sched_h,h_sched=h_sched,h_policy=h_pol_bad)
        epLenData[3,t], epEndData[3,t] = nstep_sarsa_h_by_episode(env=env,num_episodes=ne,gamma=gamma, step_size=alpha,eps_sched=eps_sched_h,h_sched=h_sched,h_policy=h_pol_mediocre)
    

    for k in range(ne):
        for agt_idx in range(nt):
            epLenAvg[agt_idx,k] = np.mean(epLenData[agt_idx,:,k]) #averages across trials
            epEndAvg[agt_idx,k] = np.mean(epEndData[agt_idx,:,k]) #averages across trials

    elSerr = np.zeros((nt,ne))
    eeSerr = np.zeros((nt,ne))

    for i in range(ne): #serr formula is stdev/sqrt(n)
        for a in range(nt):
            elSerr[a,i] = (np.std(epLenData[a,:,i]))/np.sqrt(trials)
            eeSerr[a,i] = (np.std(epEndData[a,:,i]))/np.sqrt(trials)
    
    xind = list(range(ne))

    #with all trials done, time to make plots!
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)

    eba = .2 #error bar alpha

    ax.plot(xind,epLenAvg[0], label = "N-Step SARSA, n = 4, using good Heuristic")
    ax.fill_between(range(0,ne), epLenAvg[0] - 1.96*elSerr[0], epLenAvg[0] + 1.96*elSerr[0], alpha=eba)
    ax.plot(xind,epLenAvg[1], label = "N-Step SARSA, n = 4, no Heuristic")
    ax.fill_between(range(0,ne), epLenAvg[1] - 1.96*elSerr[1], epLenAvg[1] + 1.96*elSerr[1], alpha=eba)
    ax.plot(xind,epLenAvg[2], label = "N-Step SARSA, n = 4, using adversarial Heuristic")
    ax.fill_between(range(0,ne), epLenAvg[2] - 1.96*elSerr[2], epLenAvg[2] + 1.96*elSerr[2], alpha=eba)
    ax.plot(xind,epLenAvg[3], label = "N-Step SARSA, n = 4, using mediocre Heuristic")
    ax.fill_between(range(0,ne), epLenAvg[3] - 1.96*elSerr[3], epLenAvg[3] + 1.96*elSerr[3], alpha=eba)
    ax.legend()
    ax.set_yscale('log')
    ax.set_title('Four Rooms Episode Length')
    ax.set_xlabel("Episodes Completed")
    ax.set_ylabel("Episode Length")

    ax2 = fig.add_subplot(1,2,2)

    ax2.plot(xind,epEndAvg[0], label = "N-Step SARSA, n = 4, using Heuristic")
    ax2.fill_between(range(0,ne), epEndAvg[0] - 1.96*elSerr[0], epEndAvg[0] + 1.96*elSerr[0], alpha=eba)
    ax2.plot(xind,epEndAvg[1], label = "N-Step SARSA, n = 4, no Heuristic")
    ax2.fill_between(range(0,ne), epEndAvg[1] - 1.96*elSerr[1], epEndAvg[1] + 1.96*elSerr[1], alpha=eba)
    ax2.plot(xind,epEndAvg[2], label = "N-Step SARSA, n = 4, using adversarial Heuristic")
    ax2.fill_between(range(0,ne), epEndAvg[2] - 1.96*elSerr[2], epEndAvg[2] + 1.96*elSerr[2], alpha=eba)
    ax2.plot(xind,epEndAvg[3], label = "N-Step SARSA, n = 4, using mediocre Heuristic")
    ax2.fill_between(range(0,ne), epEndAvg[3] - 1.96*elSerr[3], epEndAvg[3] + 1.96*elSerr[3], alpha=eba)
    ax2.legend()
    ax2.set_xlabel("Episodes Completed")
    ax2.set_title('Four Rooms Total Time')
    ax2.set_ylabel("Total Elapsed Time")

    plt.show()


def visualize_heuristic_shcedule(ns = 10000):
    h_sched = ExponentialSchedule(.9,.001, 6000) #schedule for heuristic - follow closely at the start but drop off quickly
    eps_sched_h = DiffSchedule(1,.02,6000, h_sched) #eps will return as the difference between the h schedule and these values

    h_plot = np.zeros(ns)
    e_plus_h_plot = np.zeros(ns)
    rest_of_prob = np.zeros(ns)
    for k in range(ns):
        h = h_sched.value(k)
        e = eps_sched_h.value(k)
        h_plot[k] = h
        e_plus_h_plot[k] = e+h
        rest_of_prob[k] = 1
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    fba = .2 #fill between alpha

    xind = list(range(ns))

    ax.plot(xind,h_plot, label = "Prob of following heuristic")
    ax.fill_between(range(0,ns), 0, h_plot, alpha=fba)
    ax.plot(xind,e_plus_h_plot, label = "Prob of acting randomly")
    ax.fill_between(range(0,ns), h_plot, e_plus_h_plot, alpha=fba)
    ax.plot(xind, rest_of_prob, label = "Prob of following greedy")
    ax.fill_between(range(0,ns), e_plus_h_plot, rest_of_prob, alpha=fba)
    ax.set_xlabel('Time steps')
    ax.set_ylabel('probability of following given policy')
    ax.legend()

    plt.show()





#no_heuristic(8000) #q3a
#heuristic(8000)

#direct_compare_wg(250,trials=20)
#direct_compare_4room(250,trials=20)
visualize_heuristic_shcedule()