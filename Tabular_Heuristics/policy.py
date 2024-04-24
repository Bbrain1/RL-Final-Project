import numpy as np
from collections import defaultdict
from typing import Callable, Tuple, Sequence
import math
from env import Action

def wg_heuristic_1(state): #this is our "good" heuristic
    x = state[0]
    y = state[1]

    #we know the goal is 7,3
    #since the wind always pushes up, we should have some weight on going "Down" if we're above 3 in y
    if x < 7: #if we're left of the goal, move right
        action = Action.RIGHT
    elif y >= 3: #if we're above the goal, move down to adjust
        action = Action.DOWN
    elif x > 7: #if we're right of the goal, move left
        action = Action.LEFT
    elif y < 3: #and if all else fails, move up
        action = Action.UP
    return action

def wg_heuristic_2(state):
    x = state[0]
    y = state[1]

    #this is our adversarial policy, i.e. *bad* advice

    if x < 7: #left of the goal, go left
        action = Action.LEFT
    elif x > 7: #right of the goal, go right
        action = Action.RIGHT
    elif y >= 3: #if we can skip the goal by going up, do so
        action = Action.UP
    else: #flee to the right to make our lives harder
        action = Action.RIGHT 
    
    return action
    

def wg_heuristic_3(state):
    x = state[0]
    y = state[1]

    #this is our mid-ground policy, i.e. mediocre but attempting-to-be-good advice.  Doesn't really account for wind

    if x <= 7: #left of the goal, go right
        action = Action.RIGHT
    elif x > 7: #right of the goal, go left
        action = Action.LEFT
    elif y > 3: #if we're above the goal, go down
        action = Action.DOWN
    else: #if we're above the goal, go up
        action = Action.UP
    
    return action


def blind_moveto(x,y,targx,targy): #helper for 4rooms - this moves WITHOUT accounting for walls
    r = np.random.random()

    act = None

    if x == targx: #if x is on target, follow by Y
        if y < targy:
            act = Action.UP
        else: 
            act = Action.DOWN
    elif y == targy: #if y is on target, follow by X
        if x < targx:
            act = Action.RIGHT
        else:
            act = Action.LEFT

    elif r < .5: #if neither is on target, pick one at random
        if y < targy:
            act = Action.UP
        else: 
            act = Action.DOWN
    
    else:
        if x < targx:
            act = Action.RIGHT
        else:
            act = Action.LEFT

    return act

def fr_heuristic_1(state): #this is our good heuristic
    x = state[0]
    y = state[1]

    act = None

    #cover hallways first
    if (x == 1) and (y in [4,5]): #upper hallway of r1
        act = Action.UP
    elif (y == 1) and (x in[4,5]): #right hallway of r1
        act = Action.RIGHT
    elif (y == 8) and (x in [4,5]): #right hallway of top-left room
        act = Action.RIGHT
    elif (x == 8) and (y in [3,4]): #right hallway of bottom-right room
        act = Action.UP
    
    #now for the rooms themselves - move to the next exit hallway
    elif (x < 5) and (y < 5): #bottom-left room
        if x > y: #bottom-right half of this room
            act = blind_moveto(x,y,4,1)
        else: #top-left half of this room
            act = blind_moveto(x,y,1,4)
    elif (x < 5) and (y > 5): #top-left room
        act = blind_moveto(x,y,4,8)
    elif (x > 5) and (y < 4): #bottom-right room
        act = blind_moveto(x,y,8,3)
    else: #in goal room
        act = blind_moveto(x,y,10,10)

    return act

def fr_heuristic_2(state): #adversarial heuristic, always moves away from goal
    r = np.random.random()
    act = None

    if r < .5: #left and down is always away from the goal
        act = Action.LEFT
    else:
        act = Action.DOWN
    
    return act

def fr_heuristic_3(state): #mediocre heuristic- doesn't account for walls but tries to go towards the goal

    x = state[0]
    y = state[1]

    act = blind_moveto(x,y,10,10)

    return act


#epsilon scheduling taken from ex8

class ExponentialSchedule:
    def __init__(self, value_from, value_to, num_steps):
        """Exponential schedule from `value_from` to `value_to` in `num_steps` steps.

        $value(t) = a \exp (b t)$

        :param value_from: Initial value
        :param value_to: Final value
        :param num_steps: Number of steps for the exponential schedule
        """
        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps

        # YOUR CODE HERE: Determine the `a` and `b` parameters such that the schedule is correct
        self.a = value_from
        self.b = math.log(value_to/self.a)/(num_steps-1)

    def value(self, step) -> float:
        """Return exponentially interpolated value between `value_from` and `value_to`interpolated value between.

        Returns {
            `value_from`, if step == 0 or less
            `value_to`, if step == num_steps - 1 or more
            the exponential interpolation between `value_from` and `value_to`, if 0 <= steps < num_steps
        }

        :param step: The step at which to compute the interpolation
        :rtype: Float. The interpolated value
        """

        # YOUR CODE HERE: Implement the schedule rule as described in the docstring,
        # using attributes `self.a` and `self.b`.
        if step >= self.num_steps-1:
            value = self.value_to
        elif step <= 0:
            value = self.value_from
        else:
            value = self.a * math.exp(self.b * step)
        
        return value
    
class DiffSchedule: #difference schedule- idea is we want to be "e2" in a schedule where e1 + e2 total follows a schedule but 
    #we have to just make up the difference
    def __init__(self, value_from, value_to, num_steps, e1_sched: ExponentialSchedule):
        """Exponential schedule from `value_from` to `value_to` in `num_steps` steps.

        $value(t) = a \exp (b t)$

        :param value_from: Initial value
        :param value_to: Final value
        :param num_steps: Number of steps for the exponential schedule
        :param e1_sched: an exponential schedule we want to complement
        """
        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps
        self.e1_sched = e1_sched

        # YOUR CODE HERE: Determine the `a` and `b` parameters such that the schedule is correct
        self.a = value_from
        self.b = math.log(value_to/self.a)/(num_steps-1)

    def value(self, step) -> float:
        """Return exponentially interpolated value between `value_from` and `value_to`interpolated value between.

        Returns {
            `value_from`, if step == 0 or less
            `value_to`, if step == num_steps - 1 or more
            the exponential interpolation between `value_from` and `value_to`, if 0 <= steps < num_steps
        }

        :param step: The step at which to compute the interpolation
        :rtype: Float. The interpolated value
        """

        # YOUR CODE HERE: Implement the schedule rule as described in the docstring,
        # using attributes `self.a` and `self.b`.
        if step >= self.num_steps-1:
            value = self.value_to
        elif step <= 0:
            value = self.value_from
        else:
            value = self.a * math.exp(self.b * step)
            
        #at this point "value" is what we need to reach.
        
        e1val = self.e1_sched.value(step)
        
        targ_val = value-e1val
        
        return targ_val


#argmax reused from ex1
def argmax(arr: Sequence[float]) -> int:
    """Argmax that breaks ties randomly

    Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

    Note: np.argmax returns the first index that matches the maximum, so we define this method to use in EpsilonGreedy and UCB agents.
    Args:
        arr: sequence of values
    """
    # TODO
    max = arr[0]
    maxIndList = [0]
    for ind in range(1,len(arr)): #can skip 0 index because we initialize max and maxIndList with 0

        if arr[ind] == max: #add all equal max indexes to a list to randomly select between them at the end
            maxIndList.append(ind)

        if arr[ind] > max: #update max, reset max list
            max = arr[ind]
            maxIndList = [ind] #resetting maxIndList

    maxIndRand = np.random.choice(maxIndList)

    return maxIndRand

def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Tuple) -> int:
        # TODO
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily
        if np.random.random() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = argmax(Q[state]) #this should break ties randomly, reusing argmax from ex1

        return action

    return get_action

def create_epsilon_schedule_policy(Q: defaultdict, eps_Sched: ExponentialSchedule) -> Callable:
    """Creates an epsilon soft policy from Q values, with exponential decay for epsilon.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Tuple, t_step: int) -> int:
        # TODO
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily
        eps_Val = eps_Sched.value(t_step) #new way of getting epsilon
        if np.random.random() < eps_Val: 
            action = np.random.randint(num_actions)
        else:
            action = argmax(Q[state]) #this should break ties randomly, reusing argmax from ex1

        return action

    return get_action

def create_epsilon_heuristic_policy(Q: defaultdict, h_Sched: ExponentialSchedule, eps_Sched: DiffSchedule, h_pol: Callable) -> Callable:

    """Creates an epsilon soft policy from Q values and a heuristic, with exponential decay for both epsilon and a heuristic.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Tuple, t_step) -> int:
        #now uses an epsilon and h-epsilon value to determine whether to pick randomly, use a heuristic, or pick argmax
        eps_Val = eps_Sched.value(t_step) 
        h_val = h_Sched.value(t_step)

        r = np.random.random()
        if r < eps_Val: 
            action = np.random.randint(num_actions)
        elif r < eps_Val + h_val: #in the "heuristic" range
            action = h_pol(state)
        else:
            action = argmax(Q[state]) #this should break ties randomly, reusing argmax from ex1

        return action

    return get_action
