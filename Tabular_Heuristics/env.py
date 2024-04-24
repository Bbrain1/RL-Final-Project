from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np

class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]

class KingAction(IntEnum):
    #possible actions with King's Moves (upleft, upright, etc)

    LEFT = 0
    DOWNLEFT = 1
    DOWN = 2
    DOWNRIGHT = 3
    RIGHT = 4
    UPRIGHT = 5
    UP = 6
    UPLEFT = 7
    #STAYSTILL = 8 #comment out for final run

def kings_actions_to_dxdy(action: KingAction) -> Tuple[int, int]:
    """
    Helper function to map King's action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        KingAction.LEFT: (-1, 0),
        KingAction.DOWN: (0, -1),
        KingAction.RIGHT: (1, 0),
        KingAction.UP: (0, 1),
        KingAction.DOWNLEFT: (-1, -1),
        KingAction.DOWNRIGHT: (1, -1),
        KingAction.UPLEFT: (-1, 1),
        KingAction.UPRIGHT: (1, 1),
        #KingAction.STAYSTILL: (0,0), #comment out for final run
    }
    return mapping[action]


class WindyGridworld(Env):
    """Windy Gridworld gym environment.

    Custom gym environment for the windy gridwolrd.
    """

    def __init__(self, start_pos = (0,3), goal_pos=(7, 3), kings_moves = False, stochastic_wind = False, timeout = False, time_max = 459) -> None:
        self.rows = 7
        self.cols = 10

        # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
        # wind corresponds to the strength of the wind for each x coordinate (i.e. 1 in  for 3,4,5,8, 2 for 6,7, and 0 otherwise)

        self.wind = [0,0,0,1,1,1,2,2,1,0]

        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.agent_pos = None

        self.timeout = timeout
        self.timemax = time_max
        self.timer = None
        self.kings_moves = kings_moves
        self.swind = stochastic_wind

        if self.kings_moves:
            self.action_space = spaces.Discrete(len(KingAction))
        else:
            self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> Tuple[int, int]:
        """Reset agent to the starting position.

        Returns:
            observation (Tuple[int,int]): returns the initial observation
        """
        self.agent_pos = self.start_pos
        self.timer = 0

        return self.agent_pos, {}

    
    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        #timer for timeout/tracking time-to-goal
        self.timer += 1

        # TODO move, with a wind adjustment
        # specifically, according to the example from the book, it seems the wind at the agent's current position affects the movement, not the wind at the agent's *next* position
            
        if self.kings_moves:
            dxdy = kings_actions_to_dxdy(action)
        else:
            dxdy = actions_to_dxdy(action)

        if self.swind:
            windAdj = self.wind[self.agent_pos[0]] #basic wind value is determined by X position
            if windAdj != 0: #we only use stochastic wind in tiles where the wind exists in the first place (I think...?)
                radj = np.random.randint(low=-1,high=2) #this provides -1, 0, or 1 with equal probability which is what we adjust our wind by if the wind exists
                windAdj += radj
            next_pos = (self.agent_pos[0]+dxdy[0], self.agent_pos[1] + dxdy[1] + windAdj) #wind adjustment is applied to y
        else:
            next_pos = (self.agent_pos[0]+dxdy[0], self.agent_pos[1] + dxdy[1] + self.wind[self.agent_pos[0]]) #wind adjustment is applied to y, according to x position

        #need to check if we end up off-grid, if so just fix in the direction we're off by (i.e. if we moved right and up to end up at a theoretical 5,8 then we should only be able to end at 5,6)
        if next_pos[0] < 0:
            next_pos = (0, next_pos[1])
        elif next_pos[0] >= self.cols:
            next_pos = (self.cols-1, next_pos[1])
        if next_pos[1] < 0:
            next_pos = (next_pos[0], 0)
        elif next_pos[1] >= self.rows:
            next_pos = (next_pos[0], self.rows-1)
        

        # Set self.agent_pos
        self.agent_pos = next_pos

        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 0 #0 for reaching goal, so combined with the -1 for not, the agent should seek to terminate episodes as quickly as possible
        else:
            done = False
            reward = -1.0 #in this example we have a constant -1 reward for not reaching the goal


        #timeout in case we go past maximum, only if "timeout" bool is set to true
        if (self.timeout) and (self.timer > self.timemax):
            done = True
            reward = -1

        return self.agent_pos, reward, done, {}, {} #added a second {} because other envs seem to have that so I updated episode in an earlier part accordingly
    
class FourRoomsEnv(Env):
    """Four Rooms gym environment.

    This is a minimal example of how to create a custom gym environment. By conforming to the Gym API, you can use the same `generate_episode()` function for both Blackjack and Four Rooms envs.
    """

    def __init__(self, goal_pos=(10, 10), timeout = False, time_max = 459) -> None:
        self.rows = 11
        self.cols = 11

        # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
        self.walls = [
            (0, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 0),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 9),
            (5, 10),
            (6, 4),
            (7, 4),
            (9, 4),
            (10, 4),
        ]

        self.start_pos = (0, 0)
        self.goal_pos = goal_pos
        self.agent_pos = None

        self.timeout = timeout
        self.timemax = time_max
        self.timer = None

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> Tuple[int, int]:
        """Reset agent to the starting position.

        Returns:
            observation (Tuple[int,int]): returns the initial observation
        """
        self.agent_pos = self.start_pos
        self.timer = 0

        return self.agent_pos, {}
    



    def expected_return(self, V, state:Tuple[int,int], action:Action, gamma: int): #added for extra credit
        action2 = (action - 1)%4 #10% chance to move counterclockwise of intended
        action3 = (action + 1)%4 #10% chance to move clockwise of intended


        sp1 = self.wall_check(state, action)
        sp2 = self.wall_check(state, action2)
        sp3 = self.wall_check(state, action3)

        rew = 0
        if state == self.goal_pos:
            return 1 #if we move from the goal position we just get a reward of 1 so that should be our exact value there

        ret = rew + gamma*(.8 *V[sp1] + .1 *V[sp2] + .1 * V[sp3])

        return ret

    def wall_check(self, state, action): 
        #returns the original position if we'd bump a wall, or the next position if we wouldn't
        #this is after the random noise in action movement so assume the action is always the one actually taken

        dxdy = actions_to_dxdy(action)
        next_pos = (state[0]+dxdy[0], state[1]+dxdy[1]) 

        # If the next position is a wall or out of bounds, stay at current position
        if ((next_pos[0] == self.cols) or (next_pos[1] == self.rows) or (-1 in next_pos) or (next_pos in self.walls)):
            # print("bumped wall") #bugtest print
            next_pos = state
        
        return next_pos

    
    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """
        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 1.0
        else:
            done = False
            reward = 0.0

        #timer for the timeout mentioned in the question
        self.timer += 1
        #timeout in case we go past maximum, only if "timeout" bool is set to true
        if (self.timeout) and (self.timer > self.timemax):
            done = True
            reward = 0.0


        # TODO modify action_taken so that 10% of the time, the action_taken is perpendicular to action (there are 2 perpendicular actions for each action).
        # You can reuse your code from ex0
            
        randNum = np.random.rand() #random number in range [0,1)
        if randNum < .1: #10% of the time
            action_taken = (action - 1)%4 #because of the way the actions are ordered, this is clockwise by 1
        elif randNum < .2: #since this is an elif, this means the range < .2 but >= .1
            action_taken = (action + 1)%4 #because of the way the actions are ordered, this is counterclockwise by 1
        else: #if neither above the above are called, then we just take the input action
            action_taken = action


        # TODO calculate the next position using actions_to_dxdy()
        # You can reuse your code from ex0

        #this returns agent_pos if we'd bump a wall but a true next position if not
        next_pos = self.wall_check(self.agent_pos,action_taken)

        # Set self.agent_pos (this does nothing if the agent bumped a wall)
        self.agent_pos = next_pos

        return self.agent_pos, reward, done, {}, {} #added a second {} because other envs seem to have that so I updated episode in an earlier part accordingly