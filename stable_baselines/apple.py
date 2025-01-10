from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Apple game environment

class AppleGameEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to play this addictive apple game 
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    def __init__(self, grid_size=10, render_mode="console"):
        super(AppleGameEnv, self).__init__()
        self.render_mode = render_mode
        self.width = grid_size
        self.length = grid_size
        self.num_turns = 1000  # maximum number of turns
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.width * self.length,), dtype=np.float32 #flattened board instead of nxn
        )
        self.board = np.zeros((grid_size * grid_size,), dtype=np.float32)
        # action space is picking two corners of the grid
        # normalizing the action space. will transform back to the grid size later
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(4,),
            dtype=np.float32
        )
        # self.action_space = spaces.Box(low=np.array([1,1,1,1]), high=np.array([self.length,self.width,self.length,self.width]), dtype=np.int8)
    
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.num_turns = 1000
        # Generate integers first, then convert to float32 afterward
        self.board = np.random.randint(0, 10, size=(self.width * self.length)).astype(np.float32)
        
        return np.array(self.board), {}



    def step(self, action):
        self.num_turns -= 1
        # extract the two points and map from [-1,1] to coordinate ranges
        # action = np.zeros_like(action)
        action[[0,2]] = ((action[[0,2]] + 1) / 2) * (self.width + 1)  # map to [0,width+1] 
        action[[1,3]] = ((action[[1,3]] + 1) / 2) * (self.length + 1) # map to [0,length+1]
        x1,y1 = action[:2].astype(np.int8)
        x2,y2 = action[2:].astype(np.int8)
        top = min(y1,y2)
        bottom = max(y1,y2)
        left = min(x1,x2)
        right = max(x1,x2)


        # check if the sum of the points in the rectangle formed by the two corners is 10
        # if 10, set all entries in the rectangle to 0
        board_2d = self.board.reshape((self.length,self.width))
        rectangle = board_2d[left:right+1,top:bottom+1]
        sum = np.sum(rectangle)


        # base reward for getting close to 10
        # closeness_reward = -abs(sum-10)/100
        # don't know if the values above are calibrated well
        # reward = closeness_reward
        # reward = closeness_reward + size_penalty
        reward = 0

        if sum == 10:
            # actual reward
            non_zero_count = np.count_nonzero(rectangle)
            print(f"sum of 10 found!. non_zero_count: {non_zero_count}")
            board_2d[left:right+1,top:bottom+1] = 0
            reward += non_zero_count 
            print(f"reward: {reward}")
            self.board = board_2d.flatten()

        if sum == 0:
            reward -= 0.5

        terminated = bool((self.num_turns <= 0) or np.all(self.board == 0))
        truncated = False
        board_2d = self.board.reshape((self.length,self.width))
        return (
            np.array(self.board).astype(np.float32),
            reward,
            terminated,
            truncated,
            {}
        ) 

    def render(self):
        board2d = self.board.reshape((self.length,self.width))
        for i in range(self.length):
            for j in range(self.width):
              if j == self.width - 1:
                  print(board2d[i,j])
              else:
                  print(board2d[i,j],end="|")             
            if i != self.length - 1:
                print("-"*(2*self.width-1))

    def close(self):
        pass


env = AppleGameEnv(grid_size=8)
check_env(env,warn=True)
vec_env = make_vec_env(AppleGameEnv, n_envs=1, env_kwargs=dict(grid_size=8))
model = PPO("MlpPolicy", env, verbose=1,learning_rate = 0.0009).learn(10000)

# After training
obs, _ = env.reset()
for step in range(20):
    print(f"\nStep {step + 1}")
    
    # Get action and transform to coordinates
    action = model.predict(obs)[0]
    x1, y1 = ((action[[0,1]] + 1) / 2 * (env.width + 1)).astype(np.int8)
    x2, y2 = ((action[[2,3]] + 1) / 2 * (env.length + 1)).astype(np.int8)
    
    print(f"Selected points: ({x1},{y1}), ({x2},{y2})")
    
    # Show rectangle coordinates
    top, bottom = min(y1,y2), max(y1,y2)
    left, right = min(x1,x2), max(x1,x2)
    print(f"Rectangle: ({left},{top}) to ({right},{bottom})")
    
    # Show rectangle contents and sum
   
    # Take step and show reward
    obs, reward, done, _, _ = env.step(action)
    print(f"Reward: {reward:.3f}")

    board_2d = obs.reshape((env.length, env.width))
    rectangle = board_2d[left:right+1,top:bottom+1]
    print("\nRectangle contents:")
    print(rectangle)
    print(f"Sum: {np.sum(rectangle)}")
 
    
    # Show full grid
    print("\nFull grid:")
    print(board_2d)
    
    if done:
        print("\nEpisode finished!")
        break