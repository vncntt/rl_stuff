from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class AppleGameEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, grid_size=10, render_mode="console"):
        super(AppleGameEnv, self).__init__()
        self.render_mode = render_mode
        self.width = grid_size
        self.length = grid_size
        self.num_turns = 1000
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.width * self.length,), dtype=np.float32
        )
        self.board = np.zeros((grid_size * grid_size,), dtype=np.float32)
        
        # New action space: x, y coordinates of top-left corner and orientation
        # orientation: 0 = horizontal (1x2), 1 = vertical (2x1)
        self.action_space = spaces.MultiDiscrete([self.width, self.length, 2])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            np.random.seed(seed)
        self.num_turns = 1000
        self.board = np.random.randint(1, 10, size=(self.width * self.length)).astype(np.float32)
        return np.array(self.board), {}

    def step(self, action):
        self.num_turns -= 1

        # Unpack action: x, y of top-left corner, and orientation
        x1, y1, orientation = action

        # Calculate second point based on orientation
        if orientation == 0:  # horizontal (1x2)
            x2, y2 = x1, y1 + 1
        else:  # vertical (2x1)
            x2, y2 = x1 + 1, y1

        # Ensure we don't go out of bounds
        if x2 >= self.width or y2 >= self.length:
            return self.board.astype(np.float32), -1, True, False, {}

        top = min(y1, y2)
        bottom = max(y1, y2)
        left = min(x1, x2)
        right = max(x1, x2)

        board_2d = self.board.reshape((self.length,self.width))

        rectangle = board_2d[left:right+1, top:bottom+1]
        sum_rect = np.sum(rectangle)

        reward = -100
        if sum_rect == 10: 
            non_zero_count = np.count_nonzero(rectangle)
            print(f"sum of 10 found!. non_zero_count: {non_zero_count}")
            board_2d[left:right+1, top:bottom+1] = 0
            reward = 1000
            print(f"reward: {reward}")
            self.board = board_2d.flatten()

        # if sum_rect > 20 or sum_rect < 5:
        #     reward -= 0.5


        terminated = bool((self.num_turns <= 0) or np.all(self.board == 0))
        truncated = False
        return self.board.astype(np.float32), reward, terminated, truncated, {}

    def render(self):
        board2d = self.board.reshape((self.length,self.width))
        for i in range(self.length):
            for j in range(self.width):
                if j == self.width - 1:
                    print(board2d[i,j])
                else:
                    print(board2d[i,j], end="|")             
            if i != self.length - 1:
                print("-"*(2*self.width-1))
    def close(self):
        pass

# Check and train with the new MultiDiscrete environment
env = AppleGameEnv()
check_env(env, warn=True)
vec_env = make_vec_env(AppleGameEnv, n_envs=1, env_kwargs=dict())
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.001,  # Slightly lower learning rate
    n_steps=2048,  # Increase steps per update
    batch_size=64,
    n_epochs=10,  # More training epochs per update
    gamma=0.95,  # Higher discount factor/rward

).learn(100000)  # More training steps

# Test policy
obs, _ = env.reset()
for step in range(1000):
    print(f"\nStep {step + 1}")
    action, _ = model.predict(obs)
    x1, y1, orientation = action
    
    # Calculate second point based on orientation
    if orientation == 0:  # horizontal (1x2)
        x2, y2 = x1, y1 + 1
    else:  # vertical (2x1)
        x2, y2 = x1 + 1, y1
        
    print(f"Selected points: ({x1},{y1}), ({x2},{y2})")
    print(f"Orientation: {'horizontal' if orientation == 0 else 'vertical'}")
    
    obs, reward, done, _, _ = env.step(action)
    print(f"Reward: {reward:.3f}")

    board_2d = obs.reshape((env.length, env.width))
    rectangle = board_2d[min(x1,x2):max(x1,x2)+1, min(y1,y2):max(y1,y2)+1]
    print("\nRectangle contents:")
    print(rectangle)
    print(f"Sum: {np.sum(rectangle)}")

    print("\nFull grid:")
    print(board_2d)

    if done:
        print("\nEpisode finished!")
        break
