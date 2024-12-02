11/25:
- Went through first two stable RL tutorials 
   - Getting started
   - Code with creating new environments, importing policies, and initializing model with policy, env
   - Training and evaluating model
   - Recording video of trained model
- Gym wrappers
   - wrap environments on different constraints such as limiting number of episodes, normalizing actions, 
   - important wrappers: Monitor, DummyVecEnv
   - coded and tested custom wrapper

11/26:
- Multiprocessing
   - still don’t really get
   - something to do with optimization between # of training envs and # of steps

11/27:
   - Callbacks
      - custom monitoring and saving models
      - important for saving models with best reward, keeping track of performance

11/28:
   - Worked through custom gyms tutorial
   - Implemented custom tictactoe environment

11/29:
   - cleaned up some code of tictactoe environment
   - went through tutorial of q-learning blackjack agent

11/30:
   - used baselines for a small ant env, recorded results
   - read math + pseudocode of ppo


12/1:
   - went through code of actual ppo 
      - there are so many specific weird details
      - don't fully understand the code