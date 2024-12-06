## Online and Offline RL
- online RL
    - "pure RL"
    - learning through exploring and interacting with the environment
    - pros:
        - has high ceiling. can learn anything
    - cons:
        - bad sample efficiency
        - less stable
    - bad sample efficiency
- offline RL
    - fixed dataset to learn from (expert demonstrations)
    - pros:
        - sample efficient
        - doesn't need full env access
    - cons:
        - limited by dataset quality 
        - generalizability
        - no exploring

- What is "learning from demonstrations"?
    - normally, your model can only learn by taking the actions from the current policy, but this is extremely
    sample inefficient. each time you update your policy, you have to re-collect all your data
    - learning from expert domonstration means having groud-truth labels of human-expert demonstrations for 
    the model to learn from so it's not total exploration. the flavor of supervised learning?
        - how is this implemented?

- What is the difference between offline RL and imitation learning?


## Learning from Demosntrations Offline
### Learning the Policy
- imitation learning
    - limited by the policy used to generate the demonstrations
    - two types: behavioral cloning, inverse RL
    - behavioral cloning
    - inverse RL
        - predicts a reward function based on the states and actions then performs RL using that reward function
        - there's no way this works wtf. 
- use imitation learning to get a base model then fine-tune using online methods
    - imitation learning = pretraining?

### Skill Discovery
- learning general useful *skills* 
    - walking is a general skill for both agents who want to play basketball or soccer

### Learning World Model
- the model wants to learn the entire MDP (S,A,T,R)
- for probabilistic transition functions, the model wants to minimize 
KL divergence of its learned transition function and the real one
- for a good world model to be learned, the demonstration dataset has to 
cover a wide range of states, actions (even bad ones)
- once a good world model is learning, it can be used to plan
    - can devolve quickly since learned world model is lossy
        - fix: fine-tune the world model by interacting with the real MDP

### Learning Reward Function:
- some demos don't have reward functions for every step but rather sparse (success,failure) signals
- "inverse RL" to train a proxy reward function given the sparse rewards

### Representation Learning
- train another model on demos to get "cleaner" representations and then trian the model of this new dataset
    - hmm seems sus. you lose information, but it can also be seen as synthesizing information in a better way

## Using Demos for Online Learning

### Demos as Off-Policy Experience
- use the demo trajectories optimizing the policy. can be re-used for the entirety of training
- replay buffer is a mix of agent exploration and demos 

### Demos as On-Policy Regularization
- add a term in the loss encouraging learned "occupancy" to be closer to the demonstrations

### Demo as Reference for Reward
- define reward as distance between demo and agent trajectories
- adversarial imitation learning

### Demo as Curriculum of start states
- limit the starting distribution by ones in the demos






Why would anyone do something that's not imitation learning?
For something like robotics it doesn't seem like you want/need too much exploration.
Your goal is consistent, general performance. 


