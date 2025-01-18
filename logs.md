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

12/2:
   - went through ppo atari code
   - i keep getting env name doesn't exist. version error?

12/3:
   - no work bc of veritasium work + finals :(

12/4: no work bc i was being a bum 

12/5: 
- read and wrote down some notes for survey on demonstrations


1/9:
finally finally back. will try to be consistent and work through some maniskill stuff now
maniskill doesn't work on my mac and using colab is so jank so i rented some gpus using vast.
but i haven't really worked with remote machines before and the whole process of sshing into them and then setting up the env and transferring files back took like 1 hr plus.
i was stuck for too long using the wrong port....
also setting up dependencies took longer than expected. 
copy-pasting from the colab notebook didn't work.
i just sent all the errors to claude and it fixed it.

- worked through quickstart notebook. mostly just ran stuff and read the code.


1/13:
- read through concepts section. didn't fully understand all of it though.


1/14:
- went through [Intro to task building](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/intro.html)
- went through code from push_cube.py -> push_cube.ipynb

1/15:
- went through most of [loading actors and articulations](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/loading_objects.html)
- finished up push_cube. ran video functionality
- having a crazy bug of trying to upload actors. 
is there something not implemented? can't upload ycb or other actors
- articulation works? only on the downloaded one (partnet-mobility:1030)

1/16, 17: no work 


1/18:
- setup baseline eval environment -> rl_eval.ipynb
- ran RL baseline code from ppo.py on PushCube-v1
- 







how to ssh into vast.ai machine:
1. start a new instance of vast gpu
2. run "ssh-keygen -t ed25519 -f ~/.ssh/vast_key which creates 
~/.ssh/vast_key (private key) and ~/.ssh/vast_key.pub (public key)
3. run "chmod 600 ~/.ssh/vast_key"
4. paste the public key into the "add ssh key" section in vast
5. run "ssh -i ~/.ssh/vast_key [the direct ssh connect command in vast (starts with -p ...)]"

opening a new window when connecting in vscode:
1. cmd + shift + p 
2. "Remote-SSH: Connect to Host..."
3. "+ Add new SSH Host"
4. enter the command from step 5 above
5. cmd + shift + p -> "Remote-SSH: Connect to Host..." -> [the address of the machine]


transferring files from vast to local (i can also just clone repo, add stuff, and push to github):
(do this from local machine)
nvm git is just better: use PAT token as password in username/password 

~~1. have a file which contains the PRIVATE KEY in ~/.ssh/vast_key
(idk if i need to redo this when i connect to another machine next time)~~

~~2. chmod 600 ~/.ssh/vast_key~~

~~3. ssh-keygen -y -f ~/.ssh/vast_key > ~/.ssh/vast_key.pub~~

~~4. copy the public key and paste it into "add ssh key" in vast~~

~~5. scp -i ~/.ssh/vast_key -P [PROXY PORT_NUMBER] -r root@ssh4.vast.ai:(PATH TO FILE IN VAST MACHINE) /Users/vincentcheng/Desktop/allcode/RL_stuff/maniskill (ONLY TRANSFER SINGULAR FILES)~~




wait given that i know how to write environments in the RL gym, i should be able to create 
an env with the apple game and train an agent to solve it right?
(spent 2 days over winter break trying to get the agent working but some stupid bug made me rage quit)