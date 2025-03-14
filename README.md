# policy_gradient_homework

For this homework you may work in groups of up to 3 students. Turn in one submission for your group with all your team members' names when submit to Canvas.

## Setup
You should use the rl_env conda environment that you set up for [Homework 4](https://github.com/dsbrown1331/q-learning-homework) This should have almost all the packages you need. The only thing missing that will need to add is scipy. To install this just activate your environment and call `conda install scipy`. 

## Part 1a: Understanding and Implementing a Basic Policy Gradient Algorithm
Read through [Part 1](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) and [Part 2](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) of the Policy Gradient Tutorial on SpinningUp.

Now start [Part 3] (https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
This tutorial will walk you through the most basic policy gradient implementation. Follow along with the tutorial and stop before the "Expected Grad-Log-Prob Lemma" section.
Now take a look at `.\1_simple_pg_gymnasium.py` in this gitrepo (this is just an updated version of what is in the SpinningUp tutorial adapted to use gymnasium rather than the deprecated gym).
Make sure that everyone on your team understands each line: play around with the code, add some print lines, run through using a debugger, copy and paste code into your favorite AI program to have it explain parts that are confusing, etc. 
You can run the code by calling 
`
python 1_simple_pg_gymnasium.py
`
This will run a simple policy gradient algorithm on the CartPole environment, a classic RL control benchmark. Read up on CartPole [here](https://gymnasium.farama.org/environments/classic_control/cart_pole/.)

You won't see anything except printouts showing learning over time. By default it will run for 50 epochs. Feel free to play around with and change any of the hyperparameters and code. **Report on what you see and what it means. Is the agent learning? What happens to it's performance over time? Is it monotonically improving? Discuss why you think that is.**

## Part 1b: Understanding and Implementing a Basic Policy Gradient Algorithm

It's kind of boring to not see learning happening. Edit the code so that after each epoch you visually render a rollout from the current policy so you can watch the agent over time. You should see it learn to balance longer and longer the more experience it gets. 

## Part 2: Reducing Variance with Reward-to-Go

Continue reading the tutorial from [here](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#expected-grad-log-prob-lemma)
and read through the section "Implementing Reward-to-Go Policy Gradient". 
Make a copy of your simple policy gradient code and make the recommended changes to your code so that you implement the reward-to-go version. Make sure you understand what is going on.

Run your new code and compare the performance with your older version. Run each method 3-5 times to get a rough idea of average performance and report on what you find. Do you notice any differences? 

## Part 3a: Continuous Actions and Baselines
Read the rest of the tutorial starting with the section on "Baselines in Policy Gradients".

We will now explore using a baseline. To start with, let's first make our code more general purpose. You probably noticed that your simple policy gradient code only works with discrete actions. One of the benefits of policy gradient methods is that they work for both discrete and continuous action spaces. Take a look through the `core.py` file that is included in this gitrepo. This contains code that you can use to make your policy gradient code work automatically for either continuous or discrete envs in Gymnasium. Note if you want to use it on an image-based domain it will require more work, but it should work for any environment with a lower-dimensional state space represented by a vector which is fine for our purposes.

Create an updated version of your policy gradient code that works for either continuous or discrete action spaces using the helper functions in `core.py`. Note you won't need to worry about the `discount_cumsum` method, just focus on using the `MLPActorCritic` class in your code. We won't be using the value function prediction from the critic yet but we will in the next section. For now you can just leave it as a random network and ignore the v output when calling step.

You should now be able to run both continuous and discrete action domains. Test your code on CartPole to make sure it still works on a discrete action environment. Next pick a continuous action environment from Gymnasium to test your code. [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/) is probably one of the simplest continuous action and is a good choice if you want something that runs fast. You can also look at some of the [MuJoCo locomotion tasks](https://gymnasium.farama.org/environments/mujoco/). InvertedPendulum is also pretty simple. Hopper is a bit more complex but more interesting and you should see learning happen. Some of the others are quite complex so I wouldn't recommend trying them without a good CPU and GPU and a better codebase than what we've been playing around with.

Report on what environment you chose and the results of using your code. You may want to play around with hyperparameters a bit to see what works best. Report what you tried and learned.

## Part 3b: Continuous Actions and Baselines

Now we want to actually learn a value function to use as a baseline. We will still use the reward to go but will subtract off the baseline  $$b(s_t) = \hat{V}(s_t)$$, where $$\hat{V}(s_t)$$ is the approximate value function of the policy. We will be approximating the value function using a neural network. The good news for you is that this is already setup if you are using the `MLPActorCritic` class. If you're not using that class go back and follow the instructions for Part 3a. 

We've got a neural network but we need to train it. As mentioned in the tutorial we've been following, the simplest way to train a value function is via a standard MSE loss. Let's add that to our code. You will now have two PyTorch optimizers, one for the policy and one for the value function. You will still collect a batch of on-policy data. Same as before you'll update the policy using a policy gradient, but now you will subtract off the value of each state when computing this update. You will also update the value function using each batch of data via a standard MSE loss. You will be simply pushing the predicted value of each state with the true cumulative return experienced in the batch (the reward-to-go from that state). Note that to improve stability, you will probably want to perform several gradient steps of value learning for each step of policy update.

After working on this for a while, if you're really stuck, you can take a look here at this vanilla policy gradient code that also uses `core.py`: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/vpg.py
Note this code uses Generalized Advantage Estimation and some parallelization using mpi.

Test out your implementation on CartPole and a continuous action environment of your choice. Report on your results and how they compare to performance without a baseline.


## Extra Credit A:

Go through the vanilla policy gradient [documentation](https://spinningup.openai.com/en/latest/algorithms/vpg.html) and [implementation](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/vpg) and the PPO [documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html) and [implementation](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo) and read through the code. Remember, AI can be your friend here and help you understand complex lines of code. Write about what is different in these implementations compared to your simple policy gradient code. What is the same. What things did you learn while going through the code.

## Extra Credit B [Advanced].

Run the above VPG and PPO code on a problem of your choice and report how PPO compares with VPG using Generalized Advantage Estimation. Do they significantly outperform our simple policy gradient code? Note, this is non-trivial. You'll either need to strip out the MPI parallelization or install MPI (https://spinningup.openai.com/en/latest/user/installation.html). If you have Windows this will be tricky since OpenMPI does not have official support for Windows...you can try using WSL (Windows Subsystem for Linux) to use OpenMPI or if that doesn't work you'll need to strip out the MPI code (probably not worth your time, but could be a good learning experience). You will also have to update the code to work with Gymnasium rather than gym. ChatGPT is really good at this conversion if you're unsure of what to do. You can also try using ChatGPT, CoPilot, etc. to refactor the code to not require MPI. Let us know if that works.
