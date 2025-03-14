# policy_gradient_homework

For this homework you may work in groups of up to 3 students. Turn in one submission for your group with all your team members' names when submit to Canvas.

## Setup
You should use the rl_env conda environment that you set up for [Homework 4](https://github.com/dsbrown1331/q-learning-homework) This should have almost all the packages you need. The only thing missing that will need to add is scipy. To install this just activate your environment and call `conda install scipy`. 

## Part 1a: Understanding and Implementing a Basic Policy Gradient Algorithm
Read through Part 1 and Part 2 of the Policy Gradient Tutorial here: 
https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html
Now start Part 3: 
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
This tutorial will walk you through the most basic policy gradient implementation. Follow along with the tutorial and stop before the "Expected Grad-Log-Prob Lemma" section.
Now take a look at `.\1_simple_pg_gymnasium.py` in this gitrepo (this is just an updated version of what is in the SpinningUp tutorial adapted to use gymnasium rather than the deprecated gym).
Make sure that everyone on your team understands each line: play around with the code, add some print lines, run through using a debugger, copy and paste code into your favorite AI program to have it explain parts that are confusing, etc. 
You can run the code by calling 
`
python 1_simple_pg_gymnasium.py
`
This will run a simple policy gradient algorithm on the CartPole environment, a classic RL control benchmark. Read up on CartPole here: https://gymnasium.farama.org/environments/classic_control/cart_pole/.

You won't see anything except printouts showing learning over time. By default it will run for 50 epochs. Feel free to play around with and change any of the hyperparameters and code. **Report on what you see and what it means. Is the agent learning? What happens to it's performance over time? Is it monotonically improving? Discuss why you think that is.**

## Part 1b: Understanding and Implementing a Basic Policy Gradient Algorithm

It's kind of boring to not see learning happening. Edit the code so that after each epoch you visually render a rollout from the current policy so you can watch the agent over time. You should see it learn to balance longer and longer the more experience it gets. 

## Part 2: Reducing Variance with Reward-to-Go

Continue reading the tutorial from here: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#expected-grad-log-prob-lemma
and read through the section "Implementing Reward-to-Go Policy Gradient". 
Make a copy of your simple policy gradient code and make the recommended changes to your code so that you implement the reward-to-go version. Make sure you understand what is going on.

Run your new code and compare the performance with your older version. Run each method 3-5 times to get a rough idea of average performance and report on what you find. Do you notice any differences? 

## Part 3: Baselines

Read the rest of the tutorial starting with the section on "Baselines in Policy Gradients".
We will now explore using a baseline. To start with, let's first make our code more general purpose. 


