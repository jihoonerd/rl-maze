# rl-maze

## Description

Repository for learning basic concept of reinforcement learning.

This provides following approaches to solve maze.

* Random walk
* Policy gradient (REINFORCE)
* Sarsa
* Q-learning

![maze-solve](https://jihoonerd.github.io/assets/images/posts/2019-05-25-ml-rl-maze-part2/REINFORCE.gif)

## Tutorial

|Contents|||
|---|---|---|
|Part 1: Maze Environment| ENG | [KOR](https://jihoonerd.github.io/machine-learning/ml-rl-maze-part1-kr/)
|Part 2: Policy Gradient (REINFORCE)| ENG | [KOR](https://jihoonerd.github.io/machine-learning/ml-rl-maze-part2-kr/)
|Part 3: Reward and Value| ENG | [KOR](https://jihoonerd.github.io/machine-learning/ml-rl-maze-part3-kr/)
|Part 4: Sarsa| ENG | [KOR](https://jihoonerd.github.io/machine-learning/ml-rl-maze-part4-kr/)
|Part 5: Q-learning| ENG | [KOR](https://jihoonerd.github.io/machine-learning/ml-rl-maze-part5-kr/)

## How to use

* Instantiate Q-learning agent and train:

```python
agent = Agent('q') # should be one of ['randwalk', 'pg', 'sarsa', 'q']
agent.train()
```

* Draw agent's move by using `agent.state_history`:

```python
maze = Maze()
maze.save_animation('maze.gif', agent.state_history) # it requires imagemagick
```