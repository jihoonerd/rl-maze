# rl-maze

## Description

Repository for learning basic concept of reinforcement learning.

This provides following approaches to solve maze.

* Random walk
* Policy gradient (REINFORCE)
* Sarsa
* Q-learning

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
