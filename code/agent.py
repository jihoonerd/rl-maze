import numpy as np


class Agent:

    def __init__(self, strategy="q"):
        self.strategy = strategy
        self.state = 0
        self.pi = None
        self.action = None
        self.state_history = [[0, np.nan]]
        self.theta = np.array([[np.nan, 1, np.nan, np.nan],    # S0: Start
                               [np.nan, 1, np.nan, 1],         # S1
                               [np.nan, np.nan, 1, 1],         # S2
                               [np.nan, np.nan, 1, np.nan],    # S3
                               [np.nan, 1, np.nan, np.nan],    # S4
                               [np.nan, 1, 1, 1],              # S5
                               [1, 1, np.nan, 1],              # S6
                               [1, np.nan, np.nan, 1],         # S7
                               [np.nan, np.nan, 1, np.nan],    # S8
                               [1, np.nan, 1, np.nan],         # S9
                               [np.nan, 1, 1, np.nan],         # S10
                               [np.nan, np.nan, 1, 1],         # S11
                               [1, 1, np.nan, np.nan],         # S12
                               [1, 1, np.nan, 1],              # S13
                               [1, np.nan, np.nan, 1],         # S14
                               ])                              # S15: End

        if strategy in ["sarsa"]:
            a, b = self.theta.shape
            self.Q = np.random.rand(a, b) * self.theta
        elif strategy in ["q"]:
            a, b = self.theta.shape
            self.Q = np.random.rand(a, b) * self.theta * 0.1
        elif strategy in ["randwalk", "pg"]:
            pass
        else:
            raise ValueError("undefined strategy {}".format(self.strategy))
        self.initial_covert_from_theta_to_pi()

    def initial_covert_from_theta_to_pi(self):

        [m, n] = self.theta.shape
        pi = np.zeros((m, n))

        if self.strategy in ["rand_walk", "sarsa", "q"]:
            for i in range(0, m):
                pi[i, :] = self.theta[i, :] / np.nansum(self.theta[i, :])
        elif self.strategy == "pg":
            beta = 1.0
            exp_theta = np.exp(beta * self.theta)
            for i in range(0, m):
                pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
        else:
            raise ValueError("undefined strategy {}".format(self.strategy))
        self.pi = np.nan_to_num(pi)

    def get_action(self, epsilon):
        """
        Return action by ε-greedy

        epsilon: explore action by epsilon(%), exploit action by (1-epsilon(%))   
        """
        direction = ["U", "R", "D", "L"]

        if np.random.rand() < epsilon:
            next_direction = np.random.choice(
                direction, p=self.pi[self.state, :])
        else:
            next_direction = direction[np.nanargmax(self.Q[self.state, :])]

        if next_direction == "U":
            action = 0
        elif next_direction == "R":
            action = 1
        elif next_direction == "D":
            action = 2
        elif next_direction == "L":
            action = 3
        else:
            raise ValueError("undefined direction {}".format(next_direction))
        return action

    def move_next_state(self, action=None, epsilon=None):
        """Returns next states for given policy and state"""
        direction = ["U", "R", "D", "L"]

        if action:
            next_direction = direction[action]
        else:
            next_direction = np.random.choice(
                direction, p=self.pi[self.state, :])

        if next_direction == "U":
            s_next = self.state - 4
        elif next_direction == "R":
            s_next = self.state + 1
        elif next_direction == "D":
            s_next = self.state + 4
        elif next_direction == "L":
            s_next = self.state - 1
        else:
            raise ValueError("Unknown Direction %s" % next_direction)

        self.state = s_next

    def update_Q(self, s, a, r, s_next, a_next, Q, eta, gamma):
        """Update Q for 'sarsa' and 'q' strategy"""

        if self.strategy in ['sarsa']:
            if s_next == 15:
                Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
            else:
                Q[s, a] = Q[s, a] + eta * \
                    (r + gamma * Q[s_next, a_next] - Q[s, a])
        elif self.strategy in ['q']:
            if s_next == 15:
                Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
            else:
                Q[s, a] = Q[s, a] + eta * \
                    (r + gamma * np.nanmax(Q[s_next, :]) - Q[s, a])
        else:
            raise ValueError("undefined strategy {}".format(self.strategy))
        self.Q = Q

    def update_theta(self):
        """Update theta for 'pg' strategy"""
        eta = 0.1
        T = len(self.state_history) - 1

        [m, n] = self.theta.shape
        delta_theta = self.theta.copy()

        for i in range(0, m):
            for j in range(0, n):
                if not(np.isnan(self.theta[i, j])):
                    sa_i = [SA for SA in self.state_history if SA[0] == i]
                    sa_ij = [SA for SA in self.state_history if SA == [i, j]]
                    sa_i_tot = len(sa_i)
                    sa_ij_tot = len(sa_ij)
                    delta_theta[i, j] = (
                        sa_ij_tot - self.pi[i, j] * sa_i_tot) / T

        self.theta = self.theta + eta * delta_theta

    def solve_maze(self, epsilon=None, eta=None, gamma=None):

        if self.strategy in ["sarsa", "q"]:
            a = a_next = self.get_action(epsilon)
            while True:
                a = a_next
                self.state_history[-1][1] = a
                s = self.state
                self.move_next_state(action=a, epsilon=epsilon)
                s_next = self.state
                self.state_history.append([s_next, np.nan])

                if s_next == 15:
                    r = 1
                    a_next = np.nan
                else:
                    r = 0
                    a_next = self.get_action(epsilon)

                self.update_Q(s, a, r, s_next, a_next, self.Q, eta, gamma)

                if s_next == 15:
                    break
        else:
            while True:
                self.move_next_state()
                self.state_history[-1][1] = self.action
                self.state_history.append([self.state, np.nan])
                if self.state == 15:
                    print("Complete in %d steps" %
                          (len(self.state_history) - 1))
                    break

    def train(self, stop_epsilon=10**-4):

        if self.strategy == 'rand_walk':
            raise ValueError('train does not support rand_walk strategy')

        elif self.strategy in ['sarsa', 'q']:
            eta = 0.1
            gamma = 0.9
            epsilon = 0.5
            v = np.nanmax(self.Q, axis=1)
            episode = 1
            tot_episode = 100
            V = []
            V.append(v)
            while True:
                print("Episode: ", str(episode))
                epsilon = epsilon / np.log(3)
                self.solve_maze(epsilon=epsilon, eta=eta, gamma=gamma)
                new_v = np.nanmax(self.Q, axis=1)
                print("State value difference: ", np.sum(np.abs(new_v - v)))
                v = new_v
                V.append(v)
                print("Complete in %d steps" % (len(self.state_history) - 1))
                episode += 1
                if episode > tot_episode:
                    print("State value after training: ", V[-1])
                    break

                self.state = 0
                self.state_history = [[0, np.nan]]
        else:

            while True:
                self.solve_maze()
                old_pi = self.pi
                self.update_theta()
                self.initial_covert_from_theta_to_pi()

                print("Complete in %d steps" % (len(self.state_history) - 1))

                if np.sum(np.abs(self.pi - old_pi)) < stop_epsilon:
                    print("Training Complete.")
                    break

                self.state = 0
                self.state_history = [[0, np.nan]]

    def __repr__(self):
        return "Agent: {}".format(self.strategy)
