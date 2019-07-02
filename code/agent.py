import numpy as np


class Agent:

    def __init__(self, strategy="e-greedy"):
        self.strategy = strategy
        self.pi = None
        self.state = 0
        self.action = None
        self.state_history = [[0, np.nan]]
        self.theta = np.array([[np.nan, 1, np.nan, np.nan], # S0
                            [np.nan, 1, np.nan, 1],         # S1
                            [np.nan, np.nan, 1, np.nan],    # S2
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
                            ])  # S15 does not need a policy.
        if strategy in ['sarsa', 'e-greedy']:
            a, b = self.theta.shape
            self.Q = np.random.rand(a, b) * self.theta

        self.initial_covert_from_theta_to_pi()


    def initial_covert_from_theta_to_pi(self):

        [m, n] = self.theta.shape
        pi = np.zeros((m, n))

        if self.strategy in ["rand_walk", "e-greedy"]:
            for i in range(0, m):
                pi[i, :] = self.theta[i, :] / np.nansum(self.theta[i, :])
        elif self.strategy == "pg":
            beta = 1.0
            exp_theta = np.exp(beta * self.theta)
            for i in range(0, m):
                pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
        else:
            raise ValueError("undefined strategy %s" % self.strategy)

        self.pi = np.nan_to_num(pi)


    def move_next_state(self, Q=None, epsilon=None):
        """Returns next states for given policy and state"""
        direction = ["U", "R", "D", "L"]

        if self.strategy in ["e-greedy"]:
            if np.random.rand() < epsilon:
                # Random move
                next_direction = np.random.choice(direction, p=self.pi[self.state, :])
            else:
                # Choose action maximizes Q
                next_direction = direction[np.nanargmax(Q[self.state, :])]
        else:
            next_direction = np.random.choice(direction, p=self.pi[self.state, :])

        if next_direction == "U":
            action = 0
            s_next = self.state - 4
        elif next_direction == "R":
            action = 1
            s_next = self.state + 1
        elif next_direction == "D":
            action = 2
            s_next = self.state + 4
        elif next_direction == "L":
            action = 3
            s_next = self.state - 1
        else:
            raise ValueError("Unknown Direction %s" % next_direction)

        self.state = s_next
        self.action = action

    def update_theta(self):
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

                    delta_theta[i, j] = (sa_ij_tot - self.pi[i, j] * sa_i_tot) / T

        self.theta = self.theta + eta * delta_theta

    def solve_maze(self):

        while True:
            self.move_next_state()
            self.state_history[-1][1] = self.action
            self.state_history.append([self.state, np.nan])
            if self.state == 15:
                print("Complete in %d steps" % (len(self.state_history) - 1))
                break

    def train(self, stop_epsilon=10**-4):

        if self.strategy == 'rand_walk':
            raise ValueError('train does not support rand_walk strategy')

        is_continue = True
        while is_continue:
            self.solve_maze()
            old_pi = self.pi
            self.update_theta()
            self.initial_covert_from_theta_to_pi()

            print("Complete in %d steps" % (len(self.state_history) - 1))

            if np.sum(np.abs(self.pi - old_pi)) < stop_epsilon:
                is_continue = False
                print("Training Complete.")
            else:
                self.state = 0
                self.state_history = [[0, np.nan]]

    def reset(self):
        self.__init__(self.strategy)

    
    def __repr__(self):
        return "Agent: {}".format(self.strategy)
