import numpy as np


class Agent:

    def __init__(self):
        self.theta = None
        self.pi = None
        self.state = 0
        self.action = None
        self.state_history = [[0, np.nan]]

    def init(self):
        theta_0 = np.array([[np.nan, 1, np.nan, np.nan],    # S0
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
        self.theta = theta_0

    def calc_policy_from_theta(self, theta, method="softmax"):
        """Assume uniform distribution for available states"""

        [m, n] = theta.shape
        pi = np.zeros((m, n))

        if method == "uniform":
            for i in range(0, m):
                pi[i, :] = self.theta[i, :] / np.nansum(self.theta[i, :])
        elif method == "softmax":
            beta = 1.0
            exp_theta = np.exp(beta * theta)
            for i in range(0, m):
                pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
        else:
            raise ValueError("undefined method %s" % method)

        self.pi = np.nan_to_num(pi)

    def move_next_state(self):
        """Returns next states for given policy and state"""
        direction = ["U", "R", "D", "L"]

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
                    SA_i = [SA for SA in self.state_history if SA[0] == i]
                    SA_ij = [SA for SA in self.state_history if SA == [i, j]]
                    N_i = len(SA_i)
                    N_ij = len(SA_ij)

                    delta_theta[i, j] = (N_ij - self.pi[i, j] * N_i) / T

        self.theta = self.theta + eta * delta_theta


    def solve_maze(self):
        while True:
            self.move_next_state()
            self.state_history[-1][1] = self.action
            self.state_history.append([self.state, np.nan])
            if self.state == 15:
                print("Complete in %d steps" % (len(self.state_history) - 1))
                break

    def train(self):
        stop_epsilon = 10**-4
        is_continue = True
        while is_continue:
            self.solve_maze()
            old_pi = self.pi
            self.update_theta()
            self.calc_policy_from_theta(self.theta)

            print("Complete in %d steps" % (len(self.state_history) - 1))

            if np.sum(np.abs(self.pi - old_pi)) < stop_epsilon:
                is_continue = False
            else:
                self.state = 0
                self.state_history = [[0, np.nan]]
