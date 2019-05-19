import numpy as np


class Agent:

    def __init__(self):
        self.theta = None
        self.pi = None
        self.state = 0
        self.state_history = [0]

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

    def calc_policy_from_theta(self, theta):
        """Assume uniform distribution for available states"""

        [m, n] = theta.shape
        pi = np.zeros((m, n))
        for i in range(m):
            pi[i, :] = self.theta[i, :] / np.nansum(self.theta[i, :])

        self.pi = np.nan_to_num(pi)

    def move_next_state(self):
        """Returns next states for given policy and state"""
        direction = ["U", "R", "D", "L"]

        next_direction = np.random.choice(direction, p=self.pi[self.state, :])

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

    def solve_maze(self):

        while True:
            self.move_next_state()
            self.state_history.append(self.state)
            if self.state == 15:
                print("Complete in %d steps" % len(self.state_history))
                break
