import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML


class Maze:

    def __init__(self):
        self.fig = None
        self.ax = None
        self.line = None

    def draw_maze(self):

        self.fig = plt.figure(figsize=(5, 5))
        self.ax = plt.gca()

        # Draw walls
        kwargs = {'color': 'black', 'linewidth': 3}
        plt.plot([1, 1], [1, 2], **kwargs)
        plt.plot([1, 0], [2, 2], **kwargs)
        plt.plot([0, 2], [3, 3], **kwargs)
        plt.plot([2, 2], [1, 2], **kwargs)
        plt.plot([2, 4], [2, 2], **kwargs)
        plt.plot([3, 3], [0, 1], **kwargs)
        plt.plot([3, 3], [3, 4], **kwargs)

        # Draw states of cells
        cell_num = 0
        cell_pos = [0.5, 1.5, 2.5, 3.5]
        for y in reversed(cell_pos):
            for x in cell_pos:
                plt.text(x, y, 'S' + str(cell_num), size=14, ha='center')
                cell_num += 1
        plt.text(0.5, 3.3, 'START', ha='center')
        plt.text(3.5, 0.3, 'GOAL', ha='center')

        self.ax.set_xlim(0, 4)
        self.ax.set_ylim(0, 4)
        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False, labelleft=False)

        self.line, = self.ax.plot([0.5], [3.5], marker='o', color='g', markersize=60)

    def save_animation_html(self, file_name, state_history):

        def init():
            self.line.set_data([], [])
            return self.line,

        def animate(i):
            state = state_history[i]
            x = (state % 4) + 0.5
            y = 3.5 - int(state / 4)
            self.line.set_data(x, y)
            return self.line,

        anim = animation.FuncAnimation(self.fig, animate, init_func=init, frames=len(state_history), repeat=False)
        html = HTML(anim.to_jshtml()).data

        with open(file_name, 'w') as f:
            f.write(html)
            print("Animation is saved at %s." % file_name)
