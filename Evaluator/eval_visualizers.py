import matplotlib.pyplot as plt
import ray
from matplotlib.animation import FuncAnimation


def show_confustion_matrix(cm):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Actuals', fontsize=16)
    plt.title('Confusion Matrix', fontsize=16)
    plt.show()

@ray.remote(num_gpus=0.01, num_cpus=0.5)
class GameResultPlotter:
    def __init__(self):
        # Initialize empty lists to store data for each metric
        self.avg_game_length = {}
        self.furthest_reach = {}
        self.worst_result = {}
        self.games_played = {}

        # Create the subplots
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))

        # Titles for each subplot
        self.axs[0, 0].set_title("Average Game Length")
        self.axs[0, 1].set_title("Furthest Reached")
        self.axs[1, 0].set_title("Worst Result")
        self.axs[1, 1].set_title("Games Played")
        self.num_updates = 0

    def update_data(self, episode, result_data):
        # Append new data for each metric
        self.avg_game_length[episode] = result_data['average_game_length']
        self.furthest_reach[episode] = result_data['furthest_reached']
        self.worst_result[episode] = result_data['worst_result']
        self.games_played[episode] = result_data['games_played']
        self.num_updates += 1
        if self.num_updates == 1000:
            self.start()

    def animate(self, i):
        # Clear each subplot to re-draw
        for ax in self.axs.flat:
            ax.clear()

        episodes = sorted(self.avg_game_length.keys())

        # Plot the updated data
        self.axs[0, 0].plot(episodes, [self.avg_game_length[e] for e in episodes], 'b-', label="Average Game Length")
        self.axs[0, 1].plot(episodes, [self.furthest_reach[e] for e in episodes], 'g-', label="Furthest Reached")
        self.axs[1, 0].plot(episodes, [self.worst_result[e] for e in episodes], 'r-', label="Worst Result")
        self.axs[1, 1].plot(episodes, [self.games_played[e] for e in episodes], 'c-', label="Games Played")

        # Set labels
        for ax in self.axs.flat:
            ax.legend()
            ax.set_xlabel("Episode")
            ax.set_ylabel("Value")

    def start(self):
        # Set up the FuncAnimation to update every 1000 ms (1 second)
        self.ani = FuncAnimation(self.fig, self.animate, interval=1000)
        plt.tight_layout()
        plt.show()

