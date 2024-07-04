import matplotlib.pyplot as plt
import seaborn as sns
import torch

class ActivationVisualizer:
    def __init__(self, model, fig_size=(12, 8)):
        self.model = model
        self.fig, self.axs = plt.subplots(3, 1, figsize=fig_size)
        self.fig.suptitle("Neural Network Activations")
        
        for ax in self.axs:
            ax.set_xticks([])
            ax.set_yticks([])

        self.axs[0].set_title("Input Layer")
        self.axs[1].set_title("Hidden Layer")
        self.axs[2].set_title("Output Layer")

        plt.tight_layout()
        plt.ion()
        self.fig.show()

    def get_activations(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.model.device)
        
        activations = []
        x = state
        for layer in [self.model.fc1, self.model.fc2, self.model.fc3]:
            x = layer(x)
            activations.append(x.detach().cpu().numpy())
            if layer != self.model.fc3:
                x = torch.relu(x)

        return activations

    def update_plot(self, state):
        activations = self.get_activations(state)

        for i, activation in enumerate(activations):
            self.axs[i].clear()
            self.axs[i].set_title(f"{'Input' if i == 0 else 'Hidden' if i == 1 else 'Output'} Layer")
            sns.heatmap(activation, ax=self.axs[i], cmap="viridis", cbar=False)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()