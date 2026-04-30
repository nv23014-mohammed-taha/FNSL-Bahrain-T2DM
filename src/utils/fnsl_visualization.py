import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

class FNSLVisualizer:
    """
    Provides visualization utilities for the Federated Neuro-Symbolic Learning (FNSL) project.
    Generates plots for global model performance, client-specific metrics, and other insights.
    """
    def __init__(self, results_dir="results/fnsl_visualizations/"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"FNSL Visualizer initialized. Results will be saved to {self.results_dir}")

    def plot_global_auc_history(self, auc_history, filename="global_auc_history.png"):
        """
        Plots the global AUC over federated rounds.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(auc_history, marker=".", linestyle="-", color="blue")
        plt.title("Global Model AUC Over Federated Rounds")
        plt.xlabel("Federated Round")
        plt.ylabel("AUC")
        plt.grid(True)
        plt.ylim(0.5, 1.0) # AUC typically ranges from 0.5 to 1.0
        filename = os.path.join(self.results_dir, filename)
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_client_loss_history(self, client_losses_per_round, filename="client_loss_history.png"):
        """
        Plots the average local loss for each client over federated rounds.
        client_losses_per_round: A dictionary where keys are client_ids and values are lists of losses.
        """
        plt.figure(figsize=(12, 7))
        for client_id, losses in client_losses_per_round.items():
            plt.plot(losses, label=client_id, marker=".", alpha=0.7)
        plt.title("Client Local Loss Over Federated Rounds")
        plt.xlabel("Federated Round")
        plt.ylabel("Average Local Loss")
        plt.grid(True)
        plt.legend()
        filename = os.path.join(self.results_dir, filename)
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_data_heterogeneity_impact(self, client_ids, client_biases, client_accuracies_final, filename="data_heterogeneity_impact.png"):
        """
        Conceptual plot to show the impact of data heterogeneity on client performance.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(client_ids, client_accuracies_final, color=sns.color_palette("viridis", len(client_ids)))
        plt.title("Final Client Accuracy vs. Data Characteristics (Conceptual)")
        plt.xlabel("Client ID")
        plt.ylabel("Final Local Accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis=\
y", grid=True)
        filename = os.path.join(self.results_dir, filename)
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_combined_metrics(self, global_auc_history, client_losses_per_round, filename="combined_metrics.png"):
        """
        Plots global AUC and average client loss on a single figure.
        """
        fig, ax1 = plt.subplots(figsize=(12, 7))

        color = 'tab:red'
        ax1.set_xlabel('Federated Round')
        ax1.set_ylabel('Global AUC', color=color)
        ax1.plot(global_auc_history, color=color, marker='.', linestyle='-')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0.5, 1.0)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Average Client Loss', color=color)  # we already handled the x-label with ax1
        
        # Calculate average client loss per round
        avg_losses = [np.mean(list(losses.values())) if losses else 0 for losses in client_losses_per_round.values()]
        ax2.plot(avg_losses, color=color, marker='x', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title("Global AUC and Average Client Loss Over Federated Rounds")
        filename = os.path.join(self.results_dir, filename)
        plt.savefig(filename)
        plt.close()
        return filename
