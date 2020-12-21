import matplotlib.pyplot as plt

class Metrics:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def plot_losses(self):
        ax = plt.gca()
        ax.set_ylabel("Loss (MSE)")
        ax.set_xlabel("Epochs")
        plt.plot(self.train_losses, label='train loss')
        plt.plot(self.val_losses, label='validation loss')
        plt.legend()
        plt.show()

    def __nonzero__(self):
        return bool(self.train_losses or self.val_losses)
