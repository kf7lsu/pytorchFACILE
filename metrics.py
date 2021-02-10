import matplotlib.pyplot as plt

class Metrics:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_losses_quant = []
        self.val_losses_quant = []

    def plot_losses(self):
        ax = plt.gca()
        ax.set_ylabel("Loss (MSE)")
        ax.set_xlabel("Epochs")
        plt.plot(self.train_losses, label='train loss')
        plt.plot(self.val_losses, label='validation loss')
        plt.plot(self.train_losses_quant, label='quant train loss')
        plt.plot(self.val_losses_quant, label='quant validation loss')
        plt.legend()
        plt.show()

    def __nonzero__(self):
        return bool(self.train_losses or self.val_losses)
