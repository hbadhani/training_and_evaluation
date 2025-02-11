import matplotlib.pyplot as plt
import os

def plot_losses(train_losses, val_losses, save_dir, filename='training_progress.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path)
    plt.close()
