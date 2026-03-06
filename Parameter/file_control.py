import os,shutil
import matplotlib.pyplot as plt
def log_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
#このContinurous_Signをlog_dirにコピーする
def copy_dir(project_dir,log_dir):
    if os.path.exists(log_dir+"/Continurous_Sign"):
        shutil.rmtree(log_dir+"/Continurous_Sign")
    shutil.copytree(project_dir,log_dir+"/Continurous_Sign")

def plot_accuracy_and_loss(data, save_path,legend_fontsize=12, axis_label_fontsize=14):
    """
    Plots the training and validation accuracy and loss graphs from the provided data.

    Parameters:
    data (DataFrame): Pandas DataFrame containing 'train_acc', 'val_acc', 'train_loss', and 'val_loss' columns.
    save_path (str): Path to save the graphs.
    legend_fontsize (int): Font size for the legend.
    axis_label_fontsize (int): Font size for the axis labels.
    """
    plot_accuracy(data, save_path,legend_fontsize, axis_label_fontsize)
    plot_loss(data, save_path,legend_fontsize, axis_label_fontsize)
def plot_accuracy(data, save_path,legend_fontsize=12, axis_label_fontsize=14):
    """
    Plots the training and validation accuracy graphs from the provided data.

    Parameters:
    data (DataFrame): Pandas DataFrame containing 'train_acc', 'val_acc' columns.
    save_path (str): Path to save the graphs.
    legend_fontsize (int): Font size for the legend.
    axis_label_fontsize (int): Font size for the axis labels.
    """
    # Plotting the accuracy graph
    plt.figure(figsize=(12, 6))
    epochs = range(0, len(data))
    plt.plot(data['train_acc'], label='Training Accuracy', color='blue')
    plt.plot(data['val_acc'], label='Validation Accuracy', color='red')
    plt.xlabel('Epochs', fontsize=axis_label_fontsize)
    plt.ylabel('Accuracy', fontsize=axis_label_fontsize)
    plt.xticks(epochs)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(False)
    plt.savefig(save_path+"/accuracy.png")

def plot_loss(data, save_path,legend_fontsize=12, axis_label_fontsize=14):
    """
    Plots the training and validation loss graphs from the provided data.

    Parameters:
    data (DataFrame): Pandas DataFrame containing 'train_loss', 'val_loss' columns.
    save_path (str): Path to save the graphs.
    legend_fontsize (int): Font size for the legend.
    axis_label_fontsize (int): Font size for the axis labels.
    """
    # Plotting the loss graph
    plt.figure(figsize=(12, 6))
    epochs = range(0, len(data))
    plt.plot(data['train_loss'], label='Training Loss', color='blue')
    plt.plot(data['val_loss'], label='Validation Loss', color='red')
    plt.plot(data['test_loss'], label='Test Loss', color='green')
    plt.xlabel('Epochs', fontsize=axis_label_fontsize)
    plt.ylabel('Loss', fontsize=axis_label_fontsize)
    plt.xticks(epochs)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(False)
    plt.savefig(save_path+"/loss.png")