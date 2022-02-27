import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


def save_train_history(history_dict, file_path):
    temp = pd.DataFrame(history_dict.history)

    try:
        temp.to_pickle(file_path)

    except FileNotFoundError:
        print('file not found.')


def create_metric_graph(history_dict, metric='accuracy', validation=True, file_path=None):
    epochs = list(range(len(history_dict.history[metric])))

    plt.figure(figsize=[12, 6], dpi=300)
    sns.lineplot(x=epochs,
                 y=history_dict.history[metric],
                 marker='o',
                 label='training')

    if validation:
        sns.lineplot(x=epochs,
                     y=history_dict.history[f'val_{metric}'],
                     marker='o',
                     linestyle='--',
                     label='validation')

    plt.xlabel('epoch')
    plt.ylabel(f'{metric}')

    if file_path is not None:
        plt.savefig(file_path)

    plt.show()


def create_loss_graph(history_dict, validation=True, file_path=None):
    epochs = list(range(len(history_dict.history['loss'])))

    plt.figure(figsize=[12, 6], dpi=300)
    sns.lineplot(x=epochs,
                 y=history_dict.history['loss'],
                 marker='o',
                 label='training')

    if validation:
        sns.lineplot(x=epochs,
                     y=history_dict.history['val_loss'],
                     marker='o',
                     linestyle='--',
                     label='validation')

    plt.xlabel('epoch')
    plt.ylabel('loss')

    if file_path is not None:
        plt.savefig(file_path)

    plt.show()


def create_learning_rate(history_dict, file_path=None):
    epochs = list(range(len(history_dict.history['lr'])))
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=[12, 6], dpi=300)

    sns.lineplot(x=epochs,
                 y=history_dict.history['lr'],
                 ax=axes[0])

    sns.scatterplot(x=history_dict.history['loss'],
                    y=history_dict.history['lr'],
                    label='training loss',
                    ax=axes[1])

    sns.scatterplot(x=history_dict.history['val_loss'],
                    y=history_dict.history['lr'],
                    label='validation loss',
                    ax=axes[1])

    axes[0].set_title('learning rate change through training')
    axes[1].set_title('learning rate changes with loss')

    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('learning rate')
    axes[1].set_xlabel('loss')
    axes[1].set_ylabel('learning rate')

    if file_path is not None:
        plt.savefig(file_path)

    plt.show()
