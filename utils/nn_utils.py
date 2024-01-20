import numpy as np
import pandas as pd
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import re
import fcntl
# from .other_utils import dark_figure


def class_counter(y, num_classes=None):
    num_classes = len(set(y)) if num_classes is None else num_classes
    return {cl: np.sum(y == cl) for cl in range(num_classes)}

class FileAccessManager:
    def __init__(self, file_path):
        self.file_path = file_path

    def __enter__(self):
        self.file = open(self.file_path, 'a')
        fcntl.flock(self.file, fcntl.LOCK_EX)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()

def get_non_available_numbers(log_file):
    try:
        non_availables_txt = set()
        if os.path.exists(log_file):
            with FileAccessManager(log_file) as file:
                non_availables_txt = set(np.loadtxt(file, dtype=int))
        return non_availables_txt
    except Exception as e:
        print(f"Error when reading the file: {e}")
        return set()

def save_non_available_numbers(log_file, non_availables):
    try:
        with FileAccessManager(log_file) as file:
            file.write('\n' + '\n'.join(map(str, non_availables)))
    except Exception as e:
        print(f"Error when writing to the file: {e}") # hope for the best

def get_weight_path(weights_dir, acc_info=False, new_number='random'):
    np.random.seed(None)
    files = [file for file in os.listdir(weights_dir) if file.startswith("weights")]
    log_file = os.path.join(weights_dir, "non_available_numbers.txt")

    non_available_numbers = []
    if len(files) > 0:
        non_available_numbers = [int(file.split(".")[0].split("_")[1]) for file in files]

    if os.path.exists(log_file):
        non_availables_txt = get_non_available_numbers(log_file)
        non_available_numbers = list(set(non_available_numbers) | non_availables_txt)

    non_available_max = max(non_available_numbers) if non_available_numbers else 0

    if new_number == 'random':
        candidate = np.random.randint(1, int(1e16))
        while candidate in non_available_numbers:
            candidate = np.random.randint(1, int(1e16))
    elif new_number == 'max':
        candidate = non_available_max + 1
    else:
        raise ValueError("new_number must be 'random' or 'max'")

    save_non_available_numbers(log_file, [candidate])
    weight_path = f"weights_{candidate}.hdf5"

    if acc_info:
        weight_path = weight_path.replace(".", ".{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.")

    return os.path.join(weights_dir, weight_path)

# @dark_figure
def training_visualization(history, acc_metric='sparse_categorical_accuracy'):
    acc = history.history[acc_metric]
    val_acc = history.history['val_{}'.format(acc_metric)]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.figure()
    plt.plot(epochs, acc, 'r', label='Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    # Plot training and validation loss per epoch
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    return

def training_history_from_stdin(path, metrics=[" loss", " sparse_categorical_accuracy", "val_loss", "val_sparse_categorical_accuracy"]):
    d = defaultdict(list)
    get_number = lambda s, line: float(".".join(list(re.findall(f"{s}: (\d+).(\d+)", line)[0])))

    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if all([metric in line for metric in metrics]):
                for metric in metrics:
                    d[metric].append(get_number(metric, line))
    df = pd.DataFrame(d)
    df.columns = [c.replace(" ", "") for c in df.columns]
    return df
