from matplotlib import pyplot as plt
import pandas as pd

"""
A script intended for generating and saving plots.
"""

if __name__ == '__main__':
    data = pd.read_csv('./EfficientAugContWhole/training_info.csv')
    data.columns = ['epoch', 'training_loss', 'validation_loss', 'validation_accuracy']
    plt.plot(data['epoch'], data['training_loss'], label='Training')
    # plt.plot(data['epoch'], data['validation_loss'], label='Validation')
    # plt.plot(data['epoch'], data['validation_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    # plt.ylabel('Validation Loss')
    # plt.ylabel('Validation Accuracy')
    # plt.show()
    plt.savefig('./EfficientAugContWhole/training_loss.png')
