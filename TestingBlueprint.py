from torch import nn
import torch
from torch.utils.data import DataLoader
from Model import EfficientClassifier
from Transforms import transformCenter
from Dataset import ALLDataset

"""This script serves as a blueprint for testing a model. Needs to be adjusted accordingly."""

if __name__ == '__main__':

    # Validation Data
    validation_dataset = ALLDataset(root='C:\\Users\\Dimitar\\Downloads\\PKG - '
                                         'C-NMC_Leukemia\\C-NMC_Leukemia\\C-NMC_training_data\\fold_1',
                                    transform=transformCenter)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=2)
    print('Validation dataset loaded.')

    num_epochs = 500

    for epoch in [0, 100, 200, 300, 400, 500, 700, 800, 900]:  # range(num_epochs):
        model = EfficientClassifier()
        model.load_state_dict(torch.load(f'efficient_clf_{epoch}.pt'))
        criterion = nn.BCELoss()

        # Evaluate the model on the validation set
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(validation_dataloader):
                output = model(data)
                val_loss += criterion(output, target.float().unsqueeze(1)).item()
                pred = (output > torch.tensor([0.5])).float() * 1
                val_correct += pred.eq(target.view_as(pred)).sum().item()

        # Print the results for the current epoch
        val_loss /= len(validation_dataset)
        val_acc = 100. * val_correct / len(validation_dataset)
        print(f'Validation accuracy: {val_acc}')
        print(f'Validation loss: {val_loss}')

        with open('validation_of_efficient_1000.csv', 'a') as file:
            file.write(f'{epoch},{val_acc},{val_loss}\n')
            file.flush()
