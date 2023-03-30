from torch import optim, nn
import torch
from torch.utils.data import DataLoader
from Model import EfficientClassifier
from Transforms import transformCenter
from Dataset import ALLDataset
from tqdm import tqdm
from statistics import mean

"""This script serves as a blueprint for training a model. Needs to be adjusted accordingly.
Different testing scripts are available for every model."""

if __name__ == '__main__':

    # Training Data
    dataset = ALLDataset(root='C:\\Users\\Dimitar\\Downloads\\PKG - '
                              'C-NMC_Leukemia\\C-NMC_Leukemia\\C-NMC_training_data\\fold_0',
                         transform=transformCenter)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    print('Training dataset loaded.')

    # Validation Data
    validation_dataset = ALLDataset(root='C:\\Users\\Dimitar\\Downloads\\PKG - '
                                         'C-NMC_Leukemia\\C-NMC_Leukemia\\C-NMC_training_data\\fold_0',
                                    transform=transformCenter)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=2)
    print('Validation dataset loaded.')

    model = EfficientClassifier()
    model.load_state_dict(torch.load('efficient_clf_500.pt'))
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.000001)

    num_epochs = 200

    for epoch in range(num_epochs):

        loss_epoch = []
        for i, (inputs, labels) in tqdm(enumerate(dataloader), desc=f'Batches in epoch {epoch}'):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss_epoch.append(loss.numpy()[0])
            loss.backward()
            optimizer.step()

        training_loss = mean(loss_epoch)

        # Evaluate the model on the validation set
        # print('Validating:')
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

        with open('training_info.csv', 'a') as file:
            file.write(f'{epoch},{training_loss},{val_loss},{val_acc}\n')
            file.flush()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./models/efficient_clf_{epoch}.pt")

    torch.save(model.state_dict(), f"efficient_{num_epochs}.pt")
