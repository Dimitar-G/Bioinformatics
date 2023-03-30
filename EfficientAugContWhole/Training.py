from torch import optim, nn
import torch
from torch.utils.data import DataLoader
from Model import EfficientClassifier
from Transforms import transformAugmentNoCrop, transformAugmentValNoCrop
from Dataset import ALLDataset
from tqdm import tqdm
from statistics import mean

if __name__ == '__main__':

    # Training Data
    dataset = ALLDataset(root='../Data/training_data', transform=transformAugmentNoCrop)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    print('Training dataset loaded.')

    # Validation Data
    validation_dataset = ALLDataset(root='../Data/validation_data', transform=transformAugmentValNoCrop)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=2)
    print('Validation dataset loaded.')

    model = EfficientClassifier()
    model.load_state_dict(torch.load('./models/efficient_aug_1910.pt'))
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    num_epochs = 90

    for epoch in range(0, num_epochs):

        loss_epoch = []
        for i, (inputs, labels) in tqdm(enumerate(dataloader), desc=f'Batches in epoch {epoch}'):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss_epoch.append(float(loss.detach().numpy()))

        training_loss = mean(loss_epoch)

        # Evaluate the model on the validation set
        # print('Validating:')
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(validation_dataloader):
                output = model(data)
                val_loss += criterion(output, target.float().unsqueeze(1)).item()
                pred = (output > torch.tensor([0.5])).float()*1
                val_correct += pred.eq(target.view_as(pred)).sum().item()

        # Print the results for the current epoch
        val_loss /= len(validation_dataset)
        val_acc = 100. * val_correct / len(validation_dataset)
        print(f'Validation accuracy: {val_acc}')
        print(f'Validation loss: {val_loss}')

        with open('training_info.csv', 'a') as file:
            file.write(f'{epoch+1910},{training_loss},{val_loss},{val_acc}\n')
            file.flush()

        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), f"./models/efficient_aug_{epoch+1910}.pt")

    torch.save(model.state_dict(), f"./models/efficient_aug_{num_epochs+1910}.pt")
