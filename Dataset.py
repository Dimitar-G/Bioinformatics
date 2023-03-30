from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class ALLDataset(Dataset):
    """
    This class is a custom defined Image Dataset for this project.
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        images_hem = os.listdir(root + '/hem')
        images_all = os.listdir(root + '/all')
        images_hem = map(lambda filename: (root + '/hem/' + filename, 0), images_hem)
        images_all = map(lambda filename: (root + '/all/' + filename, 1), images_all)
        self.images = list(images_hem) + list(images_all)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        item = self.images[index]
        image = Image.open(item[0])

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, item[1]
