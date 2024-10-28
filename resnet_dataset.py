from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.io import read_image

class ResNetDataset():
    def __init__(self, path, image_list, transform):
        self.path = path
        self.image_list = image_list
        self.transform = transform

        self.data = self.load_data()

    def load_data(self):
        images = []

        for i in tqdm(range(len(self.image_list)), desc='Loading images'):
            image_path = self.path + f'c{self.image_list['classname'][i]}/' + self.image_list['img'][i]
            img = read_image(image_path)
            img = self.transform(img)
            images.append(img)
        return images
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.image_list['classname'][idx]