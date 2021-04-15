import glob
import random
import os
import natsort
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_ = None):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root ) + '*.*'))        
        self.files_A = natsort.natsorted(self.files_A)        
    def __getitem__(self, index):
        
        item_A = self.transform(Image.open(self.files_A[index]))        
        return {'A': item_A}
        
    def __len__(self):
        return len(self.files_A)