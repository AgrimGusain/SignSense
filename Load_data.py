import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class NoiseDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        # Collect (noisy, clean, label) file paths
        for label in sorted(os.listdir(noisy_dir)):
            noisy_folder = os.path.join(noisy_dir, label)
            clean_folder = os.path.join(clean_dir, label)

            if not os.path.isdir(noisy_folder):
                continue

            for file in sorted(os.listdir(noisy_folder)):
                noisy_path = os.path.join(noisy_folder, file)
                clean_path = os.path.join(clean_folder, file)
                if os.path.isfile(noisy_path) and os.path.isfile(clean_path):
                    self.data.append((noisy_path, clean_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy_path, clean_path, label = self.data[idx]
        noisy_image = Image.open(noisy_path).convert("RGB")
        clean_image = Image.open(clean_path).convert("RGB")

        return self.transform(noisy_image), self.transform(clean_image), label
