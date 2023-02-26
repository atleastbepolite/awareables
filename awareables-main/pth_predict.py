import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

def make_prediction(img_path):
    model = CNN()
    model.load_state_dict(torch.load("./models/aeye-relabeled.pth"))
    image = Image.open(img_path)
    image = image.convert('RGB')
    width, height = image.size
    num = round(width/height/0.78)
    w = width/num

    letters = []
    cropped = image.crop((0, 0, w, height))
    cropped = np.array(cropped)
    cropped = cv2.resize(cropped, (28, 28))
    cropped = cropped.astype(np.float32) / 255.0
    cropped = torch.from_numpy(cropped[None, :, :, :])
    cropped = cropped.permute(0, 3, 1, 2)
    predicted_tensor = model(cropped)
    _, predicted_letter = torch.max(predicted_tensor, 1)
    softmax = torch.nn.functional.softmax(predicted_tensor, dim=1)
    top5 = torch.topk(softmax, 5)
    return (int(predicted_letter), float(top5[0][0][0]), [int(i) for i in top5[1][0]])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            # 3x28x28
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            # 16x28x28
            nn.MaxPool2d(kernel_size=2),
            # 16x14x14
            nn.LeakyReLU()
        )
        # 16x14x14
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            # 32x14x14
            nn.MaxPool2d(kernel_size=2),
            # 32x7x7
            nn.LeakyReLU()
        )
        # linearly
        self.block3 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 37)
        )
        # 1x37

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        # flatten the dataset
        out = out.view(-1, 32 * 7 * 7)
        out = self.block3(out)
        
        return out
