from email.mime import image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image # from skimage import io, transform
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
import sys
from BT19ECE078_mini_project import Edge_dataset, transform_input, transform_output, device, batch_size, list_images, Net
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
def test(image_path): 
    # Load the model that we saved at the end of the training loop 
    net = Net()
    PATH = './ann_model.pth'
    net.load_state_dict(torch.load(PATH))
     
    running_accuracy = 0 
    total = 0 
 
    with torch.no_grad():  
        inputs  = Image.open(image_path) 
        inputs = transform_input(inputs)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        net.to(device)
        inputs.to(device)
        inputs.reshape(1, 3, 100, 100)
        predicted_outputs = net(inputs) 
        output = predicted_outputs.unsqueeze(0).cpu().numpy()
        mask =  output>0.5
        plt.imshow(mask.reshape(100,100))
        plt.show()
        # print('Accuracy of the model based on the test set of', test_split ,'inputs is: %d %%' % (100 * running_accuracy / total))    
        return mask

test("/users/navya/ML lab/dataset/images/img1.png")