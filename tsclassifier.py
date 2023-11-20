import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

import urllib.request
import shutil
from datetime import datetime
import pickle


print("Ich lade das Modul tsclassifier ein!")


imgtransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


def download_and_extract_dataset():    
    fname = "gtsrb.zip"
    URL = "http://www.juergenbrauer.org/datasets/" + fname
    urllib.request.urlretrieve(URL, fname)

    shutil.unpack_archive(fname, ".")


def get_data_loaders():

    folder = "/home/juebrauer/link_to_vcd/07_datasets/36_gtsrb/gtsrb/"
    train_dataset = datasets.ImageFolder(root=f"{folder}/train", transform=imgtransform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32,
                                               shuffle=True)
    test_dataset = datasets.ImageFolder(root=f"{folder}/test_subfolders", transform=imgtransform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=32,
                                              shuffle=False)
    return train_loader, test_loader


class CNN4TSRecognition(nn.Module):

    def __init__(self):
        super(CNN4TSRecognition, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()

        with torch.no_grad():
            self.build_classifier(torch.rand(1, 3, 224, 224))
        

    def build_classifier(self, x):

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        
        input_dim = x.size(1)
        output_dim = 43
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        # Feature hierarchy
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)

        # Classifier
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        # Output-Tensor zurückliefern
        return x


def train_model(model, device, num_epochs, train_loader, test_loader):
   
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    accs = []
    for epoch in range(1, num_epochs+1):
    
        start = datetime.now()
    
        print(f"Trainingsepoche #{epoch} / {num_epochs} startet.")
    
        batch_nr = 1
        for images, labels in train_loader:
    
            images = images.to(device)
            labels = labels.to(device)
    
            if batch_nr % 250 == 0:
                print(f"\tBatch #{batch_nr} / {len(train_loader)}")
    
            outputs = model(images)
    
            loss = criterion(outputs, labels)
    
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_nr += 1
    
        stop = datetime.now()

        # Modell speichern
        datei = open(f"models/model_{epoch:03}.pkl", "wb")
        pickle.dump(model, datei)
        datei.close()

        acc = test_model(model, test_loader, device)
        accs.append( acc )
  
        print(f"Trainingsepoche #{epoch} / {num_epochs}:")
        print(f"\t Benötigte Dauer: {stop-start}")
        print(f"\t acc={acc:.2f} %")

        import matplotlib.pyplot as plt
        plt.plot(accs)
        plt.xlabel("Epoche")
        plt.ylabel("Genauigkeit auf Testdaten [%]")
        plt.title("Lernkurve")
        plt.show()

    return accs


def test_model(model, test_loader, device):

    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.shape[0]

    accuracy = (correct/total)*100.0

    return accuracy
  



