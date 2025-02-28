import cProfile
import os
from random import Random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
from neural_network import Neural_network
from model_handler import Model_handler
from torch.utils.tensorboard import SummaryWriter

class Agent():
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        self.writer = SummaryWriter(log_dir='logs')
        self.input_size = 784 #28x28
        self.hidden_size = 200
        self.num_classes = 62
        self.batch_size = 64
        self.model = Neural_network().to(self.device)
        self.model_path = "./model/model.pt"
        self.training_transform = transforms.Compose([
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True),
                ])
        self.test_transform = transforms.Compose([
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True),
                ])
        
        self.train_dataset = torchvision.datasets.EMNIST(root="./data", train=True, split="byclass",
                                                        transform=self.training_transform, download=True)
        
        self.test_dataset = torchvision.datasets.EMNIST(root="./data", train=False, split="byclass",
                                                        transform=self.test_transform)
        numbers = "abcdefghijklmnopqrstuvwxyz"
        numbers_caps = numbers.upper()
        numbers_list = [*numbers]
        numbers_caps_list = [*numbers_caps]
        self.dataset_classes = []
        for i in range(10):
            self.dataset_classes.append(i)
        self.dataset_classes += numbers_caps_list
        self.dataset_classes += numbers_list

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)
        if os.path.exists(self.model_path):
            Model_handler.load(self.model, self.model_path)

            if False:
                mislabeled_images = []
                true_labels = []
                predicted_labels = []
                with torch.no_grad():
                    n_correct = 0
                    n_samples = 0
                    for images_cpu, labels_cpu in self.test_loader:
                        images = images_cpu.to(self.device)
                        labels_gpu = labels_cpu.to(self.device)
                        outputs = self.model(images)
                        
                        _, predictions = torch.max(outputs, 1)
                        n_samples += labels_gpu.shape[0]
                        n_correct += (predictions == labels_gpu).sum().item()
                        for i in range(len(predictions)):
                            if predictions[i] != labels_gpu[i]:
                                mislabeled_images.append(images_cpu[i])
                                true_labels.append(self.dataset_classes[labels_gpu[i].item()])
                                predicted_labels.append(self.dataset_classes[predictions[i].item()])
                        if n_samples > 10000:
                            break
              
                acc = 100 * n_correct / n_samples
                print(f"accuracy: {acc}%")
                num_images = len(mislabeled_images)
                if num_images == 0:
                    print("No mislabeled images to display.")
                    return
                max_images_per_fig = 20
                num_cols = 5
                images_per_fig = min(max_images_per_fig, num_images)
                num_rows = (images_per_fig + num_cols - 1) // num_cols
                
                # Display the mislabeled images in multiple figures if necessary
                for start_idx in range(0, num_images, images_per_fig):
                    end_idx = min(start_idx + images_per_fig, num_images)
                    fig = plt.figure(figsize=(15, 3 * num_rows))
                    for i in range(start_idx, end_idx):
                        plt.subplot(num_rows, num_cols, i - start_idx + 1)
                        self.imshow(mislabeled_images[i])
                        plt.title(f'True: {true_labels[i]}\nPred: {predicted_labels[i]}')
                        plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                    break # to only show one plot
                
    def start_training(self):
        self.num_epochs = 1
        self.learning_rate = 0.001

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.examples = iter(self.train_loader)
        self.samples, self.labels = next(self.examples)
        #plotting transformed data
        #for i in range(10):
            #plt.subplot(2, 5, i+1)
            #plt.imshow(self.samples[i][0], cmap="gray")
        #plt.show()
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.n_total_steps = len(self.train_loader)
        running_loss = 0
        running_correct_pred = 0
        global_step = 0
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.outputs = self.model(images)
                self.loss = self.criterion(self.outputs, labels)
                
                self.optimizer.zero_grad(set_to_none=True)
                self.loss.backward()
                self.optimizer.step()
                
                _, predictions = torch.max(self.outputs, 1)
                
                running_loss += self.loss.item()
                running_correct_pred += (predictions==labels).sum().item()
                if i % 100 == 0:
                    self.writer.add_scalar('Loss', self.loss.item(), global_step=global_step)
                    print(f"training... epoch: {epoch+1}/{self.num_epochs}, Step {i}/{self.n_total_steps}, loss: {self.loss.item():.4f}")
                    self.writer.add_scalar("training loss", running_loss/100, global_step)
                    self.writer.add_scalar("accuracy", running_correct_pred/100, global_step)
                    running_correct_pred = 0
                    running_loss = 0.0
                global_step += 1
                
            #print(f"done!, loss: {self.loss.item():.4f}")
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for images, labels in self.test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    
                    _, predictions = torch.max(outputs, 1)
                    n_samples += labels.shape[0]
                    n_correct += (predictions == labels).sum().item()
                    
                acc = 100 * n_correct / n_samples
                print(f"accuracy: {acc}%")
            if acc > 99.5:
                break
        Model_handler.save(self.model, self.model_path)
        agent.writer.close()

        
    def imshow(self, img, title=None):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        if title:
            plt.title(title)
        
class AddNoise(object):
    def __init__(self, mean=0, std=1,  device="cpu"):
        self.mean = mean
        self.std = std
        self.device = device
        
    def __call__(self, tensor):
        rand_tensor = torch.randn(tensor.size(), device=self.device)
        rand_tensor = rand_tensor * self.std + self.mean
        total = tensor + rand_tensor
        #total = torch.softmax(total, 1, torch.float32)
        return total
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

if __name__ == "__main__":
    agent = Agent()
    #if not os.path.exists(agent.model_path):
    agent.start_training()
    