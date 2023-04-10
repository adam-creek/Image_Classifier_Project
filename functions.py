# Image Classifier Program - Functions File
# Adam Creek

# Import Packages

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import json
import numpy as np
from PIL import Image
import time
from collections import OrderedDict

def load_data():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomHorizontalFlip(),  
                                     transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize((255,255)),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize((255,255)),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms) 

    # Using the image datasets and the trainforms, define the dataloaders
    training_loader = torch.utils.data.DataLoader(training_dataset,batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset,batch_size=64,shuffle=False)
    testing_loader = torch.utils.data.DataLoader(testing_dataset,batch_size=64,shuffle=False)
    
    return training_loader, validation_loader, testing_loader, training_dataset


def classifier_model(arch):
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_size = 25088
    else:
        model = models.alexnet(pretrained = True)
        input_size = 9216
    return model, input_size


def build_classifier(input_size, device, dropout, learn_rate, hidden_units, model):
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier=nn.Sequential(OrderedDict([
    ('fc1',nn.Linear(input_size,hidden_units,bias=True)),
    ('relu',nn.ReLU()),
    ('dropout',nn.Dropout(p=dropout)),
    ('fc2',nn.Linear(hidden_units,102,bias=True)),
    ('output',nn.LogSoftmax(dim=1))]))  
    
    # Put the classifier on the pretrained network
    model.classifier = classifier
    
    # Train a model with a pre-trained network
    criterion = nn.NLLLoss()

    # Optimizers require the parameters to optimize and a learning rate [Adam (Adaptive Momemt Estimation)]
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
     
    to_device = torch.device('cuda' if torch.cuda.is_available() and device=='gpu' else 'cpu')    
    model.to(to_device)
    
    return  model, criterion, optimizer
    
    
def test_trained_model(model, criterion, optimizer, epochs, device, training_loader, validation_loader, testing_loader):    
    steps=0
    batch=10
    
    to_device = torch.device('cuda' if torch.cuda.is_available() and device=='gpu' else 'cpu')   
    
    for epoch in range(epochs):
        running_loss=0
        model.train()
    
        for inputs,labels in training_loader:
            steps+=1
            inputs, labels = inputs.to(to_device), labels.to(to_device)
            output=model.forward(inputs)
            loss=criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            running_loss+=loss.item()
        
            if steps%batch==0:
                valid_loss = 0
                accuracy = 0
                model.eval()
            
                with torch.no_grad():
                    for inputs, labels in validation_loader:
                            inputs, labels = inputs.to(to_device), labels.to(to_device)
                            output = model.forward(inputs)
                            loss = criterion(output, labels)
                            valid_loss += loss.item()
                        
                            # Calculate accuracy
                            ps = torch.exp(output)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor))
                
                print("Epoch: {}/{}".format(epoch+1, epochs),end=" |  ")
                print("Training Loss: {:.3f}".format(running_loss/steps), end=" |  ")
                print("Validation Loss: {:.3f}".format(valid_loss/len(testing_loader)), end=" |  ")
                print("Validation Accuracy: {:.3f}".format(accuracy.item()/len(testing_loader)))
                running_loss = 0
                model.train()
                

def test_network(model, testing_loader, device):
    total_correct = 0
    
    to_device = torch.device('cuda' if torch.cuda.is_available() and device=='gpu' else 'cpu')   
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in testing_loader:
            inputs, labels = inputs.to(to_device), labels.to(to_device)
            output = model.forward(inputs)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            total_correct += torch.mean(equals.type(torch.FloatTensor))
    print("Test accuracy: {:.3f}%".format(total_correct.item()/len(testing_loader)))

    
def save_checkpoint(arch, training_dataset, path, model, optimizer, learn_rate, epochs, hidden_units):
    model.class_to_idx=training_dataset.class_to_idx
    checkpoint = {'arch': arch, 
                  'model': model,
                  'learn_rate': learn_rate,
                  'hidden_units': hidden_units,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, path) 
    print("Model saved to {}".format(path))
    
    
def load_checkpoint(path):
    checkpoint = torch.load(path)
    arch = checkpoint['arch']
    model = checkpoint['model']
    learn_rate = checkpoint['learn_rate']
    hidden_units = checkpoint['hidden_units']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    print("*** Model successfully loaded from {} ***".format(path))
    return model

def process_image(image_path):

    pil_img = Image.open(image_path)
    img_preprocess = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224)])
    img_tensor = img_preprocess(pil_img)
    np_image = np.array(img_tensor)
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image


def predict(image_path, model, device, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    to_device = torch.device('cuda' if torch.cuda.is_available() and device=='gpu' else 'cpu')
    model.to(to_device)
    model.eval()
    
    img_tensor = process_image(image_path)
    img_tensor = torch.from_numpy(img_tensor)
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.float()
    
    if device == 'gpu':
        with torch.no_grad():
            output = model(img_tensor.cuda())
            ps = torch.exp(output)
            top_ps, top_class_idx = ps.topk(top_k, dim=1)
            
    else:
        with torch.no_grad():
            output=model(img_tensor)
            ps = torch.exp(output)
            top_ps, top_class_idx = ps.topk(top_k, dim=1)
                                       
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    top_classes = [class_to_idx_inverted[i] for i in top_class_idx.cpu().numpy()[0]]
    top_probs = top_ps.cpu().numpy()[0]
    
    return top_probs, top_classes

def load_cat_names(category_names):
    with open(category_names) as f:
        cat_to_names = json.load(f)
    return cat_to_names