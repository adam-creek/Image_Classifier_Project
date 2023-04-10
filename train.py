# Image Classifier Training Program
# Adam Creek

# Call Program with arguments. 
# e.g. python3 train.py --gpu --save_dir checkpoint.pth --epochs 3 --arch alexnet

# Import Packages
import argparse
import logging
import sys
from functions import *

# Create an Agrument Parser Object 
parser = argparse.ArgumentParser(description='Image Classifer Train.py')

# Add Arguments
parser.add_argument('--data_dir', default="./flowers/", help = 'Main Data Directory')
parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth",help = 'Checkpoint File')
parser.add_argument('--epochs', default=2, type=int, help = 'Number of epochs used in model')
parser.add_argument('--arch', default='vgg16', type=str, choices = ['vgg16', 'alexnet'],help='Model Architecture - either vgg16 (default) or Alexnet')
parser.add_argument('--hidden_units', default=4086, type=int, help = 'Number of hidden units in model')
parser.add_argument('--learn', default=0.001, type=float, help = 'Model Learning Rate, default 0.001')  
parser.add_argument('--dropout', default=0.5, type=float, help = 'Model Drop-out rate, default 0.5')
parser.add_argument('--gpu', default=False, action='store_true',help = 'Device model to be trained on') 

#Parsing Argument
pa_obj = parser.parse_args()

#Logging 
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# Assign parser arguements to variables
data_dir = pa_obj.data_dir
path = pa_obj.save_dir
epochs = pa_obj.epochs
arch = pa_obj.arch
hidden_units = pa_obj.hidden_units
learn_rate = pa_obj.learn
dropout = pa_obj.dropout
gpu = pa_obj.gpu

if gpu:
    device='gpu' 
else:
    device='cpu'

print('\n*** Image Classifier Model - Training Program ***\n')

logging.getLogger().info('Loading Datasets ...')
training_loader, validation_loader, testing_loader, training_dataset = load_data()

logging.getLogger().info('Loading Pre-Trained Model Architecture ...')
model, input_size = classifier_model(arch)

logging.getLogger().info('Build Classifier ...')
model, criterion, optimizer = build_classifier(input_size, device, dropout, learn_rate, hidden_units, model)

logging.getLogger().info('Test Image Classifier Model ...')
test_trained_model(model, criterion, optimizer, epochs, device, training_loader, validation_loader,testing_loader)

logging.getLogger().info('Test Network Accuracy ...')
test_network(model, testing_loader,device)

logging.getLogger().info('Saving Model ...')
save_checkpoint(arch, training_dataset, path, model, optimizer, learn_rate, epochs, hidden_units)

print('\n*** Image Classifier Model Successfully Trained & Model Saved ***\n')
