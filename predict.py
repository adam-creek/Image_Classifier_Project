# Image Classifier Prediction Program
# Adam Creek

# Call Program with arguments. 
# e.g. python3 predict.py flowers/test/11/image_03130.jpg checkpoint.pth --gpu -- top_k 5    --category_names cat_to_name.json
# Alternative Test Image: flowers/test/28/image_05230.jpg

# Import Packages
import argparse
import logging
import sys
import json 
from functions import *

# Create an Agrument Parser Object 
parser = argparse.ArgumentParser(description='Image Classifer Predict.py')

# Add Arguments
parser.add_argument('input', type=str, default="flowers/test/28/image_05230.jpg", help = 'Provide an Image Path')
parser.add_argument('checkpoint', type=str, default='checkpoint.pth', help = 'Trained Model Checkpoint File')
parser.add_argument('--top_k', default=3, type=int, help = 'Return k most likley classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json',help = 'Mapping of category names to names')
parser.add_argument('--gpu', default=False, action='store_true',help = 'Device model to be trained on') 

#Parsing Argument
pa_obj = parser.parse_args()

#Logging 
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# Assign parser arguements to variables
image_path = pa_obj.input
checkpoint = pa_obj.checkpoint
top_k = pa_obj.top_k
category_names = pa_obj.category_names
gpu = pa_obj.gpu
path = 'checkpoint.pth' 
i=0 

if gpu:
    device='gpu' 
else:
    device='cpu'
    
    
print('\n*** Image Classifier Model - Prediction Program ***\n')
    
logging.getLogger().info('Loading Datasets ...')
training_loader, validation_loader, testing_loader, training_dataset = load_data()

logging.getLogger().info('Loading Label Mapping File ...')
cat_to_name = load_cat_names(category_names)

logging.getLogger().info('Loading Pre-Trained Model ...')
model = load_checkpoint(path)

logging.getLogger().info('Image Processing ...')
top_probs, top_classes = predict(image_path, model, device, top_k)

logging.getLogger().info('Class Prediction ...')

labels = [cat_to_name[str(index)] for index in top_classes]

print('\nFile selected: ' + image_path)

while i < len(labels):
    print("Flower Class Prediction {}: {} with a probability of {}%".format(i+1,labels[i], round(top_probs[i]*100,2)))
    i += 1

print('\n*** Image Classifier Prediction Complete ***\n')  