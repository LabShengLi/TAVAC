# VTransformerHer2StTumorPred_hugface.py

"""
This script trains a Vision Transformer (ViT) model for tumor classification using H&E (Hematoxylin and Eosin) images.
The model leverages the Hugging Face library for implementation.

Detailed Steps:
1. Load the Her2 H&E image and expression data.
2. Preprocess the data and split it into training and testing sets.
3. Define and configure the ViT model using the Hugging Face library.
4. Train the ViT model on the training dataset.
5. Evaluate the model performance on the test dataset.
6. Save and log the training results.

Environment:
- Requires a conda environment: torch_env

Imports:
- Libraries for data handling: os, numpy, pandas
- PyTorch libraries for model training and evaluation: torch, torchvision, torch.nn, torch.optim
- Hugging Face libraries for Vision Transformer: transformers
- Custom data handling scripts: DataHer2ST_TumorClassification, VTransformerLib_torch

Arguments:
- --patient_id: 0 for stage 1 while 1 for stage 2.
"""


import os
import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os, sys
import copy

sys.path.insert(1, '/projects/li-lab/Yue/SpatialAnalysis/py') ##~wont work, has to start with /Users

import ensembl #from st-net
import pandas as pd
from datasets import Dataset, Image

import DataHer2ST_TumorClassification as DataObj
#import DataSTNet_test as DataObj
torch.manual_seed(1024)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--patient_id', help='patient for testing')

args = parser.parse_args()

if args.patient_id:
    patient_id = int(args.patient_id)
    print('patient: ')
    print(patient_id)
else:
    patient_id = 0
    print('using default number of patient: 0')

logger_dic = {}

import VTransformerLib_torch as MyVit

logger_dic['cancer_type'] = DataObj.cancer_type

MyVit.if_froze_vit = False
#logger_dic['if_froze_vit'] = MyVit.if_froze_vit

MyVit.num_class = 2
MyVit.learning_rate = 0.00001 ################
logger_dic['learning_rate'] = MyVit.learning_rate 

MyVit.weight_decay = 0.001
logger_dic['weight_decay'] = MyVit.weight_decay 

MyVit.batch_size = 10 ################
logger_dic['batch_size'] = MyVit.batch_size 

MyVit.num_epochs = 500 ################ 30
logger_dic['num_epochs'] = MyVit.num_epochs

###2 fold cv
patients = ['Stage1', 'Stage2']

tst_patient = patients[patient_id] ##'BC23810' #patient with largest number of voxels
logger_dic['test_patient'] = tst_patient

#2-fold cross validation
np.random.seed(0)
random_indx = np.random.choice(len(DataObj.X), len(DataObj.X)//2)

train_image_url = np.array(DataObj.X)[random_indx]
test_image_url = np.delete(np.array(DataObj.X), random_indx)

train_labels = np.array(DataObj.Y_filtered)[random_indx]
test_labels = np.delete(np.array(DataObj.Y_filtered),random_indx)

if tst_patient == 'Stage2':

    train_image_url = np.delete(np.array(DataObj.X), random_indx) 
    test_image_url = np.array(DataObj.X)[random_indx]

    train_labels = np.delete(np.array(DataObj.Y_filtered),random_indx)
    test_labels = np.array(DataObj.Y_filtered)[random_indx]


from datasets import Dataset

train_ds = Dataset.from_dict({"img": train_image_url, "label":train_labels.astype(int)}).cast_column("img", Image())
val_ds = Dataset.from_dict({"img": test_image_url, "label":test_labels.astype(int)}).cast_column("img", Image())

print(train_ds)
print(val_ds)

id2label = {
            0: 'non',
            1: DataObj.cancer_type
            }
# id2label is dictionnary. Enumerate fuction is used to iterate over the names of the class labels 
# in the 'label' feature of the'train_ds' dataset. For each label, the corresponding ID(index) is
# assigned as a key, and the labe itself is assiged as the value.

label2id = {label:id for id,label in id2label.items()}
# reverses the order of each key-value pair in the dictionary 

print(id2label)


# In[8]:




from transformers import ViTImageProcessor
# the line imports the ViTImageProcessor from the transformers library. 
# The transformers library provides state-of-the-art pretrained models and utilities
# for natural language processing and computer vision tasks 

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")




# For data augmentation, one can use any available library. Here we'll use torchvision's [transforms module](https://pytorch.org/vision/stable/transforms.html).

# In[10]:


from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples


# In[11]:


# Set the transforms
train_ds.set_transform(train_transforms)

val_ds.set_transform(val_transforms)


from torch.utils.data import DataLoader
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# This function is used by the 'Dataloader' to collate or combine individual 
# examples into a batch. It takes a list of examples as input, where each example is a 
# dictionary containinng 'pixel_values' and 'label'

train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)
# The DataLoader is created with the train_ds dataset as the first argument.
# It takes the collate_fn as the collate_fn parameter, 
# specifying how to collate the examples into a batch. 
# The batch_size is set to 4, 
# indicating that the data loader will yield batches of size 4 during training.


batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k, v.shape)
    


from transformers import ViTForImageClassification
# This line imports the 'ViTForImageClassification' class from the transformers library.
# This class is specifically designed for Vision Transformers and provides functionalities for image 
# classification tasks

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  id2label=id2label,
                                                  label2id=label2id)







#10 epoch good fitting archive

from transformers import TrainingArguments, Trainer
# These lines import the necessary classes, 'TrainingArguments' and 'Trainer',
# from the 'transformers' library. 
# These classes provide functionalities from configuring and executing the 
# training process for machine learning models.

metric_name = "f1"

# variable is set 'accuracy'
# This variable represents the name of the metric that will be used
# to evaluate performance of the model during training and determine 
# the best model based on this metric. 

data_set = 'Her2st'
args = TrainingArguments(
    f'{data_set}_checkpoints',
    save_strategy="no",
    evaluation_strategy="no",
    learning_rate=MyVit.learning_rate,
#     2e-5
    #0.0001
    per_device_train_batch_size=MyVit.batch_size,
#     10
    #32
    per_device_eval_batch_size=4,
    num_train_epochs=MyVit.num_epochs,
    weight_decay=MyVit.weight_decay,
#     it was 0.01
    #0 FOR OVERFIIY
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
    seed = 42,
#    adam_beta1 = 0.9,
#    adam_beta2 = 0.9,
    #lr_scheduler_type = 'constant',
    #max_grad_norm = 1,
    #logging_strategy = 'epoch'
)


from sklearn.metrics import accuracy_score
import numpy as np
# these lines import the necessary modules, 'accuracy_score' from 'sklearn_metrics' and 'numpy'
# for numerical computations 

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))
#  this fuction takes in the eval_pred as an argument, which represents predictions 
# labels for the evaluation dataset
# Inside the function, 'eval_pred' is unpacked into 'predictions' and 'labels'. 
# 'predictions' are the predicted probabilities or logits for each class, while labels are the ground truth labels.

# Next, np.argmax(predictions, axis=1) is used to find the index of the highest probability or 
# the predicted class for each example. This is done by specifying axis=1, 
# which indicates that the maximum value should be computed along the second axis (class axis) 
# of the predictions array.

# Finally, the accuracy is calculated by comparing the predicted classes (predictions) 
# with the ground truth labels (labels) using accuracy_score(predictions, labels). 
# The accuracy_score function from sklearn.metrics calculates the accuracy metric by comparing the predicted and true labels.

# The function returns a dictionary with the accuracy metric, 
# where the key is "accuracy" and the value is the computed accuracy score.

# Overall, this code defines a function compute_metrics that computes the accuracy metric 
# for evaluating the model's performance.



# Then we just need to pass all of this along with our datasets to the Trainer:

# In[45]:


import torch
# this code imports the torch library 

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    
)




trainer.train()
torch.save(model, "saved_models/ViT_pretrained_tumorPrediction_"+data_set+'_'+tst_patient+".pt")

outputs = trainer.predict(val_ds)

loss_tst = outputs.metrics['test_loss']
print(f"Test best loss: {round(loss_tst, 2)}")
logger_dic['loss_test'] = loss_tst

accu_tst = outputs.metrics['test_accuracy']
logger_dic['accuracy_test'] = accu_tst


outputs = trainer.predict(val_ds)
y_true = outputs.label_ids

y_pred = outputs.predictions.argmax(1)

from datasets import load_metric
metric = load_metric('f1')

final_score = metric.compute(predictions=y_pred, references=y_true)
logger_dic['f1_test'] = final_score['f1']

import json
with open('output/logs/logVitPretrainedTumorHer2st.json', 'r') as openfile:
 
    # Reading from json file
    json_list = json.load(openfile)

json_list.append(logger_dic)

with open('output/logs/logVitPretrainedTumorHer2st.json', 'w') as outfile:
    json.dump(json_list, outfile)





