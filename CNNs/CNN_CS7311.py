##################################################
# Simple CNN for classification of the 24 categories for  cs 7311

#################################################3
#
#   BIG FAILURE
#
#
##################################################

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
from cvjson.cvj import CVJ
import numpy as np
from PIL import Image
import os





def add_one(list_of_classes, index):
    list_of_classes[index] += 1


class MardctClassDataset(torch.utils.data.Dataset):

    def __init__(self, cvj_obj):
        ### CVJ objects need the image path made available

        #######
        #Get images to class

        self._remove_others(cvj_obj)
        
        np_labels, np_images = self._get_images_labels(cvj_obj)

        ############ Dumb IDEA 
        # thought that the labels and images from had to be combined so I could split them using the dataloader....WOW
        # np_test_labels, np_test_images = self._get_images_labels(test_cvj)
        # self.images = torch.from_numpy(np.concatenate((self.np_train_images, self.np_test_images), axis=0))
        # self.labels = torch.from_numpy(np.concatenate((self.np_train_labels, self.np_test_labels), axis=0))

        self.images = np_images
        self.labels = torch.from_numpy(np_labels)

        self.transforms = transforms.Compose([
                                              transforms.Resize((520,520)),
                                              transforms.ToTensor(),
                                              ]) # this is if more transforms are wanted.



    def __getitem__(self, index):
        return (self.transforms(self.images[index]), self.labels[index])

    def __len__(self):
        return self.labels.shape[0]

    def _remove_others(self, mardct_cvj):

        mardct_cvj.clean_images()
        remove_list = []

        for ann in mardct_cvj["annotations"]:
            if ann["category_id"] == 24:
                remove_list.append(ann)

        count = 0
        for ann in remove_list:
            mardct_cvj["annotations"].remove(ann)
            count += 1

        remove_cat = None
        for cat in mardct_cvj["categories"]:
            if cat["id"] ==  24:
                remove_cat = cat

        mardct_cvj["categories"].remove(remove_cat)
        print("removed {} other boats".format(count))

        mardct_cvj.clear_dicts()

    def _get_images_labels(self, cvj_obj):
        
        cvj_obj_image_id_anns = cvj_obj.get_image_id_2_anns()

        labels = []
        images = []

        count = 0
        for img_id, ann_list in cvj_obj_image_id_anns.items():
            try:
                img = Image.open(cvj_obj.get_image_id_2_filepath(img_id))
                images.append(img.copy())
                img.close()
                labels.append(ann_list[0]["category_id"])
                if count % 100 is 0:
                    print("processing image {}".format(count))

                count += 1
            
            except KeyError as _:
                continue

        return np.asarray(labels), images

mardct_train_path = "/home/ben/Desktop/M12_Folder_Mimic/Datasets/Train/Cocoized/without_coco_categories/completed_train_refinement.json"
train_images = "/home/ben/Desktop/M12_Folder_Mimic/Datasets/Train/Images/completed_train_refinement"

mardct_test_path = "/home/ben/Desktop/M12_Folder_Mimic/Datasets/Validation/Cocoized/without_coco_categories/completed_test_refinement.json"
test_images = "/home/ben/Desktop/M12_Folder_Mimic/Datasets/Validation/Images/completed_refinement_test_images"

cvj_train = CVJ(mardct_train_path, train_images)
cvj_test = CVJ(mardct_test_path, test_images)

categories = list(cvj_train.get_category_names())

cat_counts = [0 for cat in categories]

########### DATA QUEUEING PARAMS

batch_size = 20 #

train = MardctClassDataset(cvj_train)
test = MardctClassDataset(cvj_test)

train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size)

############### NETWORK PARAMS

num_conv_layers = 5

pixels = 240 * 800 * 3
input_neurons = 9000 # I'll figure that out eventually.....pixels // num_conv_layers # mardct image size

# hidden_neurons = 250 # random hidden numbers Turns out that the convolutional layers are what really matter here.
output_neurons = len(categories) # output neurons
learning_rate = .01
epochs = 10

"""
When applying convolution to the images we need to figure out how much downsampling will occur.  Downsampling is directly 
related to the filter size and the stride. Padding needs to happen to images more than likely. This is so we don't lose image
information.  Most of the time a 0 padding is used.  Padding is not necessary if the filters fit.
Images are X * Y size.  

padding_size = ((filter_size) - 1) / 2
size_of_feature_map =  (((X - (filter_size)) + 2(padding)) / (stride) )+ (1)
pooling_size = size_of_feature_map / 2
"""

class my_net(nn.Module):
    def __init__(self):
        super(my_net, self).__init__()

        # for i in range(num_conv_layers): 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # X = 520, filter_size = 3, padding=1, stride = 1
        # size_of_feature_map =  ((X - (filter_size)) + 2(padding) / (stride)) + 1
        # so size_of_feature_map = ((520-3)+(2(1)/1)) + 1 = 520


        self.batch_norm = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2) # output of feature maps is now 520/2 = 260

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, stride=1, kernel_size=3, padding=1)
        # determine padding size for input_size to = output_size
        # padding_size = ((filter_size) - 1) / 2
        # padding_size = (3-1)/2 = 1
        # so size_of_feature_map = ((260-3)+(2(1)/1)) + 1 = 260

        self.batch_norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2) # feature map is now 260/2 = 130

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, stride=1, kernel_size=3, padding=1)
        # so size_of_feature_map = ((130-3)+(2(1)/1)) + 1 = 130
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2) # 130/2 = 65

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, stride=2, kernel_size=3, padding=1)
        # so size_of_feature_map = ((65-3)+(2(1)/2)) + 1 = 64
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2) # 65/2 = 32

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, stride=1, kernel_size=3, padding=1)
        # so size_of_feature_map = ((32-3)+(2(1))/2) + 1 = 16
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(kernel_size=2) # 16/2 = 8
        # I guess this divided by 4 because of strides of 2 in the convolution and in the pooling.

        # self.conv6 = nn.Conv2d(in_channels=128, out_channels=64, stride=1, kernel_size=3, padding=1)
        # # so size_of_feature_map = ((16-3)+(2(1)/1)) + 1 = 16
        # self.batch_norm6 = nn.BatchNorm2d(64)
        # self.pool6 = nn.MaxPool2d(kernel_size=2) # 16/2 = 8

        # self.conv7 = nn.Conv2d(in_channels=64, out_channels=32, stride=1, kernel_size=3, padding=1)
        # # so size_of_feature_map = ((50-3)+(2(1)/1)) + 1 = 50
        # self.batch_norm7 = nn.BatchNorm2d(32)
        # self.pool7 = nn.MaxPool2d(kernel_size=5) # 50/5 = 10
    
        self.fc1 = nn.Linear(128*8*8, output_neurons)

        # self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        # self.fc3 = nn.Linear(hidden_neurons, output_neurons)

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.pool(out)

        # print(out.shape)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.pool2(out)

        # print(out.shape)
        out = self.conv3(out)
        out = self.batch_norm3(out)
        out = self.relu(out)
        out = self.pool3(out)

        # print(out.shape)
        out = self.conv4(out)
        out = self.batch_norm4(out)
        out = self.relu(out)
        out = self.pool4(out)

        # print(out.shape)
        out = self.conv5(out)
        out = self.batch_norm5(out)
        out = self.relu(out)
        out = self.pool5(out)

        # print(out.shape)
        # out = self.conv6(out)
        # out = self.batch_norm6(out)
        # out = self.relu(out)
        # out = self.pool6(out)
        # print(out.shape)

        # out = self.conv7(out)
        # out = self.batch_norm7(out)
        # out = self.relu(out)
        # out = self.pool7(out)

        out = out.view(-1, (128*8*8))
        out = self.fc1(out)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.fc3(out)
        return out

model = my_net()
gpu_available = torch.cuda.is_available()
if gpu_available:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)


correct_during_training = 0
total_during_training = 0
# for i in range(epochs):
#     for k, (images, labels) in enumerate(train_loader):

#         images = torch.autograd.Variable(images)
#         labels = torch.autograd.Variable(labels)

#         if gpu_available:
#             images = images.cuda()
#             labels = labels.cuda()

#         outputs = model(images)
#         pred = torch.max(outputs, 1)[1]
#         nu = [add_one(cat_counts, index) for index in  pred]
#         # print(cat_counts)
        

#         if gpu_available:
#             correct_during_training += (pred.cpu() == labels.cpu()).sum()
#             total_during_training += labels.size(0)

 
#         loss = criterion(outputs, labels)
#         loss.backward()

#         optimizer.step()
#         optimizer.zero_grad()

#         percentage = (100 * (correct_during_training.float()/total_during_training))
#         if k % 100 == 0:
#             print("epoch {}, iteration {}".format(i, k))


torch.save(model.state_dict(), "/home/ben/Desktop/bens_model.pth")

# #Later to restore:
model.load_state_dict(torch.load("/home/ben/Desktop/bens_model.pth"))
model.eval()
# for i, cat in enumerate(categories):
#     try:
#         class_count = len(cvj_train.get_class_id_2_anns()[i])
#     except KeyError:
#         continue
#     try:
#         accuracy = (class_count/cat_counts[i]) *100
#     except ZeroDivisionError:
#         accuracy = 0
#     print("category {} had {} GT and the model in training predicted {} so accuracy is {}".format(cat, class_count, cat_counts[i], accuracy))

cat_counts = [0 for cat in categories]

correct_amount = 0
total_amount = 0
for i, (images, labels) in enumerate(test_loader):

    images = torch.autograd.Variable(images)
    labels = torch.autograd.Variable(labels)

    if gpu_available:
        images = images.cuda()
        labels = labels.cuda()

    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    nu = [add_one(cat_counts, index) for index in  preds]

    if gpu_available:
        correct_amount += (preds.cpu() == labels.cpu()).sum()
        total_amount += labels.size(0)

        
accuracy = correct_amount.float() / total_amount

print("total correct = {}, total tested = {}, accuracy = {}".format(correct_amount, total_amount, accuracy))

for i, cat in enumerate(categories):
    
    try:
        class_count = len(cvj_train.get_class_id_2_anns()[i])
    except KeyError:
        continue

    try:
        accuracy = (class_count/cat_counts[i]) *100
    except ZeroDivisionError:
        accuracy = 0

    print("category {} had {} GT and the model in training predicted {} so accuracy is {}".format(cat, class_count, cat_counts[i], accuracy))

