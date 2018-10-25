import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2

batch_size = 30
input_neurons = 784 # mnist image size
hidden_neurons = 250 # random hidden numbers
output_neurons = 10 # output neurons
learning_rate = .01
epochs = 10

train = torchvision.datasets.FashionMNIST(root="./fashion_mnist", train=True, transform=transforms.ToTensor(), download=True)
test = torchvision.datasets.FashionMNIST(root="./fashion_mnist", train=False, transform=transforms.ToTensor())

# classes, ids = torchvision.datasets.folder.find_classes("./fashion_mnist") # dumb method

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# for i in range(10):
#     image, label = train[i]

#     print(classes[label])

#     print(image.numpy().shape)
#     image = image.numpy().transpose(2,1,0)
#     print(image.shape)
#     cv2.imshow("yolo",image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     print(label)
# input()

train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size)


"""
When applying convolution to the images we need to figure out how much downsampling will occur.  Downsampling is directly 
related to the filter size and the stride. Padding needs to happen to images more than likely. This is so we don't lose image
information.  Most of the time a 0 padding is used.  Padding is not necessary if the filters fit.
Images are X * Y size.  

padding_size = ((filter_size) - 1) / 2
size_of_feature_map =  ((X - (filter_size)) + 2(padding) / (stride)) + 1
pooling_size = size_of_feature_map / 2
"""

class my_net(nn.Module):
    def __init__(self):
        super(my_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        # X = 28, filter_size = 3, padding=1, stride = 1
        # size_of_feature_map =  ((X - (filter_size)) + 2(padding) / (stride)) + 1
        # so size_of_feature_map = ((28-3)+/(2(1)/1)) + 1 = 28
        self.batch_norm = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2) # output of feature maps is now 28/2 = 14
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, stride=1, kernel_size=3, padding=1)
        # determine padding size for input_size to = output_size
        # padding_size = ((filter_size) - 1) / 2
        # padding_size = (3-1)/2 = 1

        self.batch_norm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2) # feature map is now 14/2 = 7

        input_neurons = 32*7*7 # 32 feature maps that are now 7x7
        hidden_neurons = 600 # arbitrary
        output_neurons = 10 # 10 classes in dataset

        self.fc1 = nn.Linear(input_neurons, output_neurons)

        # self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        # self.fc3 = nn.Linear(hidden_neurons, output_neurons)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.pool2(out)

        out = out.view(-1, (32*7*7))
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
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


correct_during_training = 0
total_during_training = 0
for i in range(epochs):
    for k, (images, labels) in enumerate(train_loader):

        images = torch.autograd.Variable(images)
        labels = torch.autograd.Variable(labels)

        if gpu_available:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        pred = torch.max(outputs, 1)[1]

        if gpu_available:
            correct_during_training += (pred.cpu() == labels.cpu()).sum()
            total_during_training += labels.size(0)

 
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        percentage = (100 * (correct_during_training.float()/total_during_training))

        if k % 100 == 0:
            print("loss = {},  Correct = {}, Total = {},  Percentage = {}".format(loss, correct_during_training, total_during_training, percentage))



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

    if gpu_available:
        correct_amount += (preds.cpu() == labels.cpu()).sum()
        total_amount += labels.size(0)

        
accuracy = correct_amount.float() / total_amount

print("total correct = {}, total tested = {}, accuracy = {}".format(correct_amount, total_amount, accuracy))

