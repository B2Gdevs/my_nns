import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# torch.manual_seed(0)

# x = torch.randn(10, 3) # (samples, features)
# y = torch.rand(10,1)

# print(torch.max(x, 1)[1])
# print(y.round().view(-1, 10)[0].int())

# linear = nn.Linear(x.size()[1], y.size()[1])

# print("bias:", linear.bias)
# print("weight:", linear.weight)
# print("######## Gradients")
# print("bias:", linear.bias.grad)
# print("weight:", linear.weight.grad)

# criterion = nn.MSELoss() # your loss function that will create the error
# optimizer = torch.optim.SGD(linear.parameters(), lr=.001) # your optimizing function.  This updates weights

# pred = linear(x) # the actual model

# loss = criterion(pred, y) # your loss function creating your error

# loss.backward() # updates gradients

# for k in range(1000):
    
#     optimizer.step()


#     print("############# After Step\n\n")
#     print("bias:", linear.bias)
#     print("weight:", linear.weight)

#     print("######## Gradients")
#     print("bias:", linear.bias.grad)
#     print("weight:", linear.weight.grad)

#     pred = linear(x)

#     loss = criterion(pred, y)
#     print("loss : {}".format(loss))
#     loss.backward()

# print(pred)
# print(y)

batch_size = 30
input_neurons = 784 # mnist image size
hidden_neurons = 250 # random hidden numbers
output_neurons = 10 # output neurons
learning_rate = .01
epochs = 10


train = torchvision.datasets.MNIST(root="./", train=True, transform=transforms.ToTensor(), download=True)
test = torchvision.datasets.MNIST(root="./", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size)


class my_net(nn.Module):
    def __init__(self, input_neurons, output_neurons, hidden_neurons):
        super(my_net, self).__init__()
        self.fc1 = nn.Linear(input_neurons, hidden_neurons)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons) # this is totally unnecessary...maybe not. one can reuse the relu
        self.fc3 = nn.Linear(hidden_neurons, output_neurons)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

model = my_net(input_neurons, output_neurons, hidden_neurons)
gpu_available = torch.cuda.is_available()
if gpu_available:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


correct_during_training = 0
total_during_training = 0
for i in range(epochs):
    for k, (images, labels) in enumerate(train_loader):

        images = torch.autograd.Variable(images.view(-1, 28*28))
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





