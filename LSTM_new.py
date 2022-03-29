import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import os

# hyperparameters
home_number = 4
season_name = "winter"


output_size = 2
hidden_size = 150
layer_size = 2

batch_size = 10000
learning_rate = 0.001

epochs = 200

use_cuda=True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print('home_'+ str(home_number).zfill(2) + '_' + season_name)
# home_01_sum = pd.read_csv('occupancy_dataset/home2/02_summer.csv')
# home_01_win = pd.read_csv('occupancy_dataset/home2/02_winter.csv')

#home_01 = [home_01_sum, home_01_win]
#data_01 = pd.concat(home_01)

data_01 = pd.read_csv('occupancy_dataset/home' + str(home_number) + '/' + str(home_number).zfill(2) + '_' + season_name + '.csv')


# data_01.head(600)

def occ_date(c):
    val = c[3:6]
    if val == 'Jul':
        m = '07'
    elif val == 'Aug':
        m = '08'
    elif val == 'Sep':
        m = '09'
    elif val == 'Oct':
        m = '10'
    elif val == 'Nov':
        m = '11'
    elif val == 'Dec':
        m = '12'
    elif val == 'Jan':
        m = '01'
    elif val == 'Feb':
        m = '02'
    elif val == 'Mar':
        m = '03'
    elif val == 'Apr':
        m = '04'
    elif val == 'May':
        m = '05'
    elif val == 'Jun':
        m = '06'

    return c[7:11] + m + c[:2]

occ_dates= []
for i in range(data_01.iloc[:,0].shape[0]):
    occ_dates.append(occ_date(data_01.iloc[i,0]))

plugs_01 = os.listdir('occupancy_dataset/home' + str(home_number) + '/plugs/')
plugs_readings_01 = pd.DataFrame()
target_01 = pd.DataFrame()
flage = 1

input_size = len(plugs_01)
#loop plugs directory(01, 02, ... , 12)
for plug in plugs_01:

    readings_01 = pd.DataFrame()
    p = os.listdir('occupancy_dataset/home' + str(home_number) + '/plugs/' + plug)

    #loop plugs's subdirectory(ex: 01)
    for f in p:
        is_common = 1
        for plg in plugs_01:
            is_common *= os.path.exists('occupancy_dataset/home' + str(home_number) + '/plugs/' + plg + '/' + f)

        if is_common:
            for i in range(len(occ_dates)):
                #file f is in occ_dates
                if occ_dates[i][:] == f[:4] + f[5:7] + f[8:10]:
                    if flage == 1:
                        target_01 = pd.concat([target_01, data_01.iloc[i, 1:]], axis=0)

                    readings_01 = pd.concat([readings_01, pd.read_csv('occupancy_dataset/home' + str(home_number) + '/plugs/' + plug + '/' + f, header=None)], sort=False)
    flage = 0

    plugs_readings_01 = pd.concat([plugs_readings_01, readings_01], axis=1)

# plugs_readings_01

# Overall Coverage Percentage

count = plugs_readings_01.count()
#Coverage_Percentage = (sum(plugs_readings_01.count()[0]) - sum(plugs_readings_01.eq(-1).sum())) / sum(plugs_readings_01.count()[0])
# print('Coverage Percentage = ', round(Coverage_Percentage, 4) * 100, '%')

plugs_readings_01.eq(-1).sum()

def drop_missisng(X, y):

    X = np.array(X)
    y = np.array(y)
    Xout = []
    yout = []

    val = -1
    for i in range(0,X.shape[0]-1):
        flag = 1
        for v in X[i]:
            if(v==val):
               flag = 0
        if flag:
           Xout.append(X[i])
           yout.append(y[i])
    return (np.array(Xout),np.array(yout))

X = plugs_readings_01.values.tolist()
y = target_01.values.tolist()

X,Y = drop_missisng(X,y)

#for i in range (11):
#    print (np.count_nonzero(X[:,i]))

# X = np.delete(X,4,1)
# X = np.delete(X,4,1)
# X = np.delete(X,4,1)
# X.shape

# X = X.sum(axis = 1).reshape(-1, 1) #[200000:250000]
#yx = y.sum(axis = 1) #[200000:250000]

#Xx = X[:1000000]
#Yx = Y[:1000000]
#y[i,:]

seq_length = 30

dataX = []
dataY = []

for i in range(0, len(X) - seq_length):
    _x = X[i:i + seq_length]
    _y = Y[i + seq_length]
    dataX.append(_x)
    dataY.append(_y)

X = np.array(dataX)
Y = np.array(dataY)

import torch
torch.cuda.is_available()
# True

# X.reshape(-1, 30)

Y.reshape(-1)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sklearn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.init as init

# %matplotlib inline

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, shuffle=True, random_state=1)

# Scaling data
#scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
#X_train = scaling.transform(X_train)
#X_test = scaling.transform(X_test)


#scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

x = torch.FloatTensor(X_train.tolist()).to(device)
y = torch.LongTensor(y_train.tolist()).to(device)
train = torch.utils.data.TensorDataset(x, y.reshape(-1))
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
#y = y.squeeze_()

x_test = torch.FloatTensor(X_test.tolist()).to(device)
y_test = torch.LongTensor(y_test.tolist()).to(device)
test = torch.utils.data.TensorDataset(x_test, y_test.reshape(-1))
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

#y_test = y_test.squeeze_()

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


class RNNClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, sequence_length, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.sequence_length = sequence_length

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          bidirectional=bidirectional, num_layers=self.n_layers, batch_first=True)
        for param in self.gru.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

        self.fc = nn.Linear(hidden_size, output_size)
        init.xavier_normal_(self.fc.weight.data)
        init.normal_(self.fc.bias.data)

    def forward(self, input, training=True):
        # Note: we run this all at once (over the whole input sequence)

        # input = B x S . size(0) = B
        batch_size = input.size(0)

        # Make a hidden
        hidden = self._init_hidden(batch_size)

        input.view(batch_size, self.sequence_length, self.input_size)

        output, hidden = self.gru(input, hidden)

        # Use the last layer output as FC's input
        # No need to unpack, since we are going to use hidden
        hidden = F.dropout(hidden[-1], training=training)
        fc_output = self.fc(hidden)

        return fc_output

    def _init_hidden(self, batch_size):
        device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return Variable(hidden).to(device)


# model = nn.Sequential(nn.Linear(input_size, hidden_size),
#                      nn.LeakyReLU(),
#                      nn.Linear(hidden_size, hidden_size),
#                      nn.LeakyReLU(),
#                      nn.Linear(hidden_size, hidden_size),
#                      nn.LeakyReLU(),
#                      nn.Linear(hidden_size, hidden_size),
#                      nn.LeakyReLU(),
#                      nn.Linear(hidden_size, output_size),
#                      nn.Sigmoid())

model = RNNClassifier(input_size, hidden_size, output_size, seq_length,layer_size, False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)
print(model)

#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

sum(Y==1)

loss_log = []
loss_tot = []

pretrained_model = 'model_' + str(home_number) + '_' + season_name
if os.path.exists(pretrained_model):
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

print("Training The Model...")

for e in range(epochs):
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

        x_var = Variable(X_batch).to(device)
        y_var = Variable(y_batch).to(device)

        optimizer.zero_grad()
        net_out = model(x_var)

        loss = loss_func(net_out, y_var)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss_log.append(loss.data)
    loss_tot.append(loss.data)
    print('Epoch: {} - Loss: {:.6f}'.format(e, loss.data))

torch.save(model.state_dict(), pretrained_model)

#plt.figure(figsize=(10,8))

#plt.plot(loss_tot)

#sum(torch.max(net_out.data, 1)[1].numpy())

#net_out.shape


#
# out = []
# out = np.array(out)
#
# for j in range(0, x_test.shape[0], int(batch_size / 3)):
#     x_test_mini = x_test[j:j + int(batch_size / 3)].to(device)
#     y_test_mini = y_test[j:j + int(batch_size / 3)].reshape(-1).to(device)
#
#     x_test_var = Variable(x_test_mini).to(device)
#     y_test_var = Variable(y_test_mini).to(device)
#
#     net_out = model(x_test_var)
#     _, idx = net_out.max(1)
#     idx = idx.data.numpy()
#
#     out = np.append(out, idx)
#     # print(j)
#
# # out = np.where (out <0.6, 0, 1)
#
# print(out.shape)
#
# scores_out = np.where( out <0.535, 0, 1)
#
# print(scores_out)
#
# np.array(net_out.data).shape
#
# pytorch_total_params = sum(p.numel() for p in model.parameters())
# pytorch_total_params
#
# np.sum(np.array(y_test)==0)
#
# def accuracy(out, labels):
#     outputs = out #np.argmax(out, axis=1)
#     return np.sum(outputs==labels)/float(labels.size)
#
# accuracy(scores_out, y_test.reshape(-1).cpu().detach().numpy())* 100
#
# from sklearn.metrics import accuracy_score
# print(accuracy_score(scores_out, y_test.cpu().detach().numpy()))
#
# np.array(y_test)




# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test(epsilon):
    out = []
    out = np.array(out)

    vmax = x_test.max()
    vmin = x_test.min()
    correct = 0
    num_samples = 0
    i = 0
    flag = True
    pert = []
    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):

        x_test_var = Variable(X_batch).to(device)
        y_test_var = Variable(y_batch).to(device)
        num_samples += len(y_test_var)

        if flag == True:
            x_test_var.requires_grad = True
            net_out = model(x_test_var, False)
            loss = loss_func(net_out, y_test_var)
            model.zero_grad()
            loss.backward()
            data_grad = x_test_var.grad.data

            # fgsm attack
            sign_data_grad = data_grad.sign()
            pertubation = epsilon * sign_data_grad * (vmax - vmin)
            perturbed_data = x_test_var + pertubation
            flag = False
        else:
            perturbed_data = x_test_var - pertubation[:x_test_var.shape[0]]
            flag = True
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",np.shape(perturbed_data))
	pert.append(perturbed_data.tolist())
        net_out = model(perturbed_data, False)
        _, idx = net_out.max(1)

        correct += (idx == y_test_var).sum()

    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, 0, 0,
                                                             float(correct * 100) / num_samples))
    return pert

#             out = np.append(out, net_out.cpu().detach().numpy())
#             #print(j)

# out = np.where (out <0.6, 0, 1)

#     print(out.shape)


epsilons = [0, .000001, .000005, .00001, .00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.10, 0.12 , 0.13, 0.14, 0.15, 0.16, 0.17,0.18, 0.19, 0.2, 0.25, 0.3,0.35, 0.4,0.5, 0.6, 0.7]

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    print(test(eps))