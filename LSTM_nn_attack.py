CourseFolder=r""
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
import csv
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import sys
import tensorflow_addons as tfa
from sklearn.metrics import matthews_corrcoef
np.set_printoptions(threshold=sys.maxsize)

# hyperparameters
home_number = 5
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

data_01 = pd.read_csv(CourseFolder+'occupancy_dataset/home' + str(home_number) + '/' + str(home_number).zfill(2) + '_' + season_name + '.csv')

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

plugs_01 = os.listdir(CourseFolder+'occupancy_dataset/home' + str(home_number) + '/plugs/')
plugs_readings_01 = pd.DataFrame()
target_01 = pd.DataFrame()
flage = 1

input_size = len(plugs_01)
for plug in plugs_01:

    readings_01 = pd.DataFrame()
    p = os.listdir(CourseFolder+'occupancy_dataset/home' + str(home_number) + '/plugs/' + plug)

    for f in p:
        is_common = 1
        for plg in plugs_01:
            is_common *= os.path.exists(CourseFolder+'occupancy_dataset/home' + str(home_number) + '/plugs/' + plg + '/' + f)

        if is_common:
            for i in range(len(occ_dates)):
                #file f is in occ_dates
                if occ_dates[i][:] == f[:4] + f[5:7] + f[8:10]:
                    if flage == 1:
                        target_01 = pd.concat([target_01, data_01.iloc[i, 1:]], axis=0)

                    readings_01 = pd.concat([readings_01, pd.read_csv(CourseFolder+'occupancy_dataset/home' + str(home_number) + '/plugs/' + plug + '/' + f, header=None)], sort=False)
    flage = 0

    plugs_readings_01 = pd.concat([plugs_readings_01, readings_01], axis=1)

count = plugs_readings_01.count()
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

indexes0 = [i for i,j in enumerate(Y) if j[0]==0]
indexes1 = [i for i,j in enumerate(Y) if j[0]==1][:len(indexes0)]
indexes = indexes0+indexes1
new_Y = np.array([Y[i] for i in indexes])
new_X = np.array([X[i] for i in indexes])
print(np.shape(new_Y))
print(np.shape(new_X))
X_train, X_test, y_train, y1_test = train_test_split(new_X, new_Y, test_size=0.2, shuffle=True, random_state=1)

x_test = torch.FloatTensor(X_test.tolist()).to(device)
y_test = torch.LongTensor(y1_test.tolist()).to(device)
test = torch.utils.data.TensorDataset(x_test, y_test.reshape(-1))
print(np.shape(x_test))

batch_size = 10
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
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
        batch_size = input.size(0)
        hidden = self._init_hidden(batch_size)
        input.view(batch_size, self.sequence_length, self.input_size)
        output, hidden = self.gru(input, hidden)
        hidden = F.dropout(hidden[-1], training=training)
        fc_output = self.fc(hidden)

        return fc_output

    def _init_hidden(self, batch_size):
        device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return Variable(hidden).to(device)

model = RNNClassifier(input_size, hidden_size, output_size, seq_length,layer_size, False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)
print(model)

if use_cuda and torch.cuda.is_available():
    model.cuda()


loss_func = nn.CrossEntropyLoss()

pretrained_model = CourseFolder + 'model_' + str(home_number) + '_' + season_name
print(pretrained_model)
if os.path.exists(pretrained_model):
    print("model exists")
    device = torch.device("cuda")
    model.load_state_dict(torch.load(pretrained_model, map_location="cuda:0"))
    model.to(device)

def test(epsilon):
    out = []
    out = np.array(out)

    vmax = x_test.max()
    vmin = x_test.min()
    correct = 0
    num_samples = 0
    i = 0
    flag = True
    pert_x = []
    pert_y = []
    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
        if batch_idx==500:
            break
        k_x = []
        k_y = []

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

            sign_data_grad = data_grad.sign()
            pertubation = epsilon * sign_data_grad * (vmax - vmin)
            perturbed_data = x_test_var + pertubation
            flag = False
        else:
            perturbed_data = x_test_var - pertubation[:x_test_var.shape[0]]
            flag = True
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",np.shape(perturbed_data))
        # pert_x.append(perturbed_data.tolist())
        # pert_y.append(y_test_var.tolist())
        # for j in perturbed_data:
        #     k = j.item()
        #     print(k.type)
        #     k_x.append(k)
        # for j in y_test_var:
        #     k_y.append(j.item())
        # pert_x.append(k_x)
        # pert_y.append(k_y)
        for m in perturbed_data.cpu().detach().numpy():
            pert_x.append(m)
        for n in y_test_var.cpu().detach().numpy():
            pert_y.append(n)
    #     _, idx = net_out.max(1)

    #     correct += (idx == y_test_var).sum()

    # print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, 0, 0,
    #                                                          float(correct * 100) / num_samples))
    return np.array(pert_x),np.array(pert_y)

# p,q = test(0.1)
# print(p)
# print(q)
# print(np.shape(p))
# print(np.shape(q))
# with open(f'{home_number}_{season_name}_perturbed_new.csv','w') as f:
#     write = csv.writer(f) 
#     for j in p:
#         write.writerow(j)
counts = np.bincount(y_train[:, 0])
print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(y_train)
    )
)

x1 = layers.Input(shape=(30,6))
net = layers.Flatten()(x1)
net = layers.Dense(256, activation='relu')(net)
net = layers.Dense(64, activation='relu')(net)
net = layers.Dense(32, activation='relu')(net)
net = layers.Dense(16, activation='relu')(net)
net = layers.Dense(1, activation='sigmoid')(net)
model1 = Model(inputs=x1, outputs=net)
opt = keras.optimizers.Adam()
model1.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model1.summary()
model1.fit(X_train, y_train, epochs=30000, validation_data=(X_test,y1_test), batch_size=10000)
# model.save(f"home{str(home_number)}_{season_name}_new_balanced_30k.h5",save_format = 'tf')

epsilons = [0.00001, 0.0001, 0.001, 0.01, 0.1]
accuracies = []
examples = []

for eps in epsilons:
    p,q = test(eps)
    q = np.expand_dims(q, axis=-1)
    #p = tf.convert_to_tensor(p)
    #q = tf.convert_to_tensor(q)
    kk = model1.predict(p, batch_size=10000, verbose=1)
    kk1 = model1.evaluate(p, q)
    print(eps)
    kk_pred = np.array([[1] if i > 0.5 else [0] for i in kk])
    acc = np.mean(kk_pred == q)
    print("Accuracy:",acc)
    print(np.shape(q))
    print(np.shape(kk_pred))
    mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=1)
    mcc.update_state(q, kk_pred)
    # print(matthews_corrcoef(y_test, kk_pred))
    print('Matthews correlation coefficient is:',mcc.result().numpy())
    y_pred =(kk>0.5)
    cm = confusion_matrix(q, y_pred)
    print(cm)