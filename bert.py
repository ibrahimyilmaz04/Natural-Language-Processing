import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import codecs

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import transformers
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import AdamW
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

np.set_printoptions(threshold = np.inf)


use_cuda = torch.cuda.is_available() and not False
device = torch.device('cuda' if use_cuda else 'cpu')


# ad_samples
# def load_ad_test_x():
#     ad_test_x = codecs.open("ad_samples/adversarial_examples_eps0.txt", mode = "r", encoding = "utf-8")
#     line = ad_test_x.readline()
#     list = []
#     while line:
#         a = line.split()
#         a = ".".join(str(a).split(".")[:-1])
#         list.append(str(a).replace("b'", "").replace("[\"", "").replace("\"]", "").split(" ")[:])
#         line = ad_test_x.readline()
#
#     return np.array(list)
#     ad_test_x.close()
#
# ad_test_x = load_ad_test_x()
# print(ad_test_x)

# value = 0 # fake datas label
# ad_test_y = [value for i in range(ad_test_x.__len__())]
# ad_test_y = np.array(ad_test_y)


TEST_SIZE = 0.1

real = pd.read_csv('majestic_million.csv', usecols = ['Domain'])
train_size = int(135000 * (1 - TEST_SIZE))
real_train = real[:train_size]
real_test = real[train_size: 135000]


real_train['label'] = 1
real_test['label'] = 1

currentDir = os.getcwd()
datasetlist = os.listdir(currentDir + '/Dataset')

fake_train = pd.DataFrame()
fake_test = pd.DataFrame()
fake_testWithClass = []
total_malicious_count = 0

for f in datasetlist:
    faketemp = pd.read_csv(currentDir + '/Dataset/' + f, header = None, usecols = [0], names = ['Domain'])
    faketemp = faketemp.drop([0], axis = 0)
    total_malicious_count += faketemp.size
    train_size = int(faketemp.size * (1 - TEST_SIZE))
    fake_train = pd.concat([fake_train, faketemp[:train_size]])
    fake_test = pd.concat([fake_test, faketemp[train_size:]])
    fake_test_class = faketemp[train_size:]
    fake_test_class['label'] = 0
    fake_testWithClass.append([f, fake_test_class])

fake_train['label'] = 0
fake_test['label'] = 0

# Merge both real and fake to make the dataset
frames_train = [real_train, fake_train]
frames_test = [real_test, fake_test]
data_train = pd.concat(frames_train)
data_test = pd.concat(frames_test)
data_train.reset_index(inplace = True)
data_test.reset_index(inplace = True)

data_train['domain_name'] = data_train['Domain'].str.partition('.', expand=True)[[0]]
data_test['domain_name'] = data_test['Domain'].str.partition('.', expand=True)[[0]]

train_domain_name = [[str(token)] for token in data_train['domain_name']]
test_domain_name = [[str(token)] for token in data_test['domain_name']]

train_sentences = ['[CLS] ' + str(sent) + ' [SEP]' for sent in train_domain_name]
test_sentences = ['[CLS] ' + str(sent) + ' [SEP]' for sent in test_domain_name]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
train_tokens = [tokenizer.tokenize(sent) for sent in train_sentences]
test_tokens = [tokenizer.tokenize(sent) for sent in test_sentences]

MAX_LEN = 128

# 将分割后的句子转化成数字 word --> idx
train_input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in train_tokens]
test_input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in test_tokens]
print(type(train_input_ids))


train_input_ids = pad_sequences(train_input_ids, maxlen = MAX_LEN, dtype = "long", truncating = "post", padding = "post")
test_input_ids = pad_sequences(test_input_ids, maxlen = MAX_LEN, dtype = "long", truncating = "post", padding = "post")

# mask
train_attention_mask = []
for seq in train_input_ids:
    seq_mask = [float(i > 0) for i in seq]
    train_attention_mask.append(seq_mask)

test_attention_mask = []
for seq in test_input_ids:
    seq_mask = [float(i > 0) for i in seq]
    test_attention_mask.append(seq_mask)


train_labels = data_train["label"].values
test_labels = data_test["label"].values


# 到这里我们就可以生成 dataloader 放入模型中进行训练了
train_inputs = torch.tensor(train_input_ids)
train_masks = torch.tensor(train_attention_mask)
train_labels = torch.tensor(train_labels)

test_inputs = torch.tensor(test_input_ids)
test_masks = torch.tensor(test_attention_mask)
test_labels = torch.tensor(test_labels)

epochs = 3
batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = batch_size)

# 加载预训练模型， 首先加载 bert 的配置文件，然后使用 BertForSequenceClassification 这个类来进行分类
modelconfig = BertConfig.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config = modelconfig)
model.to(device)


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = optim.Adam(optimizer_grouped_parameters, lr = 2e-5)

# 定义一个计算准确率的函数
def flat_accuracy(pred, labels):
    pred_flat = np.argmax(pred, axis = 1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

train_loss_set = [] # 可以将 loss 加入到列表中， 后期画图使用

def train():
    for epoch in trange(epochs):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_inputs_mask, b_labels = batch
            b_input_ids = torch.tensor(b_input_ids, dtype=torch.long)
            b_inputs_mask = torch.tensor(b_inputs_mask, dtype=torch.long)
            b_labels = torch.tensor(b_labels, dtype=torch.long)

            optimizer.zero_grad()
            # 取第一个位置， BertForSequneceClassfication 第一个位置是 Loss， 第二个位置是 [CLS] 的 logits
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_inputs_mask, labels=b_labels)[0]
            train_loss_set.append(loss.item())
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        torch.save(model, 'mymodel.pkl')
        torch.save(model.state_dict(), "mymodel_params.pkl")


def reload_model():
    train_model = torch.load("mymodel.pkl")
    return train_model

def test():
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    model = reload_model()
    for batch in test_dataloader:
        # batch = tuple(t.to(device) for t in batch)
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_inputs_mask, b_labels = batch


        b_input_ids = torch.tensor(b_input_ids, dtype=torch.long)
        b_inputs_mask = torch.tensor(b_inputs_mask, dtype=torch.long)
        b_labels = torch.tensor(b_labels, dtype=torch.long)

        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_inputs_mask)[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("Test Accuracy: {}".format(eval_accuracy / nb_eval_steps))



if __name__ == '__main__':
    train()
    reload_model()
    test()











