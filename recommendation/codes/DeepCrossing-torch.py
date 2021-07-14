import pandas as pd
import numpy as np 

import torch
from torch import nn
from torch import optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# dense特征取对数　　sparse特征进行类别编码
def process_feat(data, dense_feats, sparse_feats):
    df = data.copy()
    features = []
    # dense
    df_dense = df[dense_feats].fillna(0.0)
    for f in tqdm(dense_feats):
        df_dense[f] = df_dense[f].apply(lambda x: np.log(1 + x) if x > -1 else -1)
    features.append(df_dense)
    # sparse
    sparse_feats_dim = []
    df_sparse = df[sparse_feats].fillna('-1')
    for f in tqdm(sparse_feats):
        lbe = LabelEncoder()
        df_sparse[f] = lbe.fit_transform(df_sparse[f])
        sparse_feats_dim.append(df_sparse[f].max()+1)
    features.append(df_sparse)
    df_new = pd.concat(features, axis=1)
    return df_new,sparse_feats_dim

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        inputs = x
        inputs = self.relu1(self.linear1(inputs))
        # print(self.linear1.weight.data)
        inputs = self.relu2(self.linear2(inputs))
        inputs = self.relu3(inputs + x)
        return inputs

class DeepCrossing(nn.Module):
    def __init__(self, dense_feats, sparse_feats, sparse_feats_dim, embedding_dim=4, hidden_dim=64):
        super(DeepCrossing,self).__init__()
        self.in_dim = len(sparse_feats)*embedding_dim + len(dense_feats)
        self.embeddings = nn.ModuleList()
        for f,i in zip(sparse_feats,sparse_feats_dim):
            self.embeddings.append(nn.Embedding(i,embedding_dim))
        self.resblock = ResidualBlock(self.in_dim, hidden_dim)
        self.dense = nn.Linear(self.in_dim, 1)
        self.act = nn.Sigmoid()
        self.partition_dim = len(dense_feats) # 预处理前部分是dense特征
        self.sparse_feats = sparse_feats

    def forward(self,feture):
        dense_f = feture[:,:self.partition_dim]
        for i in range(len(self.sparse_feats)):
            # print(feture[:,self.partition_dim+i].device,next(self.embeddings[i].parameters()).device)
            embedding = self.embeddings[i](feture[:,self.partition_dim+i].long())
            dense_f = torch.cat((dense_f, embedding), dim=1)
        y = self.resblock(dense_f)
        y = self.dense(y)
        y = self.act(y)
        return y.squeeze()


class FMdata(torch.utils.data.Dataset):
    def __init__(self,data,label):
        super(FMdata,self).__init__()
        self.data = data.astype(np.float32)
        self.label = label.astype(np.float32)

    def __getitem__(self,idx):
        feature = self.data[idx]
        label = self.label[idx]
        return feature, label
    def __len__(self):
        return len(self.label)

# 读取数据
print('loading data...')
data = pd.read_csv('./data/kaggle/train.csv')

# dense 特征开头是I，sparse特征开头是C，Label是标签
cols = data.columns.values

dense_feats = [f for f in cols[1:] if f[0] == 'I' ]
sparse_feats = [f for f in cols[1:] if f[0] == 'C']
# print(dense_feats)
# print(sparse_feats)
# 对dense数据和sparse数据分别处理
print('processing features')
feats,sparse_feats_dim = process_feat(data, dense_feats, sparse_feats)
# 划分训练和验证数据
x_trn, x_tst, y_trn, y_tst = train_test_split(feats.values, data['Label'].values, test_size=0.2, random_state=2020)
train_dataset = FMdata(x_trn, y_trn)
trainloader = torch.utils.data.DataLoader(train_dataset,batch_size = 12,
    shuffle = True,
    num_workers = 0
)
val_dataset = FMdata(x_tst, y_tst)
valloader = torch.utils.data.DataLoader(val_dataset,batch_size = 12,
    shuffle = True,
    num_workers = 0
)

# 定义模型
model = DeepCrossing(dense_feats, sparse_feats, sparse_feats_dim).cuda()
print(model)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)
    
# 训练模型
for epoch in range(100):
    model.train()
    train_losses = []
    for feature, label in trainloader:
        feature = feature.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        y = model(feature)
        # print(y.dtype,rating)
        loss = loss_fn(y,label)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    scheduler.step()
    
    # eval
    model.eval()
    with torch.no_grad():
        losses = []
        for feature, label in valloader:
            feature = feature.cuda()
            label = label.cuda()
            y = model(feature)
            loss = loss_fn(y,label)
            losses.append(loss.item())
    print('epoch: {}, train loss:{}, val loss:{}'.format(epoch, sum(train_losses)/len(train_losses), sum(losses)/len(losses)))