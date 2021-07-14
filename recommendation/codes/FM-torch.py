import pandas as pd
import numpy as np 

import torch
from torch import nn
from torch import optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
    df_sparse = df[sparse_feats].fillna('-1')
    for f in tqdm(sparse_feats):
        lbe = LabelEncoder()
        df_sparse[f] = lbe.fit_transform(df_sparse[f])
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_f = onehot_encoder.fit_transform(df_sparse[f].values.reshape(-1,1))
        onehot_f = pd.DataFrame(onehot_f, columns=['{}_{}'.format(f,i) for i in range(onehot_f.shape[1])])
        features.append(onehot_f)
    
    df_new = pd.concat(features, axis=1)
    # print(df_new.shape) # (1599, 11006)
    return df_new

# 因子分解机
class FM(nn.Module):
    def __init__(self, n_features, embedding_dim=8):
        super(FM,self).__init__()
        self.feature_weight = nn.Parameter(torch.Tensor(n_features, embedding_dim))
        nn.init.kaiming_uniform_(self.feature_weight)

    def forward(self,feture):
        a = torch.mm(feture,self.feature_weight).pow(2)
        b = torch.mm(feture.pow(2),self.feature_weight.pow(2))
        return torch.sigmoid(torch.mean(a - b,axis=1))

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
feats = process_feat(data, dense_feats, sparse_feats)
print(feats.head())
# 划分训练和验证数据
x_trn, x_tst, y_trn, y_tst = train_test_split(feats.values, data['Label'].values, test_size=0.2, random_state=2020)
train_dataset = FMdata(x_trn, y_trn)
trainloader = torch.utils.data.DataLoader(train_dataset,batch_size = 1024,
    shuffle = True,
    num_workers = 8
)
val_dataset = FMdata(x_tst, y_tst)
valloader = torch.utils.data.DataLoader(val_dataset,batch_size = 1024,
    shuffle = True,
    num_workers = 8
)

# 定义模型
model = FM(n_features=feats.shape[1]).cuda()
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