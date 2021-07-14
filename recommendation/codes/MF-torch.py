import pandas as pd
import warnings
import os

import torch
from torch.utils import data
from torch import nn
from torch import optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

def get_data(root_path):
    # 读取数据时，定义的列名
    rnames = ['user_id','movie_id','rating','timestamp']
    data = pd.read_csv(os.path.join(root_path, 'ratings.dat'), sep='::', engine='python', names=rnames)
    lbe = LabelEncoder()
    data['user_id'] = lbe.fit_transform(data['user_id'])
    data['movie_id'] = lbe.fit_transform(data['movie_id']) 

    trn_data_, val_data_ = train_test_split(data,test_size=0.1)
    return trn_data_, val_data_

# 矩阵分解模型
class MF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=8):
        super(MF,self).__init__()
        
        self.user_embedding = nn.Embedding(n_users,embedding_dim)
        self.item_embedding = nn.Embedding(n_items,embedding_dim)
        nn.init.kaiming_uniform_(self.user_embedding.weight)
        nn.init.kaiming_uniform_(self.item_embedding.weight)


    def forward(self,user_idx,item_idx):
        user_embedding = self.user_embedding(user_idx)
        item_embedding = self.item_embedding(item_idx)
        score = torch.sum(user_embedding.mul(item_embedding),axis=1)
        # print(user_embedding.shape,item_embedding.shape,score.shape)
        return score.squeeze()

class MFdata(data.Dataset):
    def __init__(self,user_id,movie_id,rating):
        super(MFdata,self).__init__()
        self.user_id = user_id
        self.movie_id = movie_id
        self.rating = rating

    def __getitem__(self,idx):
        user_id = self.user_id[idx]
        movie_id = self.movie_id[idx]
        rating = self.rating[idx]
        return user_id,movie_id,rating
    def __len__(self):
        return len(self.user_id)

if __name__ == "__main__":
    # K表示最终给用户推荐的商品数量，N表示候选推荐商品为用户交互过的商品相似商品的数量
    k = 80
    N = 10

    # 读取数据
    root_path = './data/ml-1m/'
    trn_data, val_data = get_data(root_path)

    train_dataset = MFdata(trn_data['user_id'].values, trn_data['movie_id'].values, trn_data['rating'].values)
    trainloader = data.DataLoader(train_dataset,batch_size = 1024*12,
        shuffle = True,
        num_workers = 8
    )
    val_dataset = MFdata(val_data['user_id'].values, val_data['movie_id'].values, val_data['rating'].values)
    valloader = data.DataLoader(val_dataset,batch_size = 1024*12,
        shuffle = True,
        num_workers = 8
    )
    
    # 计算user和item的数量
    n_users = trn_data['user_id'].max() + 1
    n_items = trn_data['movie_id'].max() + 1
    print('n_users,n_items:',n_users, n_items)
    embedding_dim = 64 # 用户及商品的向量维度
    model = MF(n_users,n_items,embedding_dim)
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(params=model.parameters(), lr=0.1, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)
    for epoch in range(20):
        model.train()
        scheduler.step()
        train_losses = []
        for user_id,movie_id,rating in trainloader:
            rating = rating.float()
            optimizer.zero_grad()
            y = model(user_id,movie_id)
            # print(y.dtype,rating)
            loss = loss_fn(y,rating)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # eval
        model.eval()
        with torch.no_grad():
            losses = []
            for user_id,movie_id,rating in valloader:
                rating = rating.float()
                y = model(user_id,movie_id)
                loss = loss_fn(y,rating)
                losses.append(loss.item())
        print('epoch: {}, train loss:{}, val loss:{}'.format(epoch, sum(train_losses)/len(train_losses), sum(losses)/len(losses)))