import torch

# 构造bn层
bn = torch.nn.BatchNorm2d(num_features=3)

# 打印bn层的参数
print('#'*10, '初始化设置参数', '#'*10)
print('num_features:', bn.num_features)
print('eps:', bn.eps)
print('momentum:', bn.momentum)
print('affine:', bn.affine) # 只有当该项为真时，bn才有可学习参数
print('track_running_stats:', bn.track_running_stats) # 该项为真，表示mean和var用滑动平均
print('training:', bn.training) # 用model.train()和model.eval()对此参数进行更改
print('#'*10, '随着epoch更新的参数', '#'*10)
print('running_mean', bn.running_mean)
print('running_var', bn.running_var)
print('num_batches_tracked', bn.num_batches_tracked)
print('#'*10, '可训练参数', '#'*10)
for parm in bn.named_parameters():
    print(parm, '#shape:', parm[1].shape)

'''
########## 初始化设置参数 ##########
num_features: 3
eps: 1e-05
momentum: 0.1
affine: True
track_running_stats: True
training: True
########## 随着epoch更新的参数 ##########
running_mean tensor([0., 0., 0.])
running_var tensor([1., 1., 1.])
num_batches_tracked tensor(0)
########## 可训练参数 ##########
('weight', Parameter containing:
tensor([1., 1., 1.], requires_grad=True)) #shape: torch.Size([3])
('bias', Parameter containing:
tensor([0., 0., 0.], requires_grad=True)) #shape: torch.Size([3])
'''
# 测试输入数据
for i in range(4):
    print('#'*10, 'batch num:', i)
    x = torch.randn(4,3,5,5)
    old_running_mean = bn.running_mean.clone()
    old_running_var = bn.running_var.clone()
    x_hat = bn(x)
    print('running_mean', bn.running_mean)
    print('running_var', bn.running_var)
    print('num_batches_tracked', bn.num_batches_tracked)
    print('验证动量方法的正确性: ',
        torch.mean(x[:,0,:,:])*bn.momentum+old_running_mean[0]*(1-bn.momentum)-bn.running_mean[0]<1e-5,
        torch.var(x[:,0,:,:])*bn.momentum+old_running_var[0]*(1-bn.momentum)-bn.running_var[0]<1e-5)
'''
########## batch num: 0
running_mean tensor([ 0.0266, -0.0081,  0.0009])
running_var tensor([0.9974, 0.9993, 0.9908])
num_batches_tracked tensor(1)
验证动量方法的正确性:  tensor(True) tensor(True)
########## batch num: 1
running_mean tensor([ 0.0246,  0.0010, -0.0116])
running_var tensor([1.0029, 0.9662, 0.9923])
num_batches_tracked tensor(2)
验证动量方法的正确性:  tensor(True) tensor(True)
########## batch num: 2
running_mean tensor([ 0.0180, -0.0015, -0.0075])
running_var tensor([1.0215, 0.9766, 1.0045])
num_batches_tracked tensor(3)
验证动量方法的正确性:  tensor(True) tensor(True)
########## batch num: 3
running_mean tensor([0.0146, 0.0008, 0.0048])
running_var tensor([1.0314, 0.9749, 0.9797])
num_batches_tracked tensor(4)
验证动量方法的正确性:  tensor(True) tensor(True)
'''
print('验证测试阶段：mean和var保持不变')
bn.eval()
for i in range(4):
    print('#'*10, 'batch num:', i)
    x = torch.randn(4,3,5,5)
    x_hat = bn(x)
    print('running_mean', bn.running_mean)
    print('running_var', bn.running_var)
    print('num_batches_tracked', bn.num_batches_tracked)
'''
########## batch num: 0
running_mean tensor([0.0146, 0.0008, 0.0048])
running_var tensor([1.0314, 0.9749, 0.9797])
num_batches_tracked tensor(4)
########## batch num: 1
running_mean tensor([0.0146, 0.0008, 0.0048])
running_var tensor([1.0314, 0.9749, 0.9797])
num_batches_tracked tensor(4)
########## batch num: 2
running_mean tensor([0.0146, 0.0008, 0.0048])
running_var tensor([1.0314, 0.9749, 0.9797])
num_batches_tracked tensor(4)
########## batch num: 3
running_mean tensor([0.0146, 0.0008, 0.0048])
running_var tensor([1.0314, 0.9749, 0.9797])
num_batches_tracked tensor(4)
'''
