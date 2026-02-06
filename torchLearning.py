# 加载模块
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

# 加载模块
import torch
# =====> 第一部分: 数据
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import random_split
# =====> 第二部分: 搭建深度神经网络模型
import torch.nn as nn
from torch.nn import functional as F
# =====> 第三部分: 训练深度神经网络模型
import torch.optim as optim


def tensorGenReg(num_samples=1000, w=[2, -1, 1], bias=True, delta=0.01, degree=1):
    """回归类任务数据集创建函数:
    @param num_samples: 数据集中的样本个数
    @param w: 数据集中特征变量前的权重, 包含截距项
    @param bias: 是否包含截距项
    @param delta: 控制噪声的大小
    @param degree: 每一个特征变量的系数

    @return: 特征张量X和标签张量y组成的元组
    """
    if bias:  # => bias 等于 True:
        num_features = len(w)-1
        features_true = torch.randn(size=(num_samples, num_features), dtype=torch.float32)
        w_true = torch.tensor(w[:-1]).reshape(-1, 1).float()
        b_true = torch.tensor(w[-1]).float()
        # labels_true表示客观规律值
        if num_features == 1:
            labels_true = torch.pow(features_true, degree) * w_true + b_true
        else:
            labels_true = torch.mm(torch.pow(features_true, degree), w_true) + b_true
        # 在特征张量的最后一列添加一列全是1的列
        features = torch.cat((features_true, torch.ones(size=(len(features_true), 1))), dim=1)
        # labels表示实际观测值
        labels = labels_true + torch.randn(size=labels_true.shape) * delta
        
    else:  # => bias 等于 False
        num_features = len(w)
        features_true = torch.randn(size=(num_samples, num_features), dtype=torch.float32)
        w_true = torch.tensor(w).reshape(-1, 1).float()
        if num_features == 1:
            label_true = torch.pow(features_true, degree) * w_true
        else:
            label_true = torch.mm(torch.pow(features_true, degree), w_true)
        features = features_true
        labels = label_true + torch.randn(size=label_true.shape) * delta

    return features, labels


def tensorGenCla(num_samples=500, num_features=2, num_classes=3, deg_dispersion=[4, 2], bias=False):
    """生成基础的分类任务数据集的函数

    @param num_samples: 数据集中样本的个数
    @param num_features: 数据集中每一个样本的特征变量的个数
    @param num_classes: 数据集中每一个样本的分类的个数
    @param deg_dispersion: 各个类别标签下的数据样本点的均值和标准差的参考数值
    @param bias: 逻辑回归模型是否包含截距项

    @return: 特征张量和标签
    """
    # 每一个类别下的数据样本点的均值参考值
    mean_ = deg_dispersion[0]
    # 每一个类别下的数据样本点的标准差参考值
    std_ = deg_dispersion[1]
    # 将每一个类别下的特征张量按照顺序存储在列表中
    X_lst = []
    # 将每一个类别下的标签按照顺序存储在列表中
    y_lst = []

    # 不同的类别的数据样本点的均值不同
    k = mean_ * (num_classes-1) / 2

    for i in range(num_classes):  # 0 1 2
        # 在计算机的内存中一次性存储num_samples x num_features个随机数
        # 随机数来自于一个随机变量, 服从指定均值和制定标准差的正态分布
        X_temp = torch.normal(mean=i*mean_-k, std=std_, size=(num_samples, num_features))
        X_lst.append(X_temp)
        # 在计算机的内存中一次性存储num_samples x 1个与i相同的整数值
        y_temp = torch.full(size=(num_samples, ), fill_value=i)
        y_lst.append(y_temp)

    # 合并生成完整的数据集
    X = torch.cat(tensors=X_lst, dim=0).float()
    y = torch.cat(tensors=y_lst, dim=0).reshape(-1, 1).long()

    if bias:  # bias 等于 True
        X = torch.cat(tensors=(X, torch.ones(size=(len(X), 1))), dim=1)

    return X, y


def split_dataset(batch_size, X, y):
    """将完整的数据集(特征张量, 标签)按照相同的大小, 划分为若干个互不相交的子集
    @param batch_size: 每一个子数据集中样本的个数
    @param X: 包含所有样本的特征张量
    @param y: 包含所有样本的标签

    @return batched_dataset: 列表, 包含有一定数量的元组, 每一个元组的第一个元素是子集的特征张量, 第二个元素是子集的标签
    """
    # 数据集中样本的数量
    num_samples = len(X)
    # 获取所有样本的对应的索引
    indices = list(range(num_samples))
    # 打乱顺序排序的样本索引
    random.shuffle(indices)

    batched_dataset = []
    for i in range(0, num_samples, batch_size):  # 每一次都可以通过索引i:i+batch_size获取batch_size个样本
        sample_indices = torch.tensor(indices[i: min(i+batch_size, num_samples)], dtype=torch.int32)
        batched_dataset.append((torch.index_select(input=X, dim=0, index=sample_indices), torch.index_select(input=y, dim=0, index=sample_indices)))

    return batched_dataset


def linear_regression(X, w):
    return torch.mm(X, w)


def squared_loss(y_hat, y):
    # 计算数据集中样本的个数
    num_ = y.numel()
    # 计算SSE
    sse = torch.sum((y_hat.reshape(-1, 1) - y.reshape(-1, 1)) ** 2)
    return sse / num_


def stochastic_gradient_descent(params, lr):
    # 获取模型中所有参数的梯度信息
    # params.data = params - lr * params.grad
    params.data -= lr * params.grad
    # 清除已经使用过的梯度信息
    params.grad.zero_()

    
def sigmoid(z):  # z可以是标量scaler, 也可以是向量vector
    return 1. / (1 + torch.exp(-z))


def logistic_regression(X, w):
    # 1.整合信息
    z_hat = torch.mm(X, w)
    # 2.加工信息
    sigma = sigmoid(z_hat)
    return sigma


def classify(sigma, p=0.5):  # sigma可以是标量scaler, 也可以是向量vector
    return (sigma >= p).float()


def accuracy(sigma, y):
    """计算模型的分类准确率
    @param: sigma, 模型的预测输出结果, 预测当前样本属于各个类别的概率值
    @param: y, 样本的真实标签
    """
    accuracy_bool = classify(sigma=sigma).flatten() == y.flatten()
    accuracy = torch.mean(accuracy_bool.float())
    return accuracy


def binary_cross_entropy(sigma, y):
    """按照二分类交叉熵损失函数公式, 计算模型的预测输出标记sigma和真实标签y之间的误差
    @param sigma: 模型的预测输出标记
    @param y: 真实标签
    """
    return (-1. / len(sigma)) * torch.sum(input=y*torch.log(sigma) + (1-y)*torch.log(1-sigma))


def softmax(X, w):
    """计算特征张量X和权重向量w整合信息, 然后使用softmax函数加工信息的结果
    @param X: 特征张量
    @param w: 模型的参数
    """
    # 1.整合信息
    z_hat = torch.mm(X, w)
    # 2.加工信息
    # 2-1.计算分子
    numerator = torch.exp(z_hat)
    # 2-2.计算分母
    denominator = torch.sum(torch.exp(z_hat), 1).reshape(-1, 1)
    return numerator / denominator


def m_cross_entropy(soft_z, y):
    # 将真实标签y作为索引, 从模型的softmax函数加工信息的输出结果中取出与真实标签对应类别的预测概率
    y = y.long()
    prob_real = torch.gather(input=soft_z, dim=1, index=y)
    return (-1. / y.numel() * torch.log(prob_real).sum())


def m_accuracy(soft_z, y):
    """计算分类准确率:
    @param soft_z: 模型预测输出标记, 每一个样本属于各个类别的预测概率
    @param y: 真实标签, 每一个样本的真实标签
    """
    acc_bool = torch.argmax(input=soft_z, dim=1).flatten() == y.flatten()
    accuracy = torch.mean(acc_bool.float())
    return accuracy


class tensorGenRegDataset(Dataset):  # 以数据集的名称Dataset命名该类
    # 构造器 + 类/对象属性
    # => 子类的对象调用父类的构造器
    # => 在数据存储在类/对象数据字典中
    def __init__(self, X, y):
        self.X = X
        self.y = y
        # 计算一个完整的数据集中的样本的个数 => 标量scalar
        self.m = len(X)

    # 方法
    # => 重写__getitem__(self, index)方法
    # => 重写__len(self)__方法
    def __getitem__(self, index):
        """根据样本的编号, 返回样本的特征变量和标签
        @param index: 样本的编号
        @return 样本的特征变量和标签, 存储在元组数据集结构中
        """
        return self.X[index, :], self.y[index]

    def __len__(self):
        """返回完整数据集中样本的个数
        """
        return self.m

        
def split_loader(X, y, rate=0.7, batch_size=10):
    """
    先将一个完整的数据集按照给定的比例划分为训练数据集和测试数据集
    再将训练数据集和测试数据集分别按照给定的大小分别划分为互不相交的子集合
    @param X: 特征变量
    @param y: 标签
    @param rate: 训练数据集中的样本数占完整数据集中的样本数的比例
    @param batch_size: 用于划分子集合的数据集的大小

    @return 批处理化后的训练数据集和测试数据集
    """
    # 1.计算一个完整的数据集包含的样本个数 => 标量
    m = len(X)
    # 2.计算训练数据集中包含的样本的个数 = 完整的数据集中包含的样本的个数 * 训练数据集中的样本数占完整数据集中的样本数的比例
    m_train = int(m * rate)
    # 3.计算测试数据集中包含的样本的个数 = 完整的数据集中包含的样本的个素 - 训练数据集中的样本数
    m_test = m - m_train
    # 4.将一个完整的数据集按照给定的比例划分为训练数据集和测试数据集
    dataset = tensorGenRegDataset(X=X, y=y)
    dataset_train, dataset_test = random_split(dataset=dataset, lengths=[m_train, m_test])

    # 5.将训练数据集和测试数据集分别按照给定的大小分别划分为互不相交的子集合
    batched_dataset_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    batched_dataset_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    return batched_dataset_train, batched_dataset_test


def fit(batched_dataset, n_epochs, model, criterion, optimizer, task="reg"):
        """让模型拟合数据, 在训练过程中更新模型的参数, 使得损失函数值减小, 尽可能最小
        @param n_epochs: 对于一个完整的数据集进行学习的遍数
        @param batched_dataset: 将一个完整的数据集按照指定的样本容量划分为若干个互不相交的子集
        @param model: 模型
        @param criterion: 损失函数
        @param optimizer: 优化器

        @return: 完成训练流程的模型(不一定是性能最好的模型)
        """
        for i_epoch in range(n_epochs):
            for (i_X, i_y) in batched_dataset:
                # 1.前向传播, 计算以当前的模型参数对该批次的数据进行学习, 模型的预测输出标记
                z_hat = model.forward(i_X)
                # 2.计算损失, 计算模型的预测输出标记与真实标签之间的误差, 构建完整的计算图
                if task == "clf":
                    i_y = i_y.flatten().long()

                # 5.清空梯度信息
                optimizer.zero_grad()
                loss = criterion(z_hat, i_y)
                # 3.反向传播, 计算以当前的模型参数对应的偏导函数表达式和值, 以确定各个模型的参数在对应维度上应该向哪个方向进行更新(该维度所在的正方向or负方向)
                loss.backward()
                # 4.更新模型的参数, 优化器根据梯度信息, 计算当前的模型参数在各个维度的已知方向上应该更新的数值
                optimizer.step()


def calc_mse(dataset, model):
    """对一个完成训练流程的模型, 计算模型在训练数据集或测试数据集上的MSE
    @param dataset: 训练数据集or测试数据集
    @param model: 一个已经完成训练流程的模型
    """
    # 1.从完整的数据集中获取样本的特征变量和标签
    X = dataset.dataset[:][0]
    y = dataset.dataset[:][1]
    # 2.前向传播, 计算模型以当前的参数对数据集进行学习后的预测输出标记
    z_hat = model.forward(X)
    # 3.计算损失, 计算模型的预测输出标记和真实标签之间的误差
    loss = F.mse_loss(z_hat, y)
    return loss


def calc_accuracy(dataset, model):
    """对一个完成训练的模型, 计算模型在训练数据集或者测试数据集上的分类准确度
    @param dataset: 训练数据集或者测试数据集
    @param model: 完成训练的模型
    @return 模型在训练数据集或者测试数据集上的分类准确度
    """
    # 1.计算数据集中样本的总数
    # m = len(dataset.dataset[:][0])
    # 2.计算模型在当前数据集上的预测输出标记
    # model.forward(dataset.data[:][0])

    # 获取数据集中的特征变量
    X = dataset.dataset[:][0]
    # 获取数据集中的标签
    y = dataset.dataset[:][1]
    # 计算模型在当前数据集上的预测输出标记
    z_hat = model.forward(X)
    sigma = F.softmax(z_hat, dim=1)
    is_correct = torch.argmax(input=sigma, dim=1).flatten() == y.flatten()
    # 计算分类准确度
    accuracy = torch.mean(is_correct.float())
    
    return accuracy


def calc_train_test_losses(dataset_train, dataset_test, n_epochs, model, criterion, optimizer, task="reg", evaluation=calc_mse):
    """记录模型在训练过程中在训练数据集和测试数据集的性能表现
    @param n_epochs: 对完整的数据集学习的遍数
    @param dataset_train: 训练数据集
    @param dataset_test: 测试数据集
    @param model: 参数初始化后的模型
    @param criterion: 选择合适的损失函数
    @param optimizer: 选择合适的优化器
    @param task: 任务类型

    @return: 模型在每一遍的训练过程中在训练数据集和测试数据集上的模型性能评估指标
    """
    # 在计算机的内存中逐个存储多个数值 => 选择列表数据结构
    losses_train = []
    losses_test = []

    for i_epoch in range(n_epochs):
        # 在每一遍完整学习训练数据集的之前开启模型的训练模式
        model.train()
        # 模型进行训练
        fit(batched_dataset=dataset_train, n_epochs=i_epoch, model=model, criterion=criterion, optimizer=optimizer, task=task)

        # 在每一遍完整学习训练数据集的之后关闭模型的训练模式, 开启模型的测试模式
        model.eval()
        # 对当前完成训练的模型分别计算在训练数据集和测试数据集上的模型性能评估指标
        losses_train.append(evaluation(dataset=dataset_train, model=model).detach())
        losses_test.append(evaluation(dataset=dataset_test, model=model).detach())

    return losses_train, losses_test


def compare_models_performance(dataset_train, dataset_test, n_epochs, models, model_names, criterion=nn.MSELoss(), optimizer=optim.SGD, lr=0.03, task="reg", evaluation=calc_mse):
    """计算同一个任务的不同模型的性能评估指标
    @param dataset_train: 训练数据集
    @param dataset_test: 测试数据集
    @param n_epochs: 对完整的数据集学习的遍数
    @param models: 参数初始化后的多个模型
    @param model_names: 参数初始化后的多个模型的名称
    @param criterion: 选择合适的损失函数
    @param optimizer: 选择合适的优化器
    @param task: 任务类型
    @param evaluation: 在给定任务下的模型性能评估指标计算方法

    @return: 各个模型在每一遍的训练过程中在训练数据集和测试数据集上的模型性能评估指标
    """
    # 在计算机的内存中逐个存储多个数值, 选择张量数据结构 => 张量的维度, 形状和数据类型
    mse_train = torch.zeros(size=(len(models), n_epochs), dtype=torch.float32)
    mse_test = torch.zeros(size=(len(models), n_epochs), dtype=torch.float32)
    
    for epochs in range(n_epochs):
        for i, model in enumerate(models):
            # 在每一遍完整学习训练数据集的之前开启模型的训练模式
            model.train()
            # 模型进行训练
            opt = optimizer(params=model.parameters(), lr=lr)
            fit(batched_dataset=dataset_train, n_epochs=epochs, model=model, criterion=criterion, optimizer=opt, task=task)
            # 在每一遍完整学习训练数据集的之后关闭模型的训练模式, 开启模型的测试模式
            model.eval()
            mse_train[i][epochs] = evaluation(dataset=dataset_train, model=model).detach()
            mse_test[i][epochs] = evaluation(dataset=dataset_test, model=model).detach()

    return mse_train, mse_test


def plot_violin_param(model, param="grad"):
    """绘制模型中的各个层上的指定参数的小提琴分布图
    @param model: 模型
    @param param: 指定参数的名称. e.g. "weight" "grad"(默认值) "bias"
    """
    # 在计算机的内存中逐个存储多个数值 => 选择列表数据结构
    vp = []
    # 获取完整的模型和单独的层, 然后仅保留单独的层
    layers = list(model.modules())[1:]
    for i, layer in enumerate(layers):
        # 获取当前层的指定参数. param有三种可能的取值
        if param == "grad":
            vp_param = layer.weight.grad.detach().reshape(-1, 1).numpy()
        elif param == "weight":
            vp_param = layer.weight.detach().reshape(-1, 1).numpy()
        else:  # param == bias
            vp_param = layer.bias.detach().reshape(-1, 1).numpy()
        
        # 获取当前层的索引
        vp_index = np.full_like(vp_param, i+1)
        vp_i_layer = np.concatenate((vp_param, vp_index), 1)
        vp.append(vp_i_layer)
        
    vp = np.concatenate((vp), 0)
    # 绘制小提琴图
    ax = sns.violinplot(x=vp[:, 1], y=vp[:, 0])
    ax.set(xlabel="#hidden layer", title=param)