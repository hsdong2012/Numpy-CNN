import math
import numpy as np
from scipy.io import loadmat

def onehot(targets, depth):
    """
    对label进行one-hot编码

    :param targets: [num, 1], 标记样本对应的类别
    :param depth: 编码的深度

    :return: [num, depth], one-hot编码后的标签矩阵
    """

    return np.eye(depth)[targets]


def img2col(x, ksize, stride):
    """
    对输入图像x，根据[ksize, ksize]大小的卷积核变换，即使用卷积核在图像上滑动，
    将滑动窗口内的图像patch转换为向量, 这样可以加速卷积运算

    :param x: [wx, hx, cx] 输入图像
    :param ksize: 卷积核大小，这里卷积核高、宽相等
    :param stride: 移动步长

    :return: image_col卷积核在图像每次移动时窗口内patch的图像转换为一个向量
    """
    wx, hx, cx = x.shape                     # [width,height,channel]

    feature_w = (wx - ksize) // stride + 1   # 返回的特征图尺寸

    image_col = np.zeros((feature_w*feature_w, ksize*ksize*cx))

    num = 0
    for i in range(feature_w):
        for j in range(feature_w):
            image_col[num] =  x[i*stride : i*stride+ksize, j*stride : j*stride+ksize, :].reshape(-1)

            num += 1
    return image_col


def generate_batches(x, y, batch_size, seed):
    """
    生成批次样本
    
    :param x:[#num, #dim] 按行组织的样本矩阵
    :param y:[#num, #classes] 按行组织的标签矩阵
    :param batch_size: scalar, 批次中的样本数
    :param seed: scalar, 随机数种子

    :return:mini_batches, 批次样本列表, 列表每个元素为一个批次的样本矩阵
    """

    nx, hx, wx, cx = x.shape

    np.random.seed(seed)
    mini_batches = []

    # 打乱样本的下标
    permutation = list(np.random.permutation(nx))
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation, :]

    # 完整的批次
    num_complete_batches = math.floor( nx / batch_size)

    for k in range(0, num_complete_batches):
        batch_x = shuffled_x[k * batch_size:(k + 1) * batch_size]
        batch_y = shuffled_y[k * batch_size:(k + 1) * batch_size]

        mini_batches.append((batch_x, batch_y))

    # 数量不足的一个批次
    if nx % batch_size != 0:
        batch_x = shuffled_x[num_complete_batches * batch_size]
        batch_y = shuffled_y[num_complete_batches * batch_size]

        mini_batches.append((batch_x, batch_y))

    return mini_batches


class Linear(object):
    """全连接层

    参数注意!
    self.W -- [n_{l},n_{l+1}]
    self.b -- [n_{l+1}]
    self.x -- [batchsize, n{l}], 注意按行组织!
    """

    def __init__(self, inChannel, outChannel):
        scale = np.sqrt(inChannel/2)

        # 使用标准正态分布对权重与偏置向量进行初始化
        self.W = np.random.standard_normal((inChannel, outChannel)) / scale
        self.b = np.random.standard_normal(outChannel) / scale

        # W的梯度矩阵与W同形，b的梯度也与b同形
        self.W_grad = np.zeros((inChannel, outChannel))
        self.b_grad = np.zeros(outChannel)

    def forward(self, x):
        self.x = x
        z = np.dot(self.x, self.W) + self.b

        return z

    def backward(self, delta, learning_rate):
        batch_size = self.x.shape[0]

        self.W_grad = np.dot(self.x.T, delta) / batch_size      # [n_{l}, batchsize] * [batchsizen, n_{l+1}] -> [n_{l},n_{l+1}]
        self.b_grad = np.sum(delta, axis=0) / batch_size        # [batchsizen, n_{l+1}], 列向上平均

        delta_backward = np.dot(delta, self.W.T)                 # [batchsizen, n_{l+1}] * [n_{l+1}, n_{l}]

        ## 梯度下降
        self.W -= self.W_grad * learning_rate
        self.b -= self.b_grad * learning_rate

        return delta_backward

class Conv(object):
    """卷积层"""

    def __init__(self, kernel_shape, stride=1, pad=0):
        wk, hk, n_in, n_out = kernel_shape

        self.stride = stride
        self.pad = pad

        scale = np.sqrt(3*wk*hk*n_in/n_out)
        self.k = np.random.standard_normal(kernel_shape) / scale
        self.b = np.random.standard_normal(n_out) / scale

        self.k_grad = np.zeros(kernel_shape)
        self.b_grad = np.zeros(n_out)

    def forward(self, x):
        self.x = x

        if self.pad != 0:
            self.x = np.pad(self.x, ((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)), 'constant')

        nx, wx, hx, cx = self.x.shape
        wk, hk, n_in, n_out = self.k.shape             # kernel的宽、高、通道数、个数

        assert cx==n_in, "channels of filter and image not match!" # cx必须与n_in相同!

        feature_w = (wx - wk) // self.stride + 1  # 返回的特征图尺寸
        feature = np.zeros((nx, feature_w, feature_w, n_out))

        self.image_col = []
        kernel = self.k.reshape(-1, n_out) # 转换为多个列向量，使用向量点乘代替循环的卷积操作，提升运算效率

        for i in range(nx):
            image_col = img2col(self.x[i], wk, self.stride)
            feature[i] = (np.dot(image_col, kernel)+self.b).reshape(feature_w, feature_w, n_out)

            self.image_col.append(image_col)

        return feature

    def backward(self, delta, learning_rate):
        nx, wx, hx, cx = self.x.shape # batch,12,142,inchannel
        wk, hk, n_in, n_out = self.k.shape # 5,5,inChannel,outChannel
        bd, wd, hd, cd = delta.shape  # batch,10,10,outChannel

        # 计算self.k_grad,self.b_grad
        delta_col = delta.reshape(bd, -1, cd)

        for i in range(nx):
            self.k_grad += np.dot(self.image_col[i].T, delta_col[i]).reshape(self.k.shape)

        self.k_grad /= nx

        self.b_grad += np.sum(delta_col, axis=(0, 1))
        self.b_grad /= nx

        # 计算delta_backward
        delta_backward = np.zeros(self.x.shape)

        k_180 = np.rot90(self.k, 2, (0,1))      # numpy矩阵旋转180度, 2表示rot 90度的2倍，即180度, 范围为(0,1)二个维度确定的平面
        k_180 = k_180.swapaxes(2, 3)            # 交换第2、3两个维度
        k_180_col = k_180.reshape(-1,n_in)      # 转换为向量，为使用向量点乘代替卷积做准备

        if hd-hk+1 != hx:
            pad = (hx-hd+hk-1) // 2
            pad_delta = np.pad(delta, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant')
        else:
            pad_delta = delta

        for i in range(nx):
            pad_delta_col = img2col(pad_delta[i], wk, self.stride)
            delta_backward[i] = np.dot(pad_delta_col, k_180_col).reshape(wx,hx,n_in)

        # 梯度下降
        self.k -=  self.k_grad * learning_rate
        self.b -=  self.b_grad * learning_rate

        return delta_backward

class Pool(object):
    """最大池化层"""

    def forward(self, x):
        n, w, h, c = x.shape

        feature_w = w // 2
        feature = np.zeros((n, feature_w, feature_w, c))

        self.feature_mask = np.zeros((n, w, h, c))   # 记录池化时最大值的位置信息，在反向传播时需要使用

        for ni in range(n):
            for ci in range(c):
                for i in range(feature_w):
                    for j in range(feature_w):
                        feature[ni, i, j, ci] = np.max(x[ni,i*2:i*2+2,j*2:j*2+2,ci])

                        index = np.argmax(x[ni,i*2:i*2+2,j*2:j*2+2,ci])  # 对矩阵的argmax返回的是按行排列后元素的下标值
                        self.feature_mask[ni, i*2+index//2, j*2+index%2, ci] = 1  # //2获得所在的行，%2可以获得所在的列

        return feature

    def backward(self, delta):
        return np.repeat(np.repeat(delta, 2, axis=1), 2, axis=2) * self.feature_mask


class Relu(object):
    """ReLU激活层"""

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, delta):
        delta[self.x<0] = 0
        return delta


class Softmax(object):
    """softmax层"""

    def cal_loss(self, predict, label):
        batchsize, classes = predict.shape
        self.predict(predict)              # 尽管没有返回值，但是在predict中赋给了属性变量self.softmax,下面依然可以调用

        loss = 0
        delta = np.zeros(predict.shape)

        for i in range(batchsize):
            delta[i] = self.softmax[i] - label[i]
            loss += -1*np.sum(np.log(self.softmax[i]) * label[i])

        loss /= batchsize

        return loss, delta

    def predict(self, predict):
        batchsize, classes = predict.shape
        self.softmax = np.zeros(predict.shape)

        for i in range(batchsize):
            predict_tmp = predict[i] - np.max(predict[i]) # 减去最大值，可以避免实际运算中的数据溢出
            predict_tmp = np.exp(predict_tmp)

            self.softmax[i] = predict_tmp / np.sum(predict_tmp) # 归一化

        return self.softmax

def train():
    mnist = loadmat('mnist_uint8.mat') # 使用裁剪的MNIST数据集进行训练测试, 训练集:200样本，测试集20样本

    train_x = mnist['train_x']  # [200,784]
    train_y = mnist['train_y']  # [200, 10]
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1) / 255.   # 输入向量处理

    conv1 = Conv(kernel_shape=(5,5,1,6))   # 24x24x6, [28-5+2*0]/1 + 1 = 24
    relu1 = Relu()
    pool1 = Pool()                         # 12x12x6, (24 -2)/2 + 1 = 12
    conv2 = Conv(kernel_shape=(5,5,6,16))  # 8x8x16, [12 - 5 +2*0] + 1 = 8
    relu2 = Relu()
    pool2 = Pool()                         # 4x4x16, 8/2=4
    nn = Linear(256, 10)
    softmax = Softmax()

    lr = 0.01
    batch = 5

    for epoch in range(10):
        # 生成打乱的mini batch
        mini_batches = generate_batches(train_x, train_y, 5, epoch)

        for i, (batch_x, batch_y) in enumerate(mini_batches):
            predict = conv1.forward(batch_x)
            predict = relu1.forward(predict)
            predict = pool1.forward(predict)

            predict = conv2.forward(predict)
            predict = relu2.forward(predict)
            predict = pool2.forward(predict)

            predict = predict.reshape(batch, -1) # batchsize行，每行为一张图像的feature
            predict = nn.forward(predict)

            loss, delta = softmax.cal_loss(predict, batch_y)

            delta = nn.backward(delta, lr)
            delta = delta.reshape(batch, 4, 4, 16)
            delta = pool2.backward(delta)
            delta = relu2.backward(delta)
            delta = conv2.backward(delta, lr)
            delta = pool1.backward(delta)
            delta = relu1.backward(delta)
            conv1.backward(delta, lr)       # 第一个卷积层，不需要再记录delta了

            print("Epoch-{}-{:05d}".format(str(epoch), i), ":", "loss:{:.4f}".format(loss))

        lr *= 0.95**(epoch+1)  # 学习率指数衰减
        np.savez("data2.npz", k1=conv1.k, b1=conv1.b, k2=conv2.k, b2=conv2.b, w3=nn.W, b3=nn.b)

def test():
    r = np.load("data2.npz")                                     # 载入训练好的参数
    mnist = loadmat('mnist_uint8.mat')

    test_x = mnist['test_x']                                     # [50,784]
    test_y = mnist['test_y']                                      # [50, 10]
    test_x = test_x.reshape(len(test_x), 28, 28, 1) / 255.      # 归一化

    conv1 = Conv(kernel_shape=(5, 5, 1, 6))  # 24x24x6
    relu1 = Relu()
    pool1 = Pool()  # 12x12x6

    conv2 = Conv(kernel_shape=(5, 5, 6, 16))  # 8x8x16
    relu2 = Relu()
    pool2 = Pool()  # 4x4x16

    nn = Linear(256, 10)
    softmax = Softmax()

    conv1.k = r["k1"]
    conv1.b = r["b1"]
    conv2.k = r["k2"]
    conv2.b = r["b2"]

    nn.W = r["w3"]
    nn.n = r["b3"]

    num = 0
    for i in range(len(test_x)):
        X = test_x[i]
        X = X[np.newaxis, :]
        Y = test_y[i]

        predict = conv1.forward(X)
        predict = relu1.forward(predict)
        predict = pool1.forward(predict)

        predict = conv2.forward(predict)
        predict = relu2.forward(predict)
        predict = pool2.forward(predict)

        predict = predict.reshape(1, -1)
        predict = nn.forward(predict)

        predict = softmax.predict(predict)

        if np.argmax(predict) == np.argmax(Y):
            num += 1

    print("TEST-ACC: ", num/len(test_x)*100, "%")

if __name__ == '__main__':
    train()
    test()
