#coding:utf-8
#0 导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 30
seed = 2
#基于seed产生随机数
rdm = np.random.RandomState(seed)
#随机数返回300行2列的矩阵，表示300组坐标点（x0，x1）作为输入数据集
X = rdm.randn(300,2)
# 从这个300行2列的矩阵中取出1行，判断如果两个坐标的平方和小于2，给Y赋值1，其余赋值0
# 作为输入数据集的标签（正确答案）
Y_ = [int(x0*x0 + x1*x1 <2) for (x0,x1) in X]
# 遍历Y中的每个元素，1赋值‘red’，其余赋值‘blue’，这样可视化显示时人可以直观区分
Y_c = [[ 'red' if y else 'blue'] for y in Y_]
# 对数据集X和标签Y进行shape整理，第一个元素用-1表示，随第二个参数计算得到，第二个元素表示多少列，
# 把X整形为n行2列，把Y整形为n行1列
X = np.vstack(X).reshape(-1,2)    #-1 表示为n行
Y_ = np.vstack(Y_).reshape(-1,1)
print(X)
print(Y_)
print(Y_c)
# 利用plot.scatter画出数据集X各行中第0列元素与第1列元素的点，即各行的（x0，x1），用各行Y_c对应的
# 值表示颜色(c是color的缩写)
plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.show()

# 定义神经网络的输入，参数和输出，定义前向传播过程
def get_weight(shape,regularizer):
    w = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01,shape=shape))
    return b

x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))

w1 = get_weight([2,11],0.01)
b1 = get_bias([11])   #b1为11个
y1 = tf.nn.relu(tf.matmul(x,w1)+b1)

w2 = get_weight([11,1],0.01)
b2 = get_bias([1])
y1 = tf.matmul(y1,w2)+b2 #输出层不过激活函数

# 定义损失函数
lose_mse = tf.reduce_mean(tf.square(y1-y_))
loss_total = lose_mse + tf.add_n(tf.get_collection('losses'))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

# 定义反向传播方法，不含正则化
with tf.Session() as sess:  # with结构中初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000  # 给出训练轮数
    for i in range(STEPS):  # 用for循环开始训练
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        # 每轮从X的数据集，以及标准答案Y_抽取相应的从start开始到end结束，Y是特征，喂入神经网络
        # 对train_step进行运算
        if i % 2000 == 0:  # 计算第一层神经网络的参数，每2000轮打印一次第一层神经网络参数的值w1
            loss_mse_v = sess.run(lose_mse,feed_dict={x:X,y_:Y_})
            print("After %d raining steps, loss is:%f" % (i,loss_mse_v))
    # xx在-3到3之间以步长为0.01，yy在-3到3之间以步长为0.01，生成二维网络坐标点
    xx,yy = np.mgrid[-3:3:0.01,-3:3:0.01]
    # 将xx，yy拉直，并合并为一个2列的矩阵，得到一个网格坐标点的集合
    grid = np.c_[xx.ravel(),yy.ravel()]
    # 将网格坐标点喂入神经网络，probs为输出
    probs = sess.run(y1,feed_dict={x:grid})
    #probs的shape调整为xx的样子
    probs = probs.reshape(xx.shape)
    print("w1:\n",sess.run(w1))
    print("b1:\n", sess.run(b1))
    print("w2:\n", sess.run(w2))
    print("b2:\n", sess.run(b2))

plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5])
plt.show()

# 定义方向传播方法
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)
with tf.Session() as sess:  # with结构中初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000  # 给出训练轮数
    for i in range(STEPS):  # 用for循环开始训练
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        # 每轮从X的数据集，以及标准答案Y_抽取相应的从start开始到end结束，Y是特征，喂入神经网络
        # 对train_step进行运算
        if i % 2000 == 0:  # 计算第一层神经网络的参数，每2000轮打印一次第一层神经网络参数的值w1
            loss_v = sess.run(loss_total,feed_dict={x:X,y_:Y_})
            print("After %d raining steps, loss is:%f" % (i,loss_v))
    # xx在-3到3之间以步长为0.01，yy在-3到3之间以步长为0.01，生成二维网络坐标点
    xx,yy = np.mgrid[-3:3:0.01,-3:3:0.01]
    # 将xx，yy拉直，并合并为一个2列的矩阵，得到一个网格坐标点的集合
    grid = np.c_[xx.ravel(),yy.ravel()]
    # 将网格坐标点喂入神经网络，probs为输出
    probs = sess.run(y1,feed_dict={x:grid})
    #probs的shape调整为xx的样子
    probs = probs.reshape(xx.shape)
    print("w1:\n",sess.run(w1))
    print("b1:\n", sess.run(b1))
    print("w2:\n", sess.run(w2))
    print("b2:\n", sess.run(b2))

plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5])
plt.show()