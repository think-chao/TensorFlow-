
# =================================================
# 1. 优化损失函数，根据实际情况设计自己的损失函数
#   例如：酸奶问题中，如果预测量大于实际销量则会损失成本，小于实际销量则会损失利润
#         假设酸奶的成本是一瓶1元，利润是卖出一瓶8元，那么我们宁愿我们的预测模型的预测量大于实际销量
#         所以我们在设计损失函数的时候，就不应该对预测多和预测少同等看待，而是对其分配不同的权重
#   tf.where(condition, a, b)
#   最后得到 Final w1 is 
#       [[1.01746  ]
#       [1.0445641]]
# 
#2. 交叉熵表示两个分布的距离： 
#    模型首先通过softmax获得每一个类别的概率，然后通过概率计算交叉熵得到损失函数
#    crossEntropy = tf.reduce_mean(
#    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1)) 
# =================================================


# 预测多或预测少的影响一样，即预测多与预测少给的权重相同
# 0 导入模块，生成数据集
import tensorflow as tf
import numpy as np
 
BATCH_SIZE = 8  # 一次喂入神经网络的一小撮特征是8个
SEED = 23455  # 随机种子是23455，保证每次生成的数据集相同
COST, PROFIT = 1, 8
 
rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)  # 生成32行个0~1之间的随机数，包括X1和X2，即32行2列的数据集
Y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X]  # 取出每组的x1与x2求和，
# 再加上随机噪声，构建标准答案Y_,.rand()函数会生成0~1之间的开区间随机数，除以10变成0~0.1
# 之间的随机数，再减去0.05再变成-0.05到+0.05之间的随机数
 
# 1定义神经网络的输入，参数和输出，定义前向传播过程。
# 给神经网络的输入x,y_用tf.placeholder占位
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))  # 定义w1参数
y = tf.matmul(x, w1)  # 定义输出y
 
# 2 定义损失函数及反向传播方法
# 定义损失函数为MSE，反向传播方法为梯度下降,学习率为0.001，让均方误差向减小的方向优化
#lose_mse = tf.reduce_mean(tf.square(y_ - y))
lose_mse = tf.reduce_mean(tf.where(tf.greater(y, y_), (y-y_)*COST, (y_-y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(lose_mse)
 
# 3 生成会话，训练STEPS轮
with tf.Session() as sess:  # with结构中初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000  # 给出训练轮数
    for i in range(STEPS):  # 用for循环开始训练
        start = (i * BATCH_SIZE) % 32
        end = (i * BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        # 每轮从X的数据集，以及标准答案Y_抽取相应的从start开始到end结束，Y是特征，喂入神经网络
        # 对train_step进行运算
        if i % 500 == 0:  # 计算第一层神经网络的参数，每500轮打印一次第一层神经网络参数的值w1
            print("After %d raining steps, w1 is:" % (i))
            print(sess.run(w1), "\n")
    print("Final w1 is", "\n", sess.run(w1))