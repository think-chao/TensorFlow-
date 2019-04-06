#coding=utf-8
"""
1. tensorflow基本概念：
	a 用张量表示数据，用计算图搭建NN，用会话执行计算图，优化权重得到模型
"""

import tensorflow as tf 
a = tf.constant([[1, 2, 3]])
b = tf.constant([[1], [2], [3]])

result = tf.matmul(a, b)
# 计算图：只搭建神经网络，不运行运算，不计算结果。
# print(result)
# Tensor("MatMul:0", shape=(1, 1), dtype=int32)

with tf.Session() as sess:
	print(sess.run(result))
	sess.close()

"""
2. 神经网络的参数
	a 用 tf.Variable()表示
	b 常见的参数初始化方法：
		tf.random_normal() 生成正太分布随机数
		tf.truncated_normal() 生成截断过大偏离点的正太随机数
		tf.random_uniform() 生成均匀分布随机数
"""

w = tf.Variable(tf.random_normal([2, 3], stddev=2, mean=0, seed=1))
# 输出为 <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32_ref>

"""
3. 神经网络的搭建
	a 准备数据集
	b 搭建网络，即构建计算图，再执行会话
	c 前向传播，计算输出，根据优化算法反向传播更新权重
例1：
	一批零件，体积为x1, 重量x2, 这两个就是我们的特征，通过这两个特征的值我们需要预测输出的值
	搭建一个有1个隐藏层(隐藏层包含3个hidden unit)，1个输出层（输入层不算）的神经网络
	则每一层都有一个参数
"""
# 一、初始化输入以及变量
x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, mean=0, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, mean=0, seed=1))

# 二、定义前向传播过程，即构建计算图
hidden_output = tf.matmul(x, w1)
output = tf.matmul(hidden_output, w2)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	print(sess.run(w1))
	print(sess.run(output, feed_dict={
		x:[[0.7, 0.5]]
		}))
# 输出 w1:[[-0.8113182 输出 w1:  1.4845988   0.06532937]
# 		  [-2.4427042   0.09924		  84   0.591224#3 ]]
#      output: [[3.0904665]]

"""
3. 反向传播，训练模型参数，对所有参数进行梯度下降更新，使得损失函数最小
	a 常见损失函数及表示
		均方误差MSE： loss = tf.reduce_mean(tf.square(y_ - y))
	b 反向传播训练方法：
		梯度下降： gsd = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
		momentum优化器 momentum = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
		adam优化器 adam = tf.train.Adam(learning_rate).minimize(loss)
"""









