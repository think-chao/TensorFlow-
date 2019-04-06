"""
1. 构建NN的八股
	a 导入数据
	b 定义输入，参数
	c 定义损失函数，反向传播方法
	d 生成会话，开始训练
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			EPOCHES = 200
			for i in range(EPOCHES):
				sess.run(train, feed_dict={})
"""
import tensorflow as tf
import numpy as np 

BATCH_SIZE = 8
EPOCHES = 3000
seed = 2355
learning_rate = 0.001

rng = np.random.RandomState(seed)
# 表示10组两个特征的数据
X = rng.rand(32, 2)
#print(X)
"""
[[0.20518377 0.6247016 ]
 [0.98575016 0.28036245]
 [0.9563865  0.93030728]
 [0.96710297 0.65036802]
 [0.63935099 0.01533441]
 [0.60944392 0.66054765]
 [0.13150912 0.58272759]
 [0.2312768  0.9226707 ]
 [0.78820518 0.48611991]
 [0.6937683  0.26882106]]
 """
Y = [[int(x0+x1 < 1)] for (x0, x1) in X]
#print(Y)
"""
[[1], [0], [0], [0], [1], [0], [1], [0], [0], [1]]
"""

# 1.定义网络的输入，参数，输出
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, mean=0, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, mean=0, seed=1))

# 2. 定义计算图
output = tf.matmul(tf.matmul(x, w1), w2)

# 3. 定义损失和优化方法
loss = tf.reduce_mean(tf.square(output-y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 4. 执行会话，开始训练
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print('没有经过训练的参数值w1:{}'.format(sess.run(w1)))
	print('没有经过训练的参数值w2:{}'.format(sess.run(w2)))
	print('\n')

	for i in range(EPOCHES):
		start = (i*BATCH_SIZE)%32
		end = start + BATCH_SIZE

		sess.run(
			optimizer,
			feed_dict={
			x: X[start:end],
			y_: Y[start:end]
			})
		if i % 500 == 0:
			total_loss = sess.run(
				loss,feed_dict={x:X, y_:Y}
				)
			print('After {} training steps, loss on all data is {}'.format(i, total_loss))

	print('训练后的参数值w1:{}'.format(sess.run(w1)))
	print('训练后的参数值w2:{}'.format(sess.run(w2)))




