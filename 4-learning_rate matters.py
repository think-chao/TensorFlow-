# 设损失函数 loss=(w+1)^2,另w初值是常数5。反向传播就是求最优的w，即求最小的loss对应的w值
# 0 导入模块，生成数据集
import tensorflow as tf
 
# 定义待优化参数初值为5
w = tf.Variable(tf.constant(5,dtype=tf.float32))
 
#定义损失函数loss
loss = tf.square(w+1)
 
#定义反向传播方法
# learning_rate太大则loss不稳定
"""
learning_rate = 1
After 0 raining steps, w1 is: -7.000000,loss is 36.000000
After 1 raining steps, w1 is: 5.000000,loss is 36.000000
After 2 raining steps, w1 is: -7.000000,loss is 36.000000
After 3 raining steps, w1 is: 5.000000,loss is 36.000000
After 4 raining steps, w1 is: -7.000000,loss is 36.000000
After 5 raining steps, w1 is: 5.000000,loss is 36.000000
"""
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
 
# 生成会话，训练STEPS轮
with tf.Session() as sess:  # with结构中初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40  # 给出训练轮数
    for i in range(STEPS):  # 用for循环开始训练
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %s raining steps, w1 is: %f,loss is %f" % (i,w_val,loss_val))