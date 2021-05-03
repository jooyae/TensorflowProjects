
import tensorflow as tf 
import numpy as np


#키와 몸무게 20개 생성 
#키는 m단위, 몸무게는 kg단위로 지정

x_data= [[1.5],[1.52],[1.53],[1.55],[1.6],
           [1.62],[1.63],[1.64],[1.65],[1.67],
           [1.68],[1.7],[1.72],[1.73],[1.75],
           [1.76],[1.79],[1.80],[1.82],[1.85]]

y_data = [[40],[41],[42],[43],[44],
           [46],[47],[48],[49],[50],
           [52],[65],[67],[56],[70],
           [72],[73],[74],[78],[80]]

#가중치와 편향을 균등분포를 가진 값으로 초기화해줌 
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

#실제값을 입력받을 placeholder 설정 
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

hypothesis = W * X + b

#손실함수 
#최적화 함수로 경사하강법 gradientdescentoptimizer 사용해주기 
#손실값 최소화해주는 역할 
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.13)
train_op = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#최적화를 수행하는 trian_op를 실행해주고 실행시 매번 손실값을 출력받게 해줌 
for step in range(100):
    _, cost_val = sess.run([train_op,cost],feed_dict = {X: x_data,Y: y_data})
    print(step, cost_val, sess.run(W), sess.run(b))

#마지막으로 키가 160과 170일 때 값을 출력해보기 
print("X: 1.55, Y:", sess.run(hypothesis, feed_dict={X:1.55}))
print("Y: 1.7, Y:", sess.run(hypothesis, feed_dict={X:1.7}))
