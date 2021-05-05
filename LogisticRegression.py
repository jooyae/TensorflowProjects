import tensorflow as tf 
import numpy as np

#[털, 날개]
x_data = np.array( [[0,0],[1,0],[1,1],[0,0],[0,0],[0,1],
                    [0,0],[1,0],[1,1],[0,0],[0,0],[0,1],
                    [0,0],[1,0],[1,1],[0,0],[0,0],[0,1],
                    [0,0],[1,0]])
#기타, 포유류, 조류 학습시켜주기 
y_data = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[1,0,0],[0,0,1],
                   [1,0,0],[0,1,0],[0,0,1],[1,0,0],[1,0,0],[0,0,1],
                   [1,0,0],[0,1,0],[0,0,1],[1,0,0],[1,0,0],[0,0,1],
                   [1,0,0],[0,1,0]]) 

#placeholder에 실측값 넣어주기 
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#가중치와 편향

#첫번째 가중치의 차원은 [특성, 히든 레이어의 뉴런갯수]
W1 = tf.Variable(tf.random_uniform([2,10],-1.,1.))
#두번째 가중치의 차원은 [첫번째 히든 레이어의 뉴런갯수,두번째 히든 레이어의 뉴런갯수 ]
W2 = tf.Variable(tf.random_uniform([10,20],-1.,1.))
#세번째 가중치의 차원은 [두번째 히든 레이어의 뉴런갯수,세번째 히든 레이어의 뉴런갯수]
W3 = tf.Variable(tf.random_uniform([20,10],-1.,1.))
#네번째 가중치의 차원은 [세번째 히든 레이어의 뉴런갯수, 분류]
W4 = tf.Variable(tf.random_uniform([10,3],-1.,1.))

#b1,b2,b3은 히든레이어의 뉴런갯수
#b4는 최종분류값=3
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([20]))
b3 = tf.Variable(tf.zeros([10]))
b4 = tf.Variable(tf.zeros([3]))

#LAYER 추가 
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))

model = tf.add(tf.matmul(L3, W4), b4)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range (100):
    sess.run(train_op, feed_dict = {X:x_data, Y:y_data})
    
    if(step + 1 )% 10 == 0:
        print(step + 1, sess.run(cost,feed_dict={X:x_data, Y:y_data}))
prediction = tf.argmax(model, 1)
target = tf.argmax(Y,1)

print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
