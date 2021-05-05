





## SWU AI Deep Learning 

<img width="800" height = "400" src="https://user-images.githubusercontent.com/58849278/117119604-46a43680-adcd-11eb-91cc-d489f17dbef1.png">


|Language| Library |
|--|--|
| Python 2.7.16 | Tensorflow |

## GOAL 

:heavy_check_mark: 딥러닝의 지도학습, 비지도학습, 강화학습 이론 이해
:heavy_check_mark: MNIST DATA 활용
:heavy_check_mark: Linear Regression 실습
:heavy_check_mark: Gradient Descent Optimizer 사용 
:heavy_check_mark: Logisitic Regression 실습 
:heavy_check_mark: Adam Optimizer 사용 
:heavy_check_mark: 과적합 방지를 위한 DropOut 정리 및 실습, 비용함수는 cross-entropy 활용 
:heavy_check_mark: RMS Prop Optimzer과 Adagrad 함수의 비교 및 실습 
:heavy_check_mark: CNN Adam Optimizer 실습 
:heavy_check_mark: CNN의 최적화 방법으로 adam optimizer과 rmsprop optimzer 실습 후 정확도 비교 
:heavy_check_mark: 학습 모델에 적합한 비용함수를 사용하여 학습률 확인 


## 선형회귀모델

![image](https://user-images.githubusercontent.com/58849278/117121470-98e65700-adcf-11eb-9066-eb808690ba87.png)

1. 무작위의 데이터를 생성하고 .csv 파일로 만들어 준다. 
2. 4개의 심층 신경망 모델을 생성한다. 
3. 가중치와 편향의 분포는 정규분포로 초기화한다.  
4. Gradient Descent Optimization 을 최적화 방법으로 사용한다.
5. 최적화 함수로 손실을 최소화한다. 

        cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
	    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(cost)

## 실행 결과

1. MNIST Data Sets 을 사용하여 데이터를 학습하고 이를 바탕으로 사람의 키가 주어졌을 때, 근사치 값에 가까운 사람의 몸무게를 예측해 볼 수 있다. 

2. 경사하강법으로 손실을 최소화 하는 최적화 과정을 거치게 된다. 

3. 선형 회귀 모델은 모든 데이터로부터 나타나는 오차의 평균을 최소화하는 최적의 기울기와 절편을 찾는 역할을 한다. 

5. 반복된 학습을 통해 정확도는 1.0에 가까워지는 것을 볼 수 있다. 

## 심층신경망 모델

<img width="800" src="https://user-images.githubusercontent.com/58849278/117121258-515fcb00-adcf-11eb-8ec3-3a2083ce5166.png">

 1. 털과 날개에 따라 포유류, 조류, 기타로 분류하는 데이터를 무작위로 생성한다 (.csv) 
 2. placeholder 에 실측 값을 넣어준다. 
 3. 심층신경망의 입력층, 3개의 은닉층 (10,20,30개의 노드로 구성), 출력층으로 되어있는 모델을 생성한다. 

	 <img src="https://user-images.githubusercontent.com/58849278/116961945-a02e3780-acdf-11eb-8143-3441b8e6cee6.png" width = 500>

4. 최적화 방법은 Gradient Descent Optimizer을 사용한다. 

	<img src="https://user-images.githubusercontent.com/58849278/116962044-edaaa480-acdf-11eb-93b8-4caee98831eb.png" width = 500>

## 결과 분석 

:heavy_check_mark: 은닉층이 3개로 이루어진 심층 신경망은 가중치와 편향을 추가해주어야 하며 이에 따른 layer도 맞춰주어야 오류가 없이 실행된다. 

:heavy_check_mark: 입력층과 출력층은 특징과 분류 개수로 맞춰야 하며 연결 부분은 꼭 맞닿은 층의 뉴런 수와 맞춰야 한다. 

:heavy_check_mark: 데이터가 증가할 수록 학습 시간이 증가하고 초기의 데이터 생성 부분에서 개별적인 학습이 요구된다. 

:heavy_check_mark: 은닉층을 추가할 때마다 계산의 부담은 지수적으로 증가한다. 

:heavy_check_mark: 2개의 은닉층으로 불연속 함수의 표현이 가능하다. 

:heavy_check_mark: 입력 패턴에 대한 훈련 집합을 신경망에 제시하면 신경망은 출력 패턴과 목표 출력간의 오차가 줄어들도록 가중치를 조정해 나간다. 

## MNIST Batch Normalization 

****구현과정**** 
	

 - MNIST 데이터 불러오기 
 - 심층신경망의 입력층 , 3개의 은닉층 (256, 256, 256), 출력층으로 되어 있는 모델 생성 
 - 최적화 방법은 Gradient Descent Optimization 
 - 과적합 방지를 위해 Dropout 사용 
 - 과적합 방지를 위해 Batch Normalization 사용 

**Input Data** 
	

 - 입력 데이터는 원 핫 인코딩 방식으로 받아옴 
 - 28 X 28 사이즈의 입력 데이터를 사용하여 결과값인 784를 입력값으로 넣어줌
 - 0~9까지 결과값을 나타내주기 위해 10개의 결과값을 만들어줌 

1. Dropout 0.8 결과 

![image](https://user-images.githubusercontent.com/58849278/117122432-e0b9ae00-add0-11eb-9829-af9a4d3b1803.png)

2. Dropout 1.0 결과 

![image](https://user-images.githubusercontent.com/58849278/117122846-44dc7200-add1-11eb-8cc5-a1c81b86d0e0.png)

## Conclusion

 1. Dropout이 1에 가까울 때, 정확도가 높게 나오는 것을 확인
 2. 과적합을 막아주는 기법인 dropout 을 사용하면 시간이 오래 걸린다는 단점이 있지만 이를 보완해주는 배치 정규화 기법을 사용하여 학습 속도를 향상 시킬 수 있었다. 

## CNN 

**설정 환경**
	

 - MNIST DATA 
 - 합성곱 신경망은 입력층, 출력층, Convolutional Layer 2개로 구성 
 - 컨볼루션 레이어는 각각 2X2 max pooling 적용 
 - Fully connected layer 2개를 추가 구성하여 각각 28 x 28 = 256 
 - Dropout 수치는 0.8 고정
 - 비용함수는 cross-entropy 사용 
 - 최적화는 Adam optimizer과 RmsProp optimizer 사용하여 비교 

## 결과 분석 

**Adam Optimizer 적용 결과** 

![image](https://user-images.githubusercontent.com/58849278/117125515-90dce600-add4-11eb-84e6-063bfc0581fc.png)

**RMS Prop - Optimizer 적용 결과**
![image](https://user-images.githubusercontent.com/58849278/117125641-c550a200-add4-11eb-9a48-cd86e00c7734.png)

**Conclusion** 

![image](https://user-images.githubusercontent.com/58849278/117125805-faf58b00-add4-11eb-9ecb-ae9366172f3b.png)

1. Adam을 사용했을 때, 1st epoch에서 cost 값이 0.472로 나왔고 Rms prop을 사용했을 때 1.177이 나왔다. 

2. RMS prop 최적화가 완료될 때까지 cost는 처음 1.177 이후에 0.118, 0.076, 0.049, ... 0.028 까지 줄어드는 것을 볼 수 있다. 

3. Adam optimizer 최적화가 완료될 때까지 cost는 처음 0.472, 0.143, 0.101, 0.078, ... 0.022 까지 줄어드는 것을 볼 수 있다. 

4. 각 네트워크마다 특성이 다르기 때문에 어떤 최적화 함수가 좋다고 판단할 수 없지만 Adam Optimizer 가 Adagrad와 Rms Prop의 장점을 섞어 놓은 것으로 좋은 성과를 내고 있다. 
