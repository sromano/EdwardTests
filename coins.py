import tensorflow as tf
import edward as ed
from edward.models import Bernoulli, Beta, Binomial, Empirical, Normal
import matplotlib.pyplot as plt
import seaborn as sns

##Single coin, multiple tosses weight inference

##Model:
theta = Beta(1.0, 1.0, sample_shape=(1,))
x = Bernoulli(probs=tf.ones(10)*theta)
#x = Binomial(total_count=5, probs=theta) #Sampling not implemented in tf
print(theta.shape)

##Sampling:
# with tf.Session() as sess:
#     for i in range(10):
#         print(x.eval())

##Observations:
#data=tf.ones(10, dtype=tf.int32) #NOT WORKING!
data=[1,1,1,1,1,1,1,1,0,1]

##Infer:

#Variational
#qtheta = Beta(tf.Variable(1.0), tf.Variable(1.0))  #Why need tf.Variable here?
# inference = ed.KLqp({theta: qtheta}, {x: data})
# inference.run(n_samples=5, n_iter=1000)

#MonteCarlo
T=10000
qtheta = Empirical(params=tf.Variable(0.5+tf.zeros([T,1])))#Beta(tf.Variable(1.0), tf.Variable(1.0))  #Why need tf.Variable here?
#proposal_theta = Beta(concentration1=1.0, concentration0=1.0, sample_shape=(1,))
#proposal_theta = Normal(loc=theta,scale=0.5)
#inference = ed.MetropolisHastings({theta: qtheta}, {theta: proposal_theta}, {x: data})
inference = ed.HMC({theta: qtheta}, {x: data})
inference.run()

##Results:
qtheta_samples = qtheta.sample(1000).eval()
print(qtheta_samples.mean())
plt.hist(qtheta_samples)
plt.show()
