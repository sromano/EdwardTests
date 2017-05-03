import tensorflow as tf
import edward as ed
from edward.models import Bernoulli, Beta, Binomial
import matplotlib.pyplot as plt
import seaborn as sns

##Single coin weight inference

##Model:
theta = Beta(1.0, 1.0)
x = Bernoulli(probs=theta)#, sample_shape=(1,))

##Sampling:
# with tf.Session() as sess:
#     for i in range(10):
#         print(x.eval())

##Observations:
data=1

##Infer:
qtheta = Beta(tf.Variable(1.0), tf.Variable(1.0))  #Why need tf.Variable here?
inference = ed.KLqp({theta: qtheta}, {x: data})
inference.run()

##Results:
qtheta_samples = qtheta.sample(1000).eval()
print(qtheta_samples.mean())
plt.hist(qtheta_samples)
plt.show()
