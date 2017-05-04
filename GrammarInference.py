import tensorflow as tf
import edward as ed
from edward.models import Bernoulli, Empirical, Beta, Normal
import matplotlib.pyplot as plt
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

#Probabilistic Grammar Inference using edward
#Grammar
#This doesn't work, recursion.
# def generateFormula(theta):
#     a = tf.constant("A")
#     b = tf.constant("B")
#     formula = tf.cond(tf.cast(Bernoulli(probs=theta), tf.bool), lambda: b, lambda: a + generateFormula(theta))
#     return formula

def generateFormula(theta):

    def cond(theta, sentence):
        return tf.cast(Bernoulli(probs=theta), tf.bool)

    def body(theta, sentence):
        return theta, a + sentence

    #sentence = tf.constant("B")
    sentence = tf.constant(0, dtype=tf.int32)
    #a = tf.constant("A")
    a = tf.constant(1, dtype=tf.int32)
    formula = tf.while_loop(cond, body, loop_vars=[theta, sentence])[1]
    return formula

N=8
theta = Beta(1.0, 1.0)
#formulas = tf.constant("", shape=(N,))+generateFormula(theta)[1]
#formulas = tf.constant(0, shape=(N,))+generateFormula(theta)[1]
#formulas = tf.stack([generateFormula(theta) for i in range(N)])
#formulas = tf.stack([generateFormula(theta), generateFormula(theta),generateFormula(theta), generateFormula(theta),
#generateFormula(theta), generateFormula(theta),generateFormula(theta), generateFormula(theta)])
formulaList=[]
for i in range(N):
    formulaList.append(generateFormula(theta))
formulas = tf.stack(formulaList)


##Sampling:
with tf.Session() as sess:
    for i in range(10):
        print(generateFormula(0.9).eval())
    print('\n')
        
#    for i in range(max(N,10)):
#        print(formulas[i].eval())

##Observations
#data = tf.constant("AAAAAAAAAAAAAAAB", shape=(N,))
#data = tf.constant("B", shape=(N,)) #
#data = tf.constant("C", shape=(N,)) #???
#data = tf.constant(0, shape=(N,))
#data = tf.constant(20, shape=(N,))
data = np.ones((N,))*17

##Infer:
T=10000
qtheta = Empirical(params=tf.Variable(0.5+tf.zeros([T]))) #Why need tf.Variable here?
tf.summary.scalar('qtheta', qtheta)

#proposal_theta = Beta(concentration1=1.0, concentration0=1.0, sample_shape=(1,))
# proposal_theta = Normal(loc=theta,scale=0.5)
# inference = ed.MetropolisHastings({theta: qtheta}, {theta: proposal_theta}, {formulas: data})

sess = ed.get_session()
inference = ed.HMC({theta: qtheta}, {formulas: data})
inference.initialize()

tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

inference.finalize()
train_writer = tf.summary.FileWriter('/tmp/tensorflow/',sess.graph)

# qtheta = Beta(tf.Variable(1.0), tf.Variable(1.0))  #Why need tf.Variable here?
# inference = ed.KLqp({theta: qtheta}, {formulas: data})





##Results:
qtheta_samples = qtheta.sample(1000).eval()
print(qtheta_samples.mean())
plt.hist(qtheta_samples)
plt.show()
