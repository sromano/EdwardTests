import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import edward as ed
from edward.models import Bernoulli, Beta, Binomial

#sess = ed.get_session()
#Bernoulli(probs=.5).sample().eval()


#Esto no anda, pero me ayudó a entender.
#Entiendo que es un grafo fijo
#condition =  tf.cast(Bernoulli(probs=.5), tf.bool)
#stop = tf.constant(0)
#suma = tf.constant(1)
#result = tf.cond(condition, lambda: stop, lambda: suma)
#suma = tf.constant(1) + result

#Esto entiendo que es un grafo dinámico
#Model
p = Beta(1.0, 1.0)
b = tf.constant("B")
condition =  lambda i: tf.cast(Bernoulli(probs=p), tf.bool)
a = lambda next: tf.string_join([tf.constant("A"),next])
result = tf.while_loop(condition, a, [b])
shaped_result = tf.stack([result])

#Observations
data = b"AAAAAAAAAAAAAAAAAAB"

##Infer
qp_a = tf.Variable(1.0)
qp_b = tf.Variable(1.0)
qp = Beta(qp_a, qp_b)

sess = ed.get_session()
inference = ed.KLqp({p: qp}, {shaped_result: [data]})
inference.initialize()

tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  info_dict = inference.update()
  #inference.print_progress(info_dict)

inference.finalize()

#Results
qp_samples = qp.sample(1000)
mean = tf.reduce_mean(qp_samples)
print(sess.run(mean))
plt.hist(sess.run(qp_samples))
plt.savefig('myfig.png')
