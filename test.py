import tensorflow as tf 

with tf. device("gpu:0"):
    print("tf.keras code in this scope will run on GPU")
with tf. device("cpu:0"):
    print("tf.keras code in this scope will run on CPU")