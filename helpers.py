import tensorflow as tf
import numpy as np

class epochCounter(tf.keras.callbacks.Callback):
    def __init__(self, e_counter):
        super(epochCounter, self).__init__()
        self.e_counter = tf.constant(e_counter, dtype=tf.float32) 

    def on_epoch_begin(self, epoch, logs={}):
        if(epoch%10==0): self.e_counter = tf.constant(epoch+1, dtype=tf.float32)
def relu_evidence(y):
    return layers.Activation('relu')(y)

def edl_loss(func, y, alpha, annealing_step, device=None):
    num_classes = 2
    annealing_step = tf.constant(10.0, dtype=tf.float32) 
    S = tf.math.reduce_sum(alpha, axis=1, keepdims=True)

    A = tf.math.reduce_sum(y * (func(S) - func(alpha)), axis=1, keepdims=True)

    annealing_coef = tf.math.minimum(tf.constant(1.0, dtype=tf.float32), tf.math.divide(instance.e_counter, annealing_step))
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha)
    return tf.math.reduce_sum(A + kl_div)
def kl_divergence(alpha):
    num_classes = 2
    ones = tf.ones([1, num_classes], dtype=tf.float32)
    sum_alpha = tf.math.reduce_sum(alpha, axis=1, keepdims=True)
    first_term = (
        tf.math.lgamma(sum_alpha)
        - tf.math.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
        + tf.math.reduce_sum(tf.math.lgamma(ones),axis=1, keepdims=True)
        - tf.math.reduce_sum(tf.math.lgamma(ones), axis=1, keepdims=True)
    )
    second_term = (
        tf.math.reduce_sum(tf.math.multiply((alpha - ones), 
                                       (tf.math.digamma(alpha) - tf.math.digamma(sum_alpha))),
                           axis=1, keepdims=True)
    )
    kl = first_term + second_term
    return kl

def edl_digamma_loss(y_true, y_pred):
    num_classes = 2
    evidence = relu_evidence(y_pred)
    alpha = evidence + 1
    loss = tf.math.reduce_mean(
        edl_loss(
            tf.math.digamma, y_true, alpha, num_classes
        )
    )
    return loss


def get_uncertainty(model, arr_scaled=[], uni_modal, split=180, x_train_left=[], x_train_right=[]):
    mri_volumes_test = []
    uncertainties = []
    probabilities = []
    outputs = []
    x_left_test, x_right_test = [], []
    
    for i in range(len(arr_scaled)):
        mri_volumes_test.append(arr_scaled[i])
        x_left_test.append(x_train_left[i])
        x_right_test.append(x_train_right[i])
        
        num_classes = 2
        if(uni_modal):
            output = model(np.expand_dims(arr_scaled[i], axis=0))
        else:
            output = model([np.expand_dims(x_train_left[i], axis=0), np.expand_dims(x_train_right[i], axis=0), 
                          np.expand_dims(arr_scaled[i], axis=0)])
        evidence = helpers.relu_evidence(output)
        alpha = evidence + 1
        uncertainty = num_classes / tf.math.reduce_sum(alpha, axis=1, keepdims=True)
        preds, _ = tf.math.maximum(output[0], tf.constant(1, dtype=tf.float32))
        prob = alpha / tf.math.reduce_sum(alpha, axis=1, keepdims=True)
        probabilities.append(prob)
        outputs.append(output)
        uncertainties.append(uncertainty)
        
    return mri_volumes_test, uncertainties, probabilities, outputs, x_left_test, x_right_test
