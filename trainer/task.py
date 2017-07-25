import numpy as np
import tensorflow as tf


def model_fn(features, labels, mode, params, config):
    logits = tf.layers.dense(features, 2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    train_op = tf.train.GradientDescentOptimizer(1e-2).minimize(loss, tf.train.get_global_step())
    prob = tf.nn.softmax(logits)
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=prob,
        loss=loss,
        train_op=train_op,
    )
    return estimator_spec


def train_input_fn():
    img = np.random.normal(size=[32, 128])
    return tf.constant(img, dtype=tf.float32),  tf.one_hot(np.random.choice(2, 32), depth=2)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    clf = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir="outputs",
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=100),
    )
    clf.train(input_fn=train_input_fn, steps=5)
