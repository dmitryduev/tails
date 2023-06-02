#!/usr/bin/env python
import fire
import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd
# import pathlib
# from sklearn.model_selection import train_test_split
# import sys
import tensorflow as tf

# from tqdm.auto import tqdm

import tails.models
from tails.utils import log


def fnr_vs_fpr(predictions, ground_truth):
    rbbins = np.arange(-0.0001, 1.0001, 0.0001)
    h_b, e_b = np.histogram(predictions[ground_truth == 0], bins=rbbins, density=True)
    h_b_c = np.cumsum(h_b)
    h_r, e_r = np.histogram(predictions[ground_truth == 1], bins=rbbins, density=True)
    h_r_c = np.cumsum(h_r)

    # h_b, e_b
    print(sum(ground_truth == 0), sum(ground_truth == 1))

    fig = plt.figure(figsize=(9, 4), dpi=200)
    ax = fig.add_subplot(111)

    rb_thres = np.array(list(range(len(h_b)))) / len(h_b)

    ax.plot(
        rb_thres,
        h_r_c / np.max(h_r_c),
        label="False Negative Rate (FNR)",
        linewidth=1.5,
    )
    ax.plot(
        rb_thres,
        1 - h_b_c / np.max(h_b_c),
        label="False Positive Rate (FPR)",
        linewidth=1.5,
    )

    mmce = (h_r_c / np.max(h_r_c) + 1 - h_b_c / np.max(h_b_c)) / 2
    ax.plot(
        rb_thres,
        mmce,
        "--",
        label="Mean misclassification error",
        color="gray",
        linewidth=1.5,
    )

    ax.set_xlim([-0.05, 1.05])

    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # vals = ax.get_yticks()
    # ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    ax.set_yscale("log")
    ax.set_ylim([5e-4, 1])
    vals = ax.get_yticks()
    ax.set_yticklabels(
        ["{:,.1%}".format(x) if x < 0.01 else "{:,.0%}".format(x) for x in vals]
    )

    # thresholds:
    # thrs = [0.5, ]
    thrs = [0.5, 0.7]
    for t in thrs:
        m_t = rb_thres < t
        fnr = np.array(h_r_c / np.max(h_r_c))[m_t][-1]
        fpr = np.array(1 - h_b_c / np.max(h_b_c))[m_t][-1]
        print(t, fnr * 100, fpr * 100)
        # ax.vlines(t_1, 0, 1.1)
        ax.vlines(t, 0, max(fnr, fpr))
        ax.text(
            t - 0.05,
            max(fnr, fpr) + 0.01,
            f" {fnr*100:.1f}% FNR\n {fpr*100:.1f}% FPR",
            fontsize=10,
        )

    ax.set_xlabel("$p_c$ score threshold")
    ax.set_ylabel("Cumulative percentage")
    ax.legend(loc="upper center")
    ax.grid(True, which="major", linewidth=0.5)
    ax.grid(True, which="minor", linewidth=0.3)
    plt.tight_layout()
    plt.show()


class TailsLoss(tf.keras.losses.BinaryCrossentropy):
    def __init__(self, w_1: float = 1, w_2: float = 1, **kwargs):
        super(TailsLoss, self).__init__(**kwargs)
        self.w_1 = w_1
        self.w_2 = w_2

    def call(self, y_true, y_pred):
        output = tf.convert_to_tensor(y_pred[..., 0])
        target = tf.cast(y_true[..., 0], output.dtype)

        # l_1: binary crossentropy for the label
        l_1 = super(TailsLoss, self).call(target, output)
        w_1 = tf.cast(self.w_1, output.dtype)
        l_1 = tf.math.multiply(l_1, w_1)

        # l_2: L1 loss
        l_2 = tf.norm(y_pred[..., 1:] - y_true[..., 1:], ord=1)

        # l_2: L1 loss + L2 regularization
        # l_2 = tf.norm(y_pred[..., 1:] - y_true[..., 1:], ord=1) + \
        #       1e-3 * tf.norm(y_pred[..., 1:] - y_true[..., 1:], ord=2)

        l_2 = tf.math.multiply(l_2, target)
        l_2 = tf.math.divide(l_2, tf.math.reduce_sum(target))
        w_2 = tf.cast(self.w_2, output.dtype)
        l_2 = tf.math.multiply(l_2, w_2)

        return l_1 + l_2


class LabelAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="label_accuracy", threshold=0.5, **kwargs):
        super(LabelAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.threshold = float(threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        output = y_pred[..., 0]
        #         target = tf.cast(y_true[..., 0], output.dtype)
        target = tf.cast(y_true[..., 0], tf.bool)

        threshold = tf.cast(0.5, output.dtype)
        output = tf.cast(output > threshold, tf.bool)

        #         values = tf.cast(tf.math.equal(target, output), output.dtype)
        values = tf.cast(tf.math.equal(target, output), tf.float32)
        ones = tf.cast(tf.math.equal(target, target), tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(values, sample_weight)

        self.count.assign_add(tf.math.reduce_sum(values, axis=-1))
        self.total.assign_add(tf.math.reduce_sum(ones, axis=-1))

    def result(self):
        return tf.math.divide(self.count, self.total)


class PositionRootMeanSquarredError(tf.keras.metrics.Metric):
    def __init__(self, name="position_rmse", scaling_factor=1, **kwargs):
        super(PositionRootMeanSquarredError, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.rmse = self.add_weight(name="rmse", initializer="zeros")
        self.scaling_factor = float(scaling_factor)

    def update_state(self, y_true, y_pred, sample_weight=None):
        output = y_pred[..., 1:]
        target = tf.cast(y_true[..., 1:], output.dtype)
        label = tf.cast(y_true[..., 0], output.dtype)

        rmse = tf.math.reduce_mean(
            tf.math.sqrt(tf.math.squared_difference(output, target)), axis=-1
        )
        # only take positive examples into account:
        rmse = tf.math.multiply(rmse, label)

        self.rmse.assign_add(tf.math.reduce_sum(rmse, axis=-1))
        # only count the positive examples:
        self.total.assign_add(tf.math.reduce_sum(label, axis=-1))

    def result(self):
        sf = tf.constant(self.scaling_factor, dtype=self.rmse.dtype.base_dtype)
        return tf.math.multiply(sf, tf.math.divide(self.rmse, self.total))


def train_and_eval(
    train_dataset,
    val_dataset,
    test_dataset,
    steps_per_epoch_train,
    steps_per_epoch_val,
    epochs,
    class_weight,
    model_name: str = "tails",
    tag="20210101",
    w_1: float = 1.2,
    w_2: float = 1,
    class_threshold: float = 0.5,
    scaling_factor=256,
    input_shape=(256, 256, 3),
    weights: str = None,
    save_model=False,
    verbose=False,
    **kwargs,
):
    classifier = tails.models.DNN(name=model_name)

    tails_loss = TailsLoss(name="loss", w_1=w_1, w_2=w_2)
    label_accuracy = LabelAccuracy(threshold=class_threshold)
    # convert position RMSE to pixels
    position_rmse = PositionRootMeanSquarredError(scaling_factor=scaling_factor)

    learning_rate = kwargs.get("learning_rate", 3e-4)
    patience = kwargs.get("patience", 30)

    classifier.setup(
        input_shape=input_shape,
        n_output_neurons=3,
        architecture="tails",
        loss=tails_loss,
        optimizer="adam",
        lr=learning_rate,  # epsilon=1e-3, beta_1=0.7,
        metrics=[label_accuracy, position_rmse],
        patience=patience,
        monitor="val_position_rmse",
        restore_best_weights=True,
        callbacks=("early_stopping", "learning_rate_scheduler" "tensorboard"),
        tag=tag,
        logdir="logs",
    )

    # pre-load weights?
    if weights is not None:
        classifier.model.load_weights(weights)

    classifier.train(
        train_dataset,
        val_dataset,
        steps_per_epoch_train,
        steps_per_epoch_val,
        epochs=epochs,
        class_weight=class_weight,
        verbose=True,
    )

    # evaluate
    stats = classifier.evaluate(test_dataset)
    if verbose:
        log(stats)

    if save_model:
        classifier.model.save_weights(f"{model_name}-{tag}")


if __name__ == "__main__":
    fire.Fire(train_and_eval)
