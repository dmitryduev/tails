from abc import ABC, abstractmethod

# from collections import defaultdict
import datetime
import os
import tensorflow as tf
import efficientnet.tfkeras as efn
from .efficientdet import Tails


class AbstractClassifier(ABC):
    def __init__(self, name):
        # classifier name: label_classifier
        self.name = name
        # model that will be trained and evaluated
        self.model = None
        # metadata needed to set the classifier up
        # self.meta = defaultdict(str)
        self.meta = dict()

    @abstractmethod
    def setup(self, **kwargs):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass


class DNN(AbstractClassifier):
    """
    Baseline model with a statically-defined graph
    """

    def setup(
        self,
        input_shape=(256, 256, 3),
        architecture="efficientnetb0",
        loss="binary_crossentropy",
        optimizer="adam",
        callbacks=("early_stopping", "learning_rate_scheduler", "tensorboard"),
        tag=None,
        logdir="logs",
        **kwargs,
    ):

        tf.keras.backend.clear_session()

        # n_output_neurons = kwargs.get("n_output_neurons", 1)

        self.model = self.build_model(
            input_shape=input_shape,
            # n_output_neurons=n_output_neurons,
            architecture=architecture,
        )

        self.meta["loss"] = loss
        if optimizer == "adam":
            lr = kwargs.get("lr", 3e-4)
            beta_1 = kwargs.get("beta_1", 0.9)
            beta_2 = kwargs.get("beta_2", 0.999)
            epsilon = kwargs.get("epsilon", 1e-7)  # None?
            decay = kwargs.get("decay", 0.0)
            amsgrad = kwargs.get("amsgrad", 3e-4)
            self.meta["optimizer"] = tf.keras.optimizers.Adam(
                lr=lr,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                decay=decay,
                amsgrad=amsgrad,
            )
        elif optimizer == "sgd":
            lr = kwargs.get("lr", 3e-4)
            momentum = kwargs.get("momentum", 0.9)
            decay = kwargs.get("decay", 1e-6)
            nesterov = kwargs.get("nesterov", True)
            self.meta["optimizer"] = tf.keras.optimizers.SGD(
                lr=lr, momentum=momentum, decay=decay, nesterov=nesterov
            )
        else:
            print("Could not recognize optimizer, using Adam with default params")
            self.meta["optimizer"] = tf.keras.optimizers.Adam(
                lr=3e-4,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                decay=0.0,
                amsgrad=False,
            )
        # self.meta['epochs'] = epochs
        # self.meta['patience'] = patience
        # self.meta['weight_per_class'] = weight_per_class

        metrics = kwargs.get(
            "metrics",
            [
                tf.keras.metrics.TruePositives(name="tp"),
                tf.keras.metrics.FalsePositives(name="fp"),
                tf.keras.metrics.TrueNegatives(name="tn"),
                tf.keras.metrics.FalseNegatives(name="fn"),
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
        self.meta["metrics"] = metrics

        self.meta["callbacks"] = []
        for callback in set(callbacks):
            if callback == "early_stopping":
                # halt training if no gain in validation accuracy over patience epochs
                monitor = kwargs.get("monitor", "val_loss")
                patience = kwargs.get("patience", 15)
                restore_best_weights = kwargs.get("restore_best_weights", True)
                early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=patience,
                    restore_best_weights=restore_best_weights,
                )
                self.meta["callbacks"].append(early_stopping_callback)

            elif callback == "reduce_lr_on_plateau":
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.1,
                    patience=10,
                    verbose=0,
                    mode="auto",
                    min_delta=0.0001,
                    cooldown=0,
                    min_lr=0,
                    **kwargs,
                )

                raise NotImplementedError("Implement ReduceLROnPlateau")

            elif callback == "learning_rate_scheduler":
                learning_rate_decay_min_epoch = kwargs.get(
                    "learning_rate_decay_min_epoch", 30
                )
                learning_rate_decay_index = kwargs.get(
                    "learning_rate_decay_index", -0.1
                )

                def scheduler(epoch, lr):
                    if epoch < learning_rate_decay_min_epoch:
                        return lr
                    else:
                        return lr * tf.math.exp(learning_rate_decay_index)

                learning_rate_scheduler_callback = (
                    tf.keras.callbacks.LearningRateScheduler(scheduler)
                )
                self.meta["callbacks"].append(learning_rate_scheduler_callback)

            elif callback == "tensorboard":
                # logs for TensorBoard:
                if tag:
                    log_tag = f'{self.name.replace(" ", "_")}-{tag}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
                else:
                    log_tag = f'{self.name.replace(" ", "_")}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
                logdir_tag = os.path.join("logs", log_tag)
                tensorboard_callback = tf.keras.callbacks.TensorBoard(
                    os.path.join(logdir_tag, log_tag), histogram_freq=1
                )
                self.meta["callbacks"].append(tensorboard_callback)

        self.model.compile(
            optimizer=self.meta["optimizer"],
            loss=self.meta["loss"],
            metrics=self.meta["metrics"],
        )

    def build_model(
        self, input_shape: tuple = (256, 256, 3), architecture: str = "tails"
    ):

        architectures = {
            "vgg": self.vgg,
            "resnet18": self.resnet18,
            "resnet50": self.resnet50,
            "efficientnetb0": self.efficientnetb0,
            "efficientnetb1": self.efficientnetb1,
            "efficientnetb2": self.efficientnetb2,
            "efficientnetb3": self.efficientnetb3,
            "tails": self.tails,
        }

        m = architectures[architecture.lower()](input_shape)

        return m

    def efficientnetb0(self, input_shape, **kwagrs):
        # x_input = tf.keras.Input(shape=input_shape, name="triplet")

        m = efn.EfficientNetB0(input_shape=(256, 256, 3), weights=None)
        # remove the output layer, leave the feature extraction part
        m_fe = tf.keras.Model(inputs=m.inputs, outputs=m.layers[-2].output)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(m_fe.output)
        m = tf.keras.Model(inputs=m_fe.inputs, outputs=output)

        return m

    def efficientnetb1(self, input_shape, **kwagrs):
        # x_input = tf.keras.Input(shape=input_shape, name="triplet")

        m = efn.EfficientNetB1(input_shape=(256, 256, 3), weights=None)
        # remove the output layer, leave the feature extraction part
        m_fe = tf.keras.Model(inputs=m.inputs, outputs=m.layers[-2].output)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(m_fe.output)
        m = tf.keras.Model(inputs=m_fe.inputs, outputs=output)

        return m

    def efficientnetb2(self, input_shape, **kwagrs):
        # x_input = tf.keras.Input(shape=input_shape, name="triplet")

        m = efn.EfficientNetB2(input_shape=(256, 256, 3), weights=None)
        # remove the output layer, leave the feature extraction part
        m_fe = tf.keras.Model(inputs=m.inputs, outputs=m.layers[-2].output)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(m_fe.output)
        m = tf.keras.Model(inputs=m_fe.inputs, outputs=output)

        return m

    def efficientnetb3(self, input_shape, **kwagrs):
        # x_input = tf.keras.Input(shape=input_shape, name="triplet")

        m = efn.EfficientNetB3(input_shape=(256, 256, 3), weights=None)
        # remove the output layer, leave the feature extraction part
        m_fe = tf.keras.Model(inputs=m.inputs, outputs=m.layers[-2].output)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(m_fe.output)
        m = tf.keras.Model(inputs=m_fe.inputs, outputs=output)

        return m

    def tails(self, input_shape, **kwagrs):

        m = Tails(input_shape=input_shape, weighted_bifpn=False, name="tails")

        return m

    @staticmethod
    def vgg(input_shape, **kwargs):
        x_input = tf.keras.Input(shape=input_shape, name="triplet")

        x_conv = tf.keras.layers.Conv2D(
            16, (9, 9), activation="relu", name="conv_conv_1"
        )(x_input)
        x_conv = tf.keras.layers.Conv2D(
            16, (9, 9), activation="relu", name="conv_conv_2"
        )(x_conv)
        x_conv = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(x_conv)
        x_conv = tf.keras.layers.Dropout(0.25)(x_conv)

        x_conv = tf.keras.layers.Conv2D(
            32, (7, 7), activation="relu", name="conv_conv_3"
        )(x_conv)
        x_conv = tf.keras.layers.Conv2D(
            32, (7, 7), activation="relu", name="conv_conv_4"
        )(x_conv)
        x_conv = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(x_conv)
        x_conv = tf.keras.layers.Dropout(0.25)(x_conv)

        x_conv = tf.keras.layers.Conv2D(
            64, (5, 5), activation="relu", name="conv_conv_5"
        )(x_conv)
        x_conv = tf.keras.layers.Conv2D(
            64, (5, 5), activation="relu", name="conv_conv_6"
        )(x_conv)
        x_conv = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(x_conv)
        x_conv = tf.keras.layers.Dropout(0.25)(x_conv)

        # x_conv = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv_conv_7')(x_conv)
        # x_conv = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv_conv_8')(x_conv)
        # x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
        # x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
        #
        # x_conv = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv_conv_9')(x_conv)
        # x_conv = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv_conv_10')(x_conv)
        # x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
        # x_conv = tf.keras.layers.Dropout(0.25)(x_conv)

        x_conv = tf.keras.layers.Flatten()(x_conv)

        x_conv = tf.keras.layers.Dense(256, activation="relu", name="conv_fc_1")(x_conv)
        x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
        x_conv = tf.keras.layers.Dense(32, activation="relu", name="conv_fc_2")(x_conv)

        # one more dense layer?
        # x_conv = tf.keras.layers.Dense(16, activation='relu', name='fc_1')(x_conv)

        # Logistic regression to output the final score
        x = tf.keras.layers.Dense(1, activation="sigmoid", name="score")(x_conv)

        m = tf.keras.Model(inputs=x_input, outputs=x)

        return m

    @staticmethod
    def identity_block(x, f, filters, stage, block):
        """
        Implementation of the identity block as defined in Figure 3
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """
        # batch norm momentum
        batch_norm_momentum = 0.2

        # defining name basis
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"

        # Retrieve Filters
        f1, f2, f3 = filters

        # Save the input value. You'll need this later to add back to the main path.
        X_shortcut = x

        # First component of main path
        x = tf.keras.layers.Conv2D(
            filters=f1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            name=conv_name_base + "2a",
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        )(x)
        # X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X, training=1)
        # X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X)
        x = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=batch_norm_momentum, name=bn_name_base + "2a"
        )(x)
        x = tf.keras.layers.Activation("relu")(x)

        # Second component of main path
        x = tf.keras.layers.Conv2D(
            filters=f2,
            kernel_size=(f, f),
            strides=(1, 1),
            padding="same",
            name=conv_name_base + "2b",
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        )(x)
        # X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X, training=1)
        x = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=batch_norm_momentum, name=bn_name_base + "2b"
        )(x)
        x = tf.keras.layers.Activation("relu")(x)

        # Third component of main path
        x = tf.keras.layers.Conv2D(
            filters=f3,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            name=conv_name_base + "2c",
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        )(x)
        # X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X, training=1)
        x = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=batch_norm_momentum, name=bn_name_base + "2c"
        )(x)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        x = tf.keras.layers.add([X_shortcut, x])
        x = tf.keras.layers.Activation("relu")(x)

        return x

    @staticmethod
    def convolutional_block(x, f, filters, stage, block, s=2):
        """
        Implementation of the convolutional block as defined in Figure 4
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used
        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """
        # batch norm momentum
        batch_norm_momentum = 0.2

        # defining name basis
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"

        # Retrieve Filters
        f1, f2, f3 = filters

        # Save the input value
        X_shortcut = x

        # MAIN PATH
        # First component of main path
        x = tf.keras.layers.Conv2D(
            f1,
            (1, 1),
            strides=(s, s),
            name=conv_name_base + "2a",
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        )(x)
        # X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X, training=1)
        x = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=batch_norm_momentum, name=bn_name_base + "2a"
        )(x)
        x = tf.keras.layers.Activation("relu")(x)

        # Second component of main path
        x = tf.keras.layers.Conv2D(
            filters=f2,
            kernel_size=(f, f),
            strides=(1, 1),
            padding="same",
            name=conv_name_base + "2b",
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        )(x)
        # X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X, training=1)
        x = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=batch_norm_momentum, name=bn_name_base + "2b"
        )(x)
        x = tf.keras.layers.Activation("relu")(x)

        # Third component of main path
        x = tf.keras.layers.Conv2D(
            filters=f3,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            name=conv_name_base + "2c",
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        )(x)
        # X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X, training=1)
        x = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=batch_norm_momentum, name=bn_name_base + "2c"
        )(x)

        # SHORTCUT PATH
        X_shortcut = tf.keras.layers.Conv2D(
            filters=f3,
            kernel_size=(1, 1),
            strides=(s, s),
            padding="valid",
            name=conv_name_base + "1",
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        )(X_shortcut)
        # X_shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(X_shortcut, training=1)
        X_shortcut = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.1, name=bn_name_base + "1"
        )(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        x = tf.keras.layers.add([X_shortcut, x])
        x = tf.keras.layers.Activation("relu")(x)

        return x

    def resnet50(self, input_shape, n_output_neurons: int = 1, **kwargs):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
        Arguments:
        input_shape -- shape of the images of the dataset
        n_output_neurons -- integer, number of output neurons
        Returns:
        model -- a Model() instance in Keras
        """
        # batch norm momentum
        batch_norm_momentum = 0.2

        x_input = tf.keras.Input(shape=input_shape, name="triplet")

        # Zero-Padding
        x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)

        # Stage 1
        x = tf.keras.layers.Conv2D(
            64,
            (7, 7),
            strides=(2, 2),
            name="conv1",
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        )(x)
        # X = BatchNormalization(axis=-1, name='bn_conv1')(X, training=1)
        x = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=batch_norm_momentum, name="bn_conv1"
        )(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        # Stage 2
        x = self.convolutional_block(
            x, f=3, filters=[64, 64, 256], stage=2, block="a", s=1
        )
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block="b")
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block="c")

        # Stage 3
        x = self.convolutional_block(
            x, f=3, filters=[128, 128, 512], stage=3, block="a", s=2
        )
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block="b")
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block="c")
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block="d")

        # Stage 4
        x = self.convolutional_block(
            x, f=3, filters=[256, 256, 512], stage=4, block="a", s=2
        )
        x = self.identity_block(x, 3, [256, 256, 512], stage=4, block="b")
        x = self.identity_block(x, 3, [256, 256, 512], stage=4, block="c")
        x = self.identity_block(x, 3, [256, 256, 512], stage=4, block="d")
        x = self.identity_block(x, 3, [256, 256, 512], stage=4, block="e")
        x = self.identity_block(x, 3, [256, 256, 512], stage=4, block="f")

        # Stage 5
        x = self.convolutional_block(
            x, f=3, filters=[512, 512, 1024], stage=5, block="a", s=2
        )
        x = self.identity_block(x, 3, [512, 512, 1024], stage=5, block="b")
        x = self.identity_block(x, 3, [512, 512, 1024], stage=5, block="c")

        # AVGPOOL
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), name="avg_pool")(x)

        x = tf.keras.layers.Flatten()(x)

        # more fully connected layers?
        x = tf.keras.layers.Dense(256, activation="relu", name="fc_1")(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(32, activation="relu", name="conv_fc_2")(x)

        # output layer
        activation = "sigmoid"
        x = tf.keras.layers.Dense(
            n_output_neurons,
            activation=activation,
            name="fcOUT",
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        )(x)

        # Create model
        model = tf.keras.models.Model(inputs=x_input, outputs=x, name="resnet50")

        return model

    def resnet18(self, input_shape, n_classes: int = 1, **kwargs):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
        Arguments:
        input_shape -- shape of the images of the dataset
        n_classes -- integer, number of classes. if = 1, sigmoid is used in the output layer; softmax otherwise
        Returns:
        model -- a Model() instance in Keras
        """
        # batch norm momentum
        batch_norm_momentum = 0.2

        x_input = tf.keras.Input(shape=input_shape, name="triplet")

        # Zero-Padding
        x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)

        # Stage 1
        x = tf.keras.layers.Conv2D(
            64,
            (7, 7),
            strides=(2, 2),
            name="conv1",
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        )(x)
        # X = BatchNormalization(axis=-1, name='bn_conv1')(x, training=1)
        x = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=batch_norm_momentum, name="bn_conv1"
        )(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        # Stage 2
        x = self.convolutional_block(
            x, f=3, filters=[64, 64, 256], stage=2, block="a", s=1
        )
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block="b")
        # X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        # Stage 3
        x = self.convolutional_block(
            x, f=3, filters=[128, 128, 512], stage=3, block="a", s=2
        )
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block="b")
        # X = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        # X = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4
        x = self.convolutional_block(
            x, f=3, filters=[256, 256, 512], stage=4, block="a", s=2
        )
        x = self.identity_block(x, 3, [256, 256, 512], stage=4, block="b")
        # X = self.identity_block(x, 3, [256, 256, 512], stage=4, block='c')
        # X = self.identity_block(x, 3, [256, 256, 512], stage=4, block='d')
        # X = self.identity_block(x, 3, [256, 256, 512], stage=4, block='e')
        # X = self.identity_block(x, 3, [256, 256, 512], stage=4, block='f')

        # Stage 5
        x = self.convolutional_block(
            x, f=3, filters=[512, 512, 1024], stage=5, block="a", s=2
        )
        x = self.identity_block(x, 3, [512, 512, 1024], stage=5, block="b")
        # X = self.identity_block(x, 3, [512, 512, 1024], stage=5, block='c')

        # AVGPOOL
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), name="avg_pool")(x)

        # output layer
        x = tf.keras.layers.Flatten()(x)
        activation = "sigmoid" if n_classes == 1 else "softmax"
        x = tf.keras.layers.Dense(
            n_classes,
            activation=activation,
            name="fcOUT",
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        )(x)

        # Create model
        model = tf.keras.models.Model(inputs=x_input, outputs=x, name="resnet18")

        return model

    def train(
        self,
        train_dataset,
        val_dataset,
        steps_per_epoch_train,
        steps_per_epoch_val,
        epochs=300,
        class_weight=None,
        verbose=False,
    ):

        if not class_weight:
            # all our problems here are binary classification ones:
            class_weight = {i: 1 for i in range(2)}

        self.meta["history"] = self.model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch_train,
            validation_data=val_dataset,
            validation_steps=steps_per_epoch_val,
            class_weight=class_weight,
            callbacks=self.meta["callbacks"],
            verbose=verbose,
        )

    def evaluate(self, test_dataset):
        return self.model.evaluate(test_dataset)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, output_path="./", output_format="hdf5", tag=None):

        assert output_format in ("SavedModel", "hdf5"), "unknown output format"

        output_name = self.name if not tag else f"{self.name}.{tag}"

        if (output_path != "./") and (not os.path.exists(output_path)):
            os.makedirs(output_path)

        if output_format == "SavedModel":
            self.model.save(os.path.join(output_path, output_name))
        elif output_format == "hdf5":
            self.model.save(os.path.join(output_path, f"{output_name}.h5"))
