import numpy as np
import tensorflow as tf
from tensorflow import keras


class FashionExpl_model(keras.Model):
    def __init__(self, factors=200,
                 mlp_color=(64,),
                 mlp_att=(64,),
                 mlp_out=(64,),
                 mlp_cnn=(64,),
                 cnn_channels=64,
                 cnn_kernels=3,
                 cnn_strides=1,
                 att_feat_agg='multiplication',
                 out_feat_agg='multiplication',
                 sampler_str='pairwise',
                 temperature=1.0,
                 dropout=0.2,
                 learning_rate=0.001,
                 l_w=0,
                 num_users=100,
                 num_items=100,
                 name="FashionExpl",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self._factors = factors
        self._mlp_color = mlp_color
        self._mlp_att = mlp_att
        self._mlp_out = mlp_out
        self._mlp_cnn = mlp_cnn
        self._cnn_channels = cnn_channels
        self._cnn_kernels = cnn_kernels
        self._cnn_strides = cnn_strides
        self._att_feat_agg = att_feat_agg
        self._out_feat_agg = out_feat_agg
        self._sampler_str = sampler_str
        self._temperature = temperature
        self.l_w = l_w
        self._learning_rate = learning_rate
        self._num_items = num_items
        self._num_users = num_users
        self._dropout = dropout

        self.initializer = tf.initializers.RandomNormal(mean=0, stddev=0.01)
        self.initializer_attentive = tf.initializers.GlorotUniform()

        self.Gu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi', dtype=tf.float32)

        self.color_encoder = keras.Sequential()
        self.shape_encoder = keras.Sequential()

        self.attention_network = dict()

        self.mlp_output = keras.Sequential()

        self.create_color_weights()
        self.create_shape_weights()
        self.create_attention_weights()

        self.create_output_weights()

        self.loss_pointwise = keras.losses.BinaryCrossentropy()

        self.optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate)

    def create_color_weights(self):
        self.color_encoder.add(keras.layers.Dropout(self._dropout))
        for units in self._mlp_color[:-1]:
            if units != 1:
                self.color_encoder.add(keras.layers.Dense(units, activation='relu'))
        self.color_encoder.add(keras.layers.Dense(units=self._factors, use_bias=False))

    def create_shape_weights(self):
        self.shape_encoder.add(keras.layers.Conv2D(filters=self._cnn_channels,
                                                   kernel_size=(self._cnn_kernels, self._cnn_kernels),
                                                   strides=(self._cnn_strides, self._cnn_strides),
                                                   padding='same',
                                                   activation='relu'))
        self.shape_encoder.add(keras.layers.MaxPool2D(padding='same'))
        self.shape_encoder.add(keras.layers.GlobalAveragePooling2D())
        self.shape_encoder.add(keras.layers.Dropout(rate=self._dropout))
        for units in self._mlp_cnn[:-1]:
            if units != 1:
                self.shape_encoder.add(keras.layers.Dense(units=units, activation='relu'))
        self.shape_encoder.add(keras.layers.Dense(units=self._factors, use_bias=False))

    def create_output_weights(self):
        self.mlp_output.add(keras.layers.Dropout(self._dropout))
        for units in self._mlp_out[:-1]:
            if units != 1:
                self.mlp_output.add(keras.layers.Dense(units, activation='relu'))
        if self._sampler_str == 'pointwise':
            self.mlp_output.add(keras.layers.Dense(units=self._mlp_out[-1], use_bias=False, activation='sigmoid'))
        elif self._sampler_str == 'pairwise':
            self.mlp_output.add(keras.layers.Dense(units=self._mlp_out[-1], use_bias=False, activation='linear'))
        else:
            raise NotImplementedError('This sampler type has not been implemented for this model yet!')

    def create_attention_weights(self):
        if self._att_feat_agg == 'multiplication' or self._att_feat_agg == 'addition':
            input_shape = self._factors
        elif self._att_feat_agg == 'concatenation':
            input_shape = 2 * self._factors
        else:
            raise NotImplementedError('This aggregation method has not been implemented yet!')

        for layer in range(len(self._mlp_att)):
            if layer == 0:
                self.attention_network['W_{}'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[input_shape, self._mlp_att[layer]]),
                    name='W_{}'.format(layer + 1),
                    dtype=tf.float32
                )
                self.attention_network['b_{}'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self._mlp_att[layer]]),
                    name='b_{}'.format(layer + 1),
                    dtype=tf.float32
                )
            else:
                self.attention_network['W_{}'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self._mlp_att[layer - 1], self._mlp_att[layer]]),
                    name='W_{}'.format(layer + 1),
                    dtype=tf.float32
                )
                self.attention_network['b_{}'.format(layer + 1)] = tf.Variable(
                    self.initializer_attentive(shape=[self._mlp_att[layer]]),
                    name='b_{}'.format(layer + 1),
                    dtype=tf.float32
                )

    @tf.function
    def propagate_attention(self, g_u, colors, shapes, classes):
        all_a_i_l = None
        for layer in range(len(self._mlp_att)):
            if layer == 0:
                if self._att_feat_agg == 'multiplication':
                    all_a_i_l = tf.tensordot(
                        tf.expand_dims(g_u, 1) * tf.concat([colors, shapes, classes], axis=1),
                        self.attention_network['W_{}'.format(layer + 1)],
                        axes=[[2], [0]]
                    ) + self.attention_network['b_{}'.format(layer + 1)]
                elif self._att_feat_agg == 'addition':
                    all_a_i_l = tf.tensordot(
                        tf.expand_dims(g_u, 1) + tf.concat([colors, shapes, classes], axis=1),
                        self.attention_network['W_{}'.format(layer + 1)],
                        axes=[[2], [0]]
                    ) + self.attention_network['b_{}'.format(layer + 1)]
                elif self._att_feat_agg == 'concatenation':
                    all_a_i_l = tf.tensordot(
                        tf.concat([tf.repeat(tf.expand_dims(g_u, 1), repeats=3, axis=1),
                                   tf.concat([colors, shapes, classes], axis=1)],
                                  axis=2),
                        self.attention_network['W_{}'.format(layer + 1)],
                        axes=[[2], [0]]
                    ) + self.attention_network['b_{}'.format(layer + 1)]
                else:
                    raise NotImplementedError('This aggregation method has not been implemented yet!')
                all_a_i_l = tf.nn.relu(all_a_i_l)
            else:
                all_a_i_l = tf.tensordot(
                    all_a_i_l,
                    self.attention_network['W_{}'.format(layer + 1)],
                    axes=[[2], [0]]
                ) + self.attention_network['b_{}'.format(layer + 1)]

        all_alpha = tf.nn.softmax(all_a_i_l / self._temperature, axis=1)
        return all_alpha

    @tf.function
    def propagate_attention_batch(self, g_u, colors, shapes, classes):
        all_a_i_l = None
        for layer in range(len(self._mlp_att)):
            if layer == 0:
                if self._att_feat_agg == 'multiplication':
                    all_a_i_l = tf.tensordot(
                        tf.expand_dims(tf.expand_dims(g_u, 1), 1) * tf.expand_dims(
                            tf.concat([colors, shapes, classes], axis=1), 0),
                        self.attention_network['W_{}'.format(layer + 1)],
                        axes=[[3], [0]]
                    ) + self.attention_network['b_{}'.format(layer + 1)]
                elif self._att_feat_agg == 'addition':
                    all_a_i_l = tf.tensordot(
                        tf.expand_dims(tf.expand_dims(g_u, 1), 1) + tf.expand_dims(
                            tf.concat([colors, shapes, classes], axis=1), 0),
                        self.attention_network['W_{}'.format(layer + 1)],
                        axes=[[3], [0]]
                    ) + self.attention_network['b_{}'.format(layer + 1)]
                elif self._att_feat_agg == 'concatenation':
                    all_a_i_l = tf.tensordot(
                        tf.concat(
                            [tf.repeat(tf.repeat(tf.expand_dims(tf.expand_dims(g_u, 1), 1), repeats=3, axis=2),
                                       repeats=colors.shape[0], axis=1),
                             tf.repeat(tf.expand_dims(
                                 tf.concat([colors, shapes, classes], axis=1), 0), repeats=g_u.shape[0], axis=0)],
                            axis=-1),
                        self.attention_network['W_{}'.format(layer + 1)],
                        axes=[[3], [0]]
                    ) + self.attention_network['b_{}'.format(layer + 1)]
                else:
                    raise NotImplementedError('This aggregation method has not been implemented yet!')
                all_a_i_l = tf.nn.relu(all_a_i_l)
            else:
                all_a_i_l = tf.tensordot(
                    all_a_i_l,
                    self.attention_network['W_{}'.format(layer + 1)],
                    axes=[[3], [0]]
                ) + self.attention_network['b_{}'.format(layer + 1)]

        all_alpha = tf.nn.softmax(all_a_i_l / self._temperature, axis=2)
        return all_alpha

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item, shapes, colors, class_i = inputs

        gamma_u = tf.nn.embedding_lookup(self.Gu, user)
        gamma_i = tf.nn.embedding_lookup(self.Gi, item)
        color_i = tf.expand_dims(self.color_encoder(colors, training), 1)
        shape_i = tf.expand_dims(self.shape_encoder(shapes, training), 1)
        class_i = tf.expand_dims(class_i, 1)

        if self._mlp_att != [0, 0, 0]:
            all_attention = self.propagate_attention(gamma_u, color_i, shape_i, class_i)
            attentive_features = tf.reduce_sum(tf.multiply(
                all_attention,
                tf.concat([color_i, shape_i, class_i], axis=1)
            ), axis=1)
        else:
            attentive_features = tf.squeeze(color_i) + tf.squeeze(shape_i) + tf.squeeze(class_i)

        # score prediction
        if self._out_feat_agg == 'multiplication':
            gamma_i = gamma_i * attentive_features
        elif self._out_feat_agg == 'addition':
            gamma_i = gamma_i + attentive_features
        else:
            raise NotImplementedError('This aggregation method has not been implemented yet!')
        xui = self.mlp_output(tf.concat([gamma_u, gamma_i], axis=1), training)

        if self._mlp_att != [0, 0, 0]:
            return xui, \
                   gamma_u, \
                   gamma_i, \
                   color_i, \
                   shape_i, \
                   class_i, \
                   all_attention
        else:
            return xui, \
                   gamma_u, \
                   gamma_i, \
                   color_i, \
                   shape_i, \
                   class_i, \
                   None

    @tf.function
    def train_step(self, batch):
        if self._sampler_str == 'pairwise':
            user, pos, shapes_pos, colors_pos, classes_pos, neg, shapes_neg, colors_neg, classes_neg = batch
            with tf.GradientTape() as t:
                # Clean Inference
                xu_pos, \
                gamma_u, \
                gamma_i_pos, \
                color_i_pos, \
                shape_i_pos, \
                class_i_pos, \
                attention_pos = self(inputs=(user, pos, shapes_pos, colors_pos, classes_pos), training=True)

                xu_neg, \
                _, \
                gamma_i_neg, \
                color_i_neg, \
                shape_i_neg, \
                class_i_neg, \
                attention_neg = self(inputs=(user, neg, shapes_neg, colors_neg, classes_neg), training=True)

                result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
                loss = tf.reduce_sum(tf.nn.softplus(-result))

                # Regularization Component
                reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                     tf.nn.l2_loss(gamma_i_pos), tf.nn.l2_loss(gamma_i_neg),
                                                     tf.nn.l2_loss(color_i_pos), tf.nn.l2_loss(color_i_neg),
                                                     tf.nn.l2_loss(shape_i_pos), tf.nn.l2_loss(shape_i_neg),
                                                     tf.nn.l2_loss(class_i_pos), tf.nn.l2_loss(class_i_neg),
                                                     *[tf.nn.l2_loss(weight) for weight in
                                                       self.color_encoder.trainable_weights],
                                                     *[tf.nn.l2_loss(weight) for weight in
                                                       self.shape_encoder.trainable_weights],
                                                     *[tf.nn.l2_loss(value) for _, value in
                                                       self.attention_network.items() if self._mlp_att != [0, 0, 0]],
                                                     *[tf.nn.l2_loss(weight) for weight in
                                                       self.mlp_output.trainable_weights]])

                # Loss to be optimized
                loss += reg_loss
        elif self._sampler_str == 'pointwise':
            user, item, pos_neg, shapes, colors, classes = batch
            with tf.GradientTape() as t:
                # Clean Inference
                xui, \
                gamma_u, \
                gamma_i, \
                color_i, \
                shape_i, \
                class_i, \
                attention_pos = self(inputs=(user, item, shapes, colors, classes), training=True)

                loss = self.loss_pointwise(pos_neg, xui)

                # Regularization Component
                reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                     tf.nn.l2_loss(gamma_i),
                                                     tf.nn.l2_loss(color_i),
                                                     tf.nn.l2_loss(shape_i),
                                                     tf.nn.l2_loss(class_i),
                                                     *[tf.nn.l2_loss(weight) for weight in
                                                       self.color_encoder.trainable_weights],
                                                     *[tf.nn.l2_loss(weight) for weight in
                                                       self.shape_encoder.trainable_weights],
                                                     *[tf.nn.l2_loss(value) for _, value in
                                                       self.attention_network.items() if self._mlp_att != [0, 0, 0]],
                                                     *[tf.nn.l2_loss(weight) for weight in
                                                       self.mlp_output.trainable_weights]])

                # Loss to be optimized
                loss += reg_loss
        else:
            raise NotImplementedError('This sampler type has not been implemented for this model yet!')

        params = [
            self.Gu,
            self.Gi,
            *self.color_encoder.trainable_weights,
            *self.shape_encoder.trainable_weights,
            *[value for _, value in self.attention_network.items() if self._mlp_att != [0, 0, 0]],
            *self.mlp_output.trainable_weights
        ]
        grads = t.gradient(loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        return loss

    @tf.function
    def predict_item_batch(self, start, stop, item, shape, color, class_, return_attentive=False):
        gamma_u = self.Gu[start:stop]
        color_i = tf.expand_dims(color, 1)
        shape_i = tf.expand_dims(shape, 1)
        class_i = tf.expand_dims(class_, 1)

        if self._mlp_att != [0, 0, 0]:
            all_attention = self.propagate_attention_batch(gamma_u, color_i, shape_i, class_i)
            attentive_features = tf.reduce_sum(tf.multiply(
                all_attention,
                tf.expand_dims(tf.concat([color_i, shape_i, class_i], axis=1), axis=0)
            ), axis=2)
        else:
            attentive_features = tf.repeat(
                tf.expand_dims(tf.squeeze(color_i) + tf.squeeze(shape_i) + tf.squeeze(class_i), 0),
                repeats=gamma_u.shape[0], axis=0)

        # score prediction
        if self._out_feat_agg == 'multiplication':
            gamma_i = tf.expand_dims(item, 0) * attentive_features
        elif self._out_feat_agg == 'addition':
            gamma_i = tf.expand_dims(item, 0) + attentive_features
        else:
            raise NotImplementedError('This aggregation method has not been implemented yet!')
        xui = tf.squeeze(self.mlp_output(
            tf.concat([tf.repeat(tf.expand_dims(gamma_u, 1), repeats=gamma_i.shape[1], axis=1), gamma_i], axis=2),
            training=False))
        if return_attentive:
            return xui, tf.squeeze(all_attention)
        else:
            return xui

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    def get_config(self):
        raise NotImplementedError
