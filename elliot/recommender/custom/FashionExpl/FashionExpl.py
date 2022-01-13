import os
import time
from ast import literal_eval as make_tuple

import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm

from elliot.recommender.base_recommender_model import init_charger

import elliot.dataset.samplers.pairwise_pipeline_sampler_fashion_expl as pairpfs
import elliot.dataset.samplers.pointwise_pipeline_sampler_fashion_expl as pointpfs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.custom.FashionExpl.FashionExpl_model import FashionExpl_model
from elliot.utils.write import store_recommendation

np.random.seed(0)
tf.random.set_seed(0)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class FashionExpl(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random

        self._params_list = [
            ("_factors", "factors", "f", 100, int, None),
            ("_batch_eval", "batch_eval", "be", 128, int, None),
            ("_learning_rate", "lr", "lr", 0.0005, None, None),
            ("_l_w", "l_w", "lw", 0.000025, None, None),
            ("_mlp_color", "mlp_color", "mlpc", "(64,1)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_cnn_channels", "cnn_channels", "cnnch", 32, None, None),
            ("_cnn_kernels", "cnn_kernels", "cnnk", 3, None, None),
            ("_cnn_strides", "cnn_strides", "cnns", 1, None, None),
            ("_mlp_cnn", "mlp_cnn", "mlpcnn", "(64,1)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_mlp_att", "mlp_att", "mlpa", "(64,1)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_mlp_out", "mlp_out", "mlpo", "(64,1)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_dropout", "dropout", "d", 0.2, None, None),
            ("_temperature", "temperature", "t", 1.0, None, None),
            ("_att_feat_agg", "att_feat_agg", "afa", "multiplication", str, None),
            ("_out_feat_agg", "out_feat_agg", "ofa", "multiplication", str, None),
            ("_sampler_str", "sampler", "s", "pairwise", str, None)
        ]

        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._item_indices = [self._data.item_mapping[self._data.private_items[item]] for item in
                              range(self._num_items)]

        if self._restore:
            # dictionary with key (user) and value (attention weights for user's positive items)
            self._attention_dict = dict()

        if self._sampler_str == 'pairwise':
            self._sampler = pairpfs.Sampler(self._data.i_train_dict,
                                            self._item_indices,
                                            self._data.side_information_data.shapes_src_folder,
                                            self._data.side_information_data.visual_color_feature_path,
                                            self._data.side_information_data.visual_class_feature_path,
                                            self._data.output_shape_size,
                                            self._epochs)
        elif self._sampler_str == 'pointwise':
            self._sampler = pointpfs.Sampler(self._data.i_train_dict,
                                             self._item_indices,
                                             self._data.side_information_data.shapes_src_folder,
                                             self._data.side_information_data.visual_color_feature_path,
                                             self._data.side_information_data.visual_class_feature_path,
                                             self._data.output_shape_size,
                                             self._epochs)
        else:
            raise NotImplementedError('This sampler type has not been implemented for this model yet!')

        self._next_batch = self._sampler.pipeline(self._data.transactions, self._batch_size)

        self._model = FashionExpl_model(self._factors,
                                        self._mlp_color,
                                        self._mlp_att,
                                        self._mlp_out,
                                        self._mlp_cnn,
                                        self._cnn_channels,
                                        self._cnn_kernels,
                                        self._cnn_strides,
                                        self._att_feat_agg,
                                        self._out_feat_agg,
                                        self._sampler_str,
                                        self._temperature,
                                        self._dropout,
                                        self._learning_rate,
                                        self._l_w,
                                        self._num_users,
                                        self._num_items)

        # only for evaluation purposes
        # self._next_eval_batch = self._sampler.pipeline_eval()

    @property
    def name(self):
        return "FashionExpl" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        # print('Test Eval')
        # self.get_recommendations(self.evaluator.get_needed_recommendations())
        # print('End Eval')

        if self._restore:
            is_restored = self.restore_weights()
            if is_restored:
                return True
            else:
                print('This Model will start the training!')

        best_metric_value = 0
        loss = 0
        steps = 0
        it = 0
        early_stopping = 5

        with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
            for batch in self._next_batch:
                steps += 1
                loss += self._model.train_step(batch)
                t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                t.update()

                # epoch is over
                if steps == self._data.transactions // self._batch_size:
                    t.reset()
                    if not (it + 1) % self._validation_rate:
                        recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                        result_dict = self.evaluator.eval(recs)
                        self._results.append(result_dict)

                        self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / steps:.3f}')

                        if self._results[-1][self._validation_k]["val_results"][
                            self._validation_metric] > best_metric_value:
                            best_metric_value = self._results[-1][self._validation_k]["val_results"][
                                self._validation_metric]
                            early_stopping = 5
                            if self._save_weights:
                                self._model.save_weights(self._saving_filepath)
                            if self._save_recs:
                                # store_recommendation(recs,
                                #                      self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")
                                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")
                        else:
                            early_stopping -= 1
                            if early_stopping == 0:
                                print('Reached Early Stopping Condition at Epoch {0}\n\tEXIT'.format(it + 1))
                                break
                    it += 1
                    steps = 0
                    loss = 0

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        print('Start Evaluation')
        print('\tExtracting Low-level Features')
        s = time.time()
        colors_feat = np.zeros(shape=(len(self._item_indices), self._factors))
        shapes_feat = np.zeros(shape=(len(self._item_indices), self._factors))
        classes_feat = np.zeros(shape=(len(self._item_indices), self._factors))
        for start_batch in range(0, len(self._item_indices), (self._batch_size * 2)):
            stop_batch = min(start_batch + (self._batch_size * 2), len(self._item_indices))
            colors = np.zeros(shape=(stop_batch - start_batch, self._data.visual_color_features_shape))
            shapes = np.zeros(shape=(stop_batch - start_batch, *self._data.output_shape_size, 1))
            classes = np.zeros(shape=(stop_batch - start_batch, self._data.visual_class_features_shape))
            for start_feature in range(start_batch, stop_batch):
                _, shape, color, class_ = self._sampler.read_feature(start_feature)
                colors[start_feature % (self._batch_size * 2)] = color
                shapes[start_feature % (self._batch_size * 2)] = shape
                classes[start_feature % (self._batch_size * 2)] = class_
            colors_feat[start_batch:stop_batch] = self._model.color_encoder(colors, training=False).numpy()
            shapes_feat[start_batch:stop_batch] = self._model.shape_encoder(shapes, training=False).numpy()
            classes_feat[start_batch:stop_batch] = classes
            print('\t\tFeatures batch {0}\\{1}'.format(stop_batch, len(self._item_indices)))
        print('\tLow-level Features Extracted in {0} sec.'.format((time.time() - s)))
        print('\tExtracting Users RecSys Lists...')
        with tf.device('/cpu:0'):
            if self._restore:
                all_attention = np.empty((self._num_users, self._num_items, 3))
            for index, offset in enumerate(range(0, self._num_users, self._batch_eval)):
                offset_stop = min(offset + self._batch_eval, self._num_users)
                print('\tUser {0}\\{1}'.format(offset_stop, self._num_users))
                predictions = np.empty((offset_stop - offset, self._num_items))
                if self._restore:
                    attention = np.empty((offset_stop - offset, self._num_items, 3))
                for item_index, item_offset in enumerate(range(0, self._num_items, self._batch_eval)):
                    item_offset_stop = min(item_offset + self._batch_eval, self._num_items)
                    if self._restore:
                        predictions[:(offset_stop - offset), item_index * self._batch_eval:item_offset_stop], \
                        attention[:(offset_stop - offset), item_index * self._batch_eval:item_offset_stop,
                        :] = self._model.predict_item_batch(offset, offset_stop,
                                                            self._model.Gi[
                                                            item_index * self._batch_eval:item_offset_stop],
                                                            tf.Variable(
                                                                shapes_feat[
                                                                item_index * self._batch_eval:item_offset_stop],
                                                                dtype=tf.float32),
                                                            tf.Variable(
                                                                colors_feat[
                                                                item_index * self._batch_eval:item_offset_stop],
                                                                dtype=tf.float32),
                                                            tf.Variable(
                                                                classes_feat[
                                                                item_index * self._batch_eval:item_offset_stop],
                                                                dtype=tf.float32), True)
                    else:
                        predictions[:(offset_stop - offset),
                        item_index * self._batch_eval:item_offset_stop] = self._model.predict_item_batch(offset,
                                                                                                         offset_stop,
                                                                                                         self._model.Gi[
                                                                                                         item_index * self._batch_eval:item_offset_stop],
                                                                                                         tf.Variable(
                                                                                                             shapes_feat[
                                                                                                             item_index * self._batch_eval:item_offset_stop],
                                                                                                             dtype=tf.float32),
                                                                                                         tf.Variable(
                                                                                                             colors_feat[
                                                                                                             item_index * self._batch_eval:item_offset_stop],
                                                                                                             dtype=tf.float32),
                                                                                                         tf.Variable(
                                                                                                             classes_feat[
                                                                                                             item_index * self._batch_eval:item_offset_stop],
                                                                                                             dtype=tf.float32))
                    mask = self.get_train_mask(offset, offset_stop)
                    v, i = self._model.get_top_k(predictions, mask, k=k)
                    items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                          for u_list in list(zip(i.numpy(), v.numpy()))]
                    if (item_index + 1) % 10 == 0:
                        print('\t\tItem {0}\\{1}'.format(item_offset_stop, self._num_items))
                predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
                if self._restore:
                    # Build Attention Dictionary
                    s = time.time()
                    all_attention[offset:offset_stop, :, :] = attention
                    self._attention_dict = {**self._attention_dict,
                                            **{u_abs: {
                                                'profile': attention[u_rel, self._data.sp_i_train.toarray()[u_abs] == 1],
                                                'test': attention[u_rel, list(self._data.test_dict[u_abs].keys())[0]]
                                            } for u_abs, u_rel in zip(range(offset, offset_stop), range(offset_stop - offset))}}
                    print("\nBuild Attention Dict in {0} seconds".format((time.time() - s)))

        if self._restore:
            with open(self._config.path_output_rec_result + f"{self.name}_attention.pkl",
                      "wb") as f:
                pickle.dump(self._attention_dict, f)
            np.save(self._config.path_output_rec_result + f"{self.name}_all_attention.npy", all_attention)

        return predictions_top_k

    def restore_weights(self):
        try:
            self._model.load_weights(self._saving_filepath)
            print(f"Model correctly Restored")

            try:
                print('Try to restore rec lists')
                recs = self.restore_recommendation(path=self._config.path_output_rec_result + f"{self.name}.tsv")
            except Exception as error:
                print(f'** Error in Try to restore rec lists\n\t{error}\n')
                print('Evaluate rec lists')
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())

            result_dict = self.evaluator.eval(recs)
            self._results.append(result_dict)

            print("******************************************")
            if self._save_recs:
                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")
            return True

        except Exception as ex:
            print(f"Error in model restoring operation! {ex}")
            return False

    def restore_recommendation(self, path=""):
        """
        Store recommendation list (top-k)
        :return:
        """
        recommendations = {}
        with open(path, 'r') as fin:
            while True:
                line = fin.readline().strip().split('\t')
                if line[0] == '':
                    break
                u = int(line[0])
                i = int(line[1])
                r = float(line[2])

                if u not in recommendations:
                    recommendations[u] = []
                recommendations[u].append((i, r))

        return recommendations