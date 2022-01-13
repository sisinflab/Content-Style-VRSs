"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

import os
from ast import literal_eval as make_tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from elliot.recommender.base_recommender_model import init_charger

import elliot.dataset.samplers.pairwise_pipeline_sampler_acf as ppsa
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.visual_recommenders.ACF.ACF_model import ACF_model
from elliot.utils.write import store_recommendation

np.random.seed(0)
tf.random.set_seed(0)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ACF(RecMixin, BaseRecommenderModel):
    r"""
    Attentive Collaborative Filtering: Multimedia Recommendation with Item- and Component-Level Attention

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3077136.3080797>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        layers_component: Tuple with number of units for each attentive layer (component-level)
        layers_item: Tuple with number of units for each attentive layer (item-level)

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        ACF:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          factors: 100
          batch_size: 128
          l_w: 0.000025
          layers_component: (64, 1)
          layers_item: (64, 1)
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random

        self._layers_component = self._params.layers_component
        self._layers_item = self._params.layers_item

        self._params_list = [
            ("_factors", "factors", "factors", 100, None, None),
            ("_learning_rate", "lr", "lr", 0.0005, None, None),
            ("_l_w", "l_w", "l_w", 0.000025, None, None),
            ("_layers_component", "layers_component", "layers_component", "(64,1)", lambda x: list(make_tuple(str(x))), lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_layers_item", "layers_item", "layers_item", "(64,1)", lambda x: list(make_tuple(str(x))), lambda x: self._batch_remove(str(x), " []").replace(",", "-"))
        ]

        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        item_indices = [self._data.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._sampler = ppsa.Sampler(self._data.i_train_dict,
                                     item_indices,
                                     self._data.side_information_data.visual_feat_map_feature_path,
                                     self._data.visual_feat_map_features_shape,
                                     self._epochs)

        self._next_batch = self._sampler.pipeline(self._data.transactions, self._batch_size)

        self._model = ACF_model(self._factors,
                                self._layers_component,
                                self._layers_item,
                                self._learning_rate,
                                self._l_w,
                                self._data.visual_feat_map_features_shape,
                                self._num_users,
                                self._num_items)

        # only for evaluation purposes
        self._next_eval_batch = self._sampler.pipeline_eval()

    @property
    def name(self):
        return "ACF" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        # print('Start Test Recommendation')
        # self.get_recommendations(self.evaluator.get_needed_recommendations())
        # print('End Test Recommendation')

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
                            early_stopping = 5
                            best_metric_value = self._results[-1][self._validation_k]["val_results"][
                                self._validation_metric]
                            if self._save_weights:
                                self._model.save_weights(self._saving_filepath)
                            if self._save_recs:
                                # store_recommendation(recs,
                                #                      self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")
                                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")

                        else:
                            early_stopping -= 1
                            if early_stopping == 0:
                                print('Reached Early Stopping Condition at Epoch {0}\n\tEXIT'.format(it+1))
                                break
                    it += 1
                    steps = 0
                    loss = 0

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for user_id, batch in enumerate(self._next_eval_batch):
            user, user_pos, feat_pos = batch
            predictions = self._model.predict(user, user_pos, feat_pos)
            mask = self.get_train_mask(user.numpy(), user.numpy() + 1)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(user.numpy(), user.numpy() + 1), items_ratings_pair)))
            print('\tUser {0}/{1}'.format(user_id+1, self._num_users))
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
