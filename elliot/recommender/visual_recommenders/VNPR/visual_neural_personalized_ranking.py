"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

from ast import literal_eval as make_tuple

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from elliot.dataset.samplers import pairwise_pipeline_sampler_vnpr as ppsv
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.visual_recommenders.VNPR.visual_neural_personalized_ranking_model import VNPRModel
from elliot.utils.write import store_recommendation

np.random.seed(42)


class VNPR(RecMixin, BaseRecommenderModel):
    r"""
    Visual Neural Personalized Ranking for Image Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3159652.3159728>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        mf_factors:: Number of latent factors for Matrix Factorization
        mlp_hidden_size: Tuple with number of units for each multi-layer perceptron layer
        prob_keep_dropout: Dropout rate for multi-layer perceptron
        batch_size: Batch size
        l_w: Regularization coefficient for user and item
        l_v: Regularization coefficient for visual

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        VNPR:
          meta:
            save_recs: True
          lr: 0.001
          epochs: 50
          mf_factors: 10
          mlp_hidden_size: (32, 1)
          prob_keep_dropout: 0.2
          batch_size: 64
          l_w: 0.001
          l_v: 0.001
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._random = np.random

        self._params_list = [
            ("_batch_eval", "batch_eval", "be", 512, int, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_l_w", "l_w", "l_w", 0.001, None, None),
            ("_l_v", "l_v", "l_v", params.l_w, None, None),
            ("_mf_factors", "mf_factors", "mffactors", 10, None, None),
            ("_mlp_hidden_size", "mlp_hidden_size", "mlpunits", "(32,1)", lambda x: list(make_tuple(str(x))), lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_dropout", "dropout", "drop", 0.2, None, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        item_indices = [self._data.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._sampler = ppsv.Sampler(self._data.i_train_dict,
                                     item_indices,
                                     self._data.side_information_data.visual_pca_feature_path,
                                     self._epochs)

        self._next_batch = self._sampler.pipeline(self._data.transactions, self._batch_size)

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = VNPRModel(self._num_users, self._num_items, self._mf_factors, self._l_w, self._l_v,
                                                     self._mlp_hidden_size,
                                                     self._dropout,
                                                     self._learning_rate,
                                                     self._data.visual_pca_features_shape)

        # only for evaluation purposes
        self._next_eval_batch = self._sampler.pipeline_eval(self._batch_eval)

    @property
    def name(self):
        return "VNPR"\
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):

        # print('*** Test Eval ***')
        # self.get_recommendations(self.evaluator.get_needed_recommendations())
        # print('*** END Test Eval ***')

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
        for index, offset in enumerate(range(0, self._num_users, self._batch_eval)):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            predictions = np.empty((offset_stop - offset, self._num_items))
            print('\tUser {0}\{1}'.format(offset_stop, self._num_users))
            for batch in self._next_eval_batch:
                item, feat = batch
                print('\t\tItem {0}\{1}'.format(item[-1], self._num_items))
                mf_item_embedding_1 = self._model.item_mf_embedding_1(item)
                mf_item_embedding_2 = self._model.item_mf_embedding_2(item)
                p = self._model.predict_item_batch(offset, offset_stop, mf_item_embedding_1, mf_item_embedding_2,
                                                   tf.Variable(feat))
                predictions[:(offset_stop - offset), item] = p
            mask = self.get_train_mask(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
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
