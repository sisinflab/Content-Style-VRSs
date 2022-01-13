"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

import gc
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from elliot.dataset.samplers import pairwise_pipeline_sampler_dvbpr as ppsd
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.visual_recommenders.DVBPR.DVBPR_model import DVBPR_model
from elliot.utils.write import store_recommendation

np.random.seed(0)
tf.random.set_seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class DVBPR(RecMixin, BaseRecommenderModel):
    r"""
    Visually-Aware Fashion Recommendation and Design with Generative Image Models

    For further details, please refer to the `paper <https://doi.org/10.1109/ICDM.2017.30>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        lambda_1: Regularization coefficient
        lambda_2: CNN regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        DVBPR:
          meta:
            save_recs: True
          lr: 0.0001
          epochs: 50
          factors: 100
          batch_size: 128
          lambda_1: 0.0001
          lambda_2: 1.0
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random

        self._params_list = [
            ("_batch_eval", "batch_eval", "be", 512, int, None),
            ("_factors", "factors", "factors", 100, None, None),
            ("_learning_rate", "lr", "lr", 0.0001, None, None),
            ("_lambda_1", "lambda_1", "lambda_1", 0.0001, None, None),
            ("_lambda_2", "lambda_2", "lambda_2", 1.0, None, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict
        self._item_indices = [self._data.item_mapping[self._data.private_items[item]] for item in
                              range(self._num_items)]

        self._sampler = ppsd.Sampler(
            self._data.i_train_dict,
            self._item_indices,
            self._data.side_information_data.images_src_folder,
            self._data.output_image_size,
            self._epochs
        )
        self._next_batch = self._sampler.pipeline(self._data.transactions, self._batch_size)

        self._model = DVBPR_model(self._factors,
                                  self._learning_rate,
                                  self._lambda_1,
                                  self._lambda_2,
                                  self._num_users,
                                  self._num_items)

        # only for evaluation purposes
        # self._next_eval_batch = self._sampler.pipeline_eval(self._batch_size)

    @property
    def name(self):
        return "DVBPR" \
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
                                print('Reached Early Stopping Condition at Epoch {0}\n\tEXIT'.format(it+1))
                                break
                    it += 1
                    steps = 0
                    loss = 0

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        print('Start Evaluation')

        print('-> Extracting Image Features')
        features = np.zeros(shape=(len(self._item_indices), self._factors))
        for start_batch in range(0, len(self._item_indices), self._batch_eval):
            stop_batch = min(start_batch + self._batch_eval, len(self._item_indices))
            images = np.zeros(shape=(stop_batch - start_batch, *self._data.output_image_size, 3))
            for start_image in range(start_batch, stop_batch):
                _, image = self._sampler.read_image(start_image)
                images[start_image % self._batch_eval] = image
            features[start_batch:stop_batch] = self._model.Cnn(images, training=False).numpy()
            if start_batch % 100 == 0:
                print('\t\tImages {0}\\{1}'.format((stop_batch), len(self._item_indices)))
        print('Image Features Extracted')

        for index, offset in enumerate(range(0, self._num_users, self._batch_eval)):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            print('\tUser {0}\\{1}'.format(offset_stop, self._num_users))
            predictions = np.empty((offset_stop - offset, self._num_items))
            for item_index, item_offset in enumerate(range(0, self._num_items, self._batch_eval)):
                item_offset_stop = min(item_offset + self._batch_eval, self._num_items)
                p = self._model.predict_item_batch(offset, offset_stop,
                                                   tf.Variable(features[item_index * self._batch_eval:item_offset_stop],
                                                               dtype=tf.float32))
                predictions[:(offset_stop - offset), item_index * self._batch_eval:item_offset_stop] = p
                if item_index % 10 == 0:
                    print('\t\tEvaluated Items {0}\\{1}'.format(item_offset_stop, self._num_items))
            mask = self.get_train_mask(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))

        del features
        gc.collect()
        print('End Evaluation')
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