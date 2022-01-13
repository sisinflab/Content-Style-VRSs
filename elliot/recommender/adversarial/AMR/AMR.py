"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from elliot.dataset.samplers import pairwise_pipeline_sampler_vbpr as vbprppsv
from elliot.recommender.adversarial.AMR import pairwise_pipeline_sampler_vbpr as ppsv
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.adversarial.AMR.AMR_model import AMR_model
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation


class AMR(RecMixin, BaseRecommenderModel):
    r"""
    Adversarial Multimedia Recommender

    For further details, please refer to the `paper <https://arxiv.org/pdf/1809.07062.pdf>`_

    The model support two adversarial perturbations methods:
        FGSM-based presented by X. He et al in `paper <https://arxiv.org/pdf/1809.07062.pdf>`

        MSAP presented by Anelli et al. in `paper <https://journals.flvc.org/FLAIRS/article/view/128443>`

    Args:
        meta:
            eval_perturbations: If True Elliot evaluates the effects of both FGSM and MSAP perturbations for each validation epoch
        factors: Number of latent factor
        factors_d: Image-feature dimensionality
        lr: Learning rate
        l_w: Regularization coefficient
        l_b: Regularization coefficient of bias
        l_e: Regularization coefficient of image matrix embedding
        eps: Perturbation Budget
        l_adv: Adversarial regularization coefficient
        adversarial_epochs: Adversarial epochs
        eps_iter: Size of perturbations in MSAP perturbations
        nb_iter: Number of Iterations in MSAP perturbations

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        AMR:
          meta:
            save_recs: True
            eval_perturbations: True
          epochs: 10
          batch_size: 512
          factors: 200
          factors_d: 20
          lr: 0.001
          l_w: 0.1
          l_b: 0.001
          l_e: 0.1
          eps: 0.1
          l_adv: 0.001
          adversarial_epochs: 5
          eps_iter: 0.00001
          nb_iter: 20
          nb_iter: 20
          eps_iter: 0.00001  # If not specified = 2.5*eps/nb_iter

    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a AMR instance.
        (see https://arxiv.org/pdf/1809.07062.pdf for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      [eps, l_adv]: adversarial budget perturbation and adversarial regularization parameter,
                                      lr: learning rate}
        """
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users

        self._params_list = [
            ("_batch_eval", "batch_eval", "be", 512, int, None),
            ("_factors", "factors", "factors", 200, int, None),
            ("_factors_d", "factors_d", "factors_d", 20, int, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_l_w", "l_w", "l_w", 0.1, None, None),
            ("_l_b", "l_b", "l_b", 0.001, None, None),
            ("_l_e", "l_e", "l_e", 0.1, None, None),
            ("_eps", "eps", "eps", 0.1, None, None),
            ("_l_adv", "l_adv", "l_adv", 0.001, None, None),
            ("_eps_iter", "eps_iter", "eps_iter", None, None, None),
            ("_nb_iter", "nb_iter", "nb_iter", 1, None, None),
            ("_adversarial_epochs", "adversarial_epochs", "adv_epochs", self._epochs // 2, int, None)
        ]
        self.autoset_params()

        if self._adversarial_epochs > self._epochs:
            raise Exception(f"The total epoch ({self._epochs}) "
                            f"is smaller than the adversarial epochs ({self._adversarial_epochs}).")

        if self._eps_iter is None:
            self._eps_iter = 2.5 * self._eps / self._nb_iter

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        item_indices = [self._data.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._sampler = ppsv.Sampler(self._data.i_train_dict,
                                     item_indices,
                                     self._data.side_information_data.visual_feature_path,
                                     self._epochs)

        self._next_batch = self._sampler.pipeline(self._data.transactions, self._batch_size)

        self._model = AMR_model(self._factors,
                                self._factors_d,
                                self._learning_rate,
                                self._l_w,
                                self._l_b,
                                self._l_e,
                                self._data.visual_features_shape,
                                self._num_users,
                                self._num_items,
                                self._eps,
                                self._l_adv,
                                self._batch_size)

        self._vbpr_sampler = vbprppsv.Sampler(self._data.i_train_dict,
                                     item_indices,
                                     self._data.side_information_data.visual_feature_path,
                                     self._epochs)

        self._next_eval_batch = self._vbpr_sampler.pipeline_eval(self._batch_eval)

    @property
    def name(self):
        return "AMR" \
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
        user_adv_train = (self._epochs - it) <= self._adversarial_epochs

        with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:

            for batch in self._next_batch:
                steps += 1
                loss += self._model.train_step(batch, user_adv_train)
                t.set_postfix({'(APR)-loss' if user_adv_train else '(BPR)-loss': f'{loss.numpy() / steps:.5f}'})
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

    def carmine_train(self):
        if self._restore:
            return self.restore_weights()

        loss = 0
        steps = 0
        it = 0
        user_adv_train = (self._epochs - it) <= self._adversarial_epochs
        with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
            for batch in self._next_batch:
                steps += 1
                loss += self._model.train_step(batch, user_adv_train)
                t.set_postfix({'(APR)-loss' if user_adv_train else '(BPR)-loss': f'{loss.numpy() / steps:.5f}'})
                t.update()

                if steps == self._data.transactions // self._batch_size:
                    t.reset()
                    self.evaluate(it, loss.numpy() / steps)
                    it += 1
                    steps = 0
                    loss = 0
                    user_adv_train = (self._epochs - it) <= self._adversarial_epochs

                    if getattr(self._params.meta, "eval_perturbations", False):
                        self.evaluate_perturbations(it)

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_eval)):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            print('\tUser {0}\{1}'.format(offset_stop, self._num_users))
            predictions = np.empty((offset_stop - offset, self._num_items))
            for batch in self._next_eval_batch:
                item, feat = batch
                print('\t\tItem {0}\{1}'.format(item[-1], self._num_items))
                p = self._model.predict_item_batch(offset, offset_stop,
                                                   item[0], item[-1],
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
                print('Rec lists correctly Restored')
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
