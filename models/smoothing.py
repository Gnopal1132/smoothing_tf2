import os.path
import tensorflow as tf
import numpy as np
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint


class Smooth:
    ABSTAIN = -1

    def __init__(self, path,
                 num_classes: int,
                 sigma: float):
        """

        :param path: Is the base classifier graph and weight path
        :param num_classes: Number of classes
        :param sigma: Noise level Hyperparam.
        """
        self.graph_path = os.path.join(path, 'graph.json')
        self.model_weight = os.path.join(path, 'best.h5')
        self.classes = num_classes
        self.sigma = sigma
        self.classifier = self.load_model()

    def load_model(self):
        json_file = open(self.graph_path, 'r')
        load_json = json_file.read()
        json_file.close()

        model = tf.keras.models.model_from_json(load_json)
        model.load_weights(self.model_weight)
        return model

    def certify(self, x: tf.constant,
                n0: int,
                n1: int,
                alpha: float,
                batchsize: int) -> (int, float):
        """
        Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: Input of dimension [Width X Height X Classes]
        :param n0: Number of Monte carlo Samples for selection
        :param n1: Number of Monte carlo Samples for estimation
        :param alpha: Failure probability
        :param batchsize: Batchsize
        :return: (predicted class, certified radius)
        """

        # Lets draw samples from f(x+noise)
        selection_count = self._sample_noise(x, n0, batchsize)
        # Top guess for the current class
        top_class = tf.argmax(selection_count, axis=1)

        # drawing more samples for evaluation
        estimation_count = self._sample_noise(x, n1, batchsize)

        # Estimating lower bound on PA
        nA = estimation_count[top_class]
        pABar = self._lower_confidence_bound(nA, n1, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return top_class, radius

    def predict(self, x: tf.constant, n: int, alpha: float, batchsize: int) -> int:
        """Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: Is the input of dimension [Width X Height X Channel]
        :param n: Number of Monte Carlo Samples to use
        :param alpha: Failure probability
        :param batchsize: Batch Size
        :return: Returns -1 in case of Abstain else returns the class predicted by function g
        """

        counts = self._sample_noise(x, n, batchsize)
        top = counts.argsort()[::-1][:2]  # Extracting the index of the first two top classes
        count1 = counts[top[0]]
        count2 = counts[top[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top[0]

    def _sample_noise(self, x: tf.constant,
                      num: int,
                      batchsize: int):
        """
        :param x: [Width X Height X Channels]
        :param num: Number of samples to take
        :param batchsize: represents the batch size
        :return: Returns the per class count
        """
        counts = np.zeros(self.classes, dtype=int)
        for _ in range(int(np.ceil(num / batchsize))):
            current_batch = min(num, batchsize)
            num -= batchsize

            batch = tf.tile(tf.expand_dims(x, axis=0), multiples=[current_batch, 1, 1, 1])
            noise = tf.random.normal(batch.shape) * self.sigma
            distribution = self.classifier(tf.cast(batch, dtype=tf.float32) + noise)
            prediction = tf.argmax(distribution, axis=1)
            # Returning the highest predicted class index
            counts += self.count_arr(prediction.numpy(), self.classes)
        return counts

    def count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
