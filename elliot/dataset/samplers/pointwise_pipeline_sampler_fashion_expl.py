import tensorflow as tf
from PIL import Image

import numpy as np
import random
np.random.seed(42)
random.seed(42)


class Sampler:
    def __init__(self, indexed_ratings, item_indices, shapes_path, colors_path, classes_path, output_shape_size, epochs):
        self._indexed_ratings = indexed_ratings
        self._item_indices = item_indices
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

        self._shapes_path = shapes_path
        self._colors_path = colors_path
        self._classes_path = classes_path
        self._output_shape_size = output_shape_size
        self._epochs = epochs

    def read_images_triple(self, user, item, pos_neg):
        # load positive and negative item images
        im = Image.open(self._shapes_path + str(item.numpy()) + '.tiff')
        # if np.array(im).shape[-1] == 3:
        #     im = im.convert('L')

        color = np.load(self._colors_path + str(item.numpy()) + '.npy')

        class_ = np.load(self._classes_path + str(item.numpy()) + '.npy')

        try:
            im.load()
        except ValueError:
            print(f'Image at path {item}.tiff was not loaded correctly!')

        im = np.expand_dims(np.array(im.resize(self._output_shape_size)) / np.float32(255.0), axis=2)

        if np.max(np.abs(color)) != 0:
            color = color / np.max(np.abs(color))

        return user.numpy(), item.numpy(), pos_neg.numpy(), im, color, class_

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        actual_inter = (events // batch_size) * batch_size * self._epochs

        counter_inter = 1

        def sample():
            u = r_int(n_users)
            ui = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                sample()
            b = random.getrandbits(1)
            if b:
                i = ui[r_int(lui)]
            else:
                i = r_int(n_items)
                while i in ui:
                    i = r_int(n_items)
            return u, i, b

        for ep in range(self._epochs):
            for _ in range(0, events):
                yield sample()
                if counter_inter == actual_inter:
                    return
                else:
                    counter_inter += 1

    def pipeline(self, num_users, batch_size):
        def load_func(u, i, p_n):
            b = tf.py_function(
                self.read_images_triple,
                (u, i, p_n,),
                (np.int64, np.int64, np.int64, np.float32, np.float32, np.float32)
            )
            return b

        data = tf.data.Dataset.from_generator(generator=self.step,
                                              output_shapes=((), (), ()),
                                              output_types=(tf.int64, tf.int64, tf.int64),
                                              args=(num_users, batch_size))
        data = data.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    def step_eval(self):
        for i in self._item_indices:
            yield i

    # this is only for evaluation
    def pipeline_eval(self):
        def load_func(i):
            b = tf.py_function(
                self.read_image,
                (i,),
                (np.int64, np.float32, np.float32, np.float32)
            )
            return b

        data = tf.data.Dataset.from_generator(generator=self.step_eval,
                                              output_shapes=(()),
                                              output_types=tf.int64)
        data = data.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    # this is only for evaluation
    def read_image(self, item):
        shape = Image.open(self._shapes_path + str(item.numpy()) + '.tiff')
        color = np.load(self._colors_path + str(item.numpy()) + '.npy')
        class_ = np.load(self._classes_path + str(item.numpy()) + '.npy')

        try:
            shape.load()
        except ValueError:
            print(f'Image at path {item}.jpg was not loaded correctly!')

        im = np.expand_dims(np.array(shape.resize(self._output_shape_size)) / np.float32(255.0), axis=2)

        if np.max(np.abs(color)) != 0:
            color = color / np.max(np.abs(color))

        return item, im, color, class_

    def read_feature(self, item):
        """
        Args:
            item: Integer

        Returns:
            item id, shape, color, and class
        """
        shape = Image.open(self._shapes_path + str(item) + '.tiff')
        # if np.array(shape).shape[-1] == 3:
        #     shape = shape.convert('L')

        color = np.load(self._colors_path + str(item) + '.npy')
        class_ = np.load(self._classes_path + str(item) + '.npy')

        try:
            shape.load()
        except ValueError:
            print(f'Image at path {item}.tiff was not loaded correctly!')

        shape = np.expand_dims(np.array(shape.resize(self._output_shape_size)) / np.float32(255.0), axis=2)

        if np.max(np.abs(color)) != 0:
            color = color / np.max(np.abs(color))

        return item, shape, color, class_
