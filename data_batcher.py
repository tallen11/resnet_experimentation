import pickle
import os
from os import walk
import random
import numpy as np

class DataBatcher:
    def __init__(self, data_path):
        data_paths = []
        test_path = ""
        for dir_path, dir_names, file_names in walk(data_path):
            for name in file_names:
                if "data_batch_" in name:
                    data_paths.append(os.path.join(dir_path, name))
                elif "test_batch" in name:
                    test_path = os.path.join(dir_path, name)
            break
        train_data_dicts = [dict(self.__unpickle(n)) for n in data_paths]
        test_data_dict = dict(self.__unpickle(test_path))

        self.train_samples = []
        for index, train_dict in enumerate(train_data_dicts):
            print("Loading training file %i" % index)
            pairs = self.__load_tensor_label_pairs(train_dict)
            self.train_samples += pairs

        print("Loading test file")
        self.test_samples = self.__load_tensor_label_pairs(test_data_dict)
        self.epoch_set = list(self.train_samples)

    def epoch_finished(self):
        return len(self.epoch_set) == 0

    def prepare_epoch(self):
        self.epoch_set = list(self.train_samples)

    def get_batch(self, size):
        if size >= len(self.epoch_set):
            size = len(self.epoch_set)
        batch = [self.epoch_set.pop(random.randrange(len(self.epoch_set))) for _ in range(size)]
        return self.__package_batch(batch, size)

    def get_test_batch(self):
        return self.__package_batch(self.test_samples, len(self.test_samples))

    def get_test_batches(self, sizes):
        full_images, full_labels = self.__package_batch(self.test_samples, len(self.test_samples))
        image_batches = np.split(full_images, len(self.test_samples) // sizes, axis=0)
        label_batches = np.split(full_labels, len(self.test_samples) // sizes, axis=0)
        return image_batches, label_batches
        # samples = [self.test_samples[i:i+sizes] for i in range(0, len(self.test_samples), sizes)]
        # image_batches = []
        # label_batches
        # for sample in samples:
        #     batches.append(self.__package_batch(sample, len(sample)))

    def get_test_training_batches(self, sizes):
        full_images, full_labels = self.__package_batch(self.train_samples, len(self.train_samples))
        image_batches = np.split(full_images, len(self.train_samples) // sizes, axis=0)
        label_batches = np.split(full_labels, len(self.train_samples) // sizes, axis=0)
        return image_batches, label_batches

    def __package_batch(self, batch, size):
        image_tensors = []
        label_tensors = []
        for image, label in batch:
            image_tensors.append(image)
            label_tensor = np.zeros(10)
            label_tensor[label] = 1
            label_tensors.append(label_tensor)
        return np.array(image_tensors, dtype=np.float32).reshape([size,32,32,3]), np.array(label_tensors, dtype=np.float32).reshape([size,10])

    def __load_tensor_label_pairs(self, data_dict):
        d = data_dict[b"data"]
        l = data_dict[b"labels"]

        raw = np.array(d, dtype=np.float32) / 255.0
        images = raw.reshape([-1,3,32,32])
        images = images.transpose([0,2,3,1])
        images = np.split(images, 10000, axis=0)
        pairs = [(images[i], l[i]) for i in range(10000)]
        return pairs

    def __unpickle(self, file):
        with open(file, "rb") as fo:
            dic = pickle.load(fo, encoding="bytes")
        return dic
