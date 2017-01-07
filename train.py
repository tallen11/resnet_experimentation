from resnet import ResNet
from data_batcher import DataBatcher
from time import time
import os
import tensorflow as tf

model = ResNet(5)

batcher = DataBatcher("cifar")
saver = tf.train.Saver()

epochs = 1000
batch_size = 512

with tf.Session() as session:
    print("Beginning training...")
    session.run(tf.global_variables_initializer())
    epoch_index = 0
    accuracy_data = []
    train_accuracy_data = []
    # step_index = 0
    epoch_start_time = time()
    while True:
        if batcher.epoch_finished():
            accuracy = 0
            image_batches, label_batches = batcher.get_test_batches(50)
            for i in range(len(image_batches)):
                accuracy += model.get_accuracy(session, image_batches[i], label_batches[i])
            accuracy /= len(image_batches)

            train_accuracy = 0
            image_batches, label_batches = batcher.get_test_training_batches(50)
            for i in range(len(image_batches)):
                train_accuracy += model.get_accuracy(session, image_batches[i], label_batches[i])
            train_accuracy /= len(image_batches)

            accuracy_data.append(accuracy)
            train_accuracy_data.append(train_accuracy)

            print("Epoch %i \t| test_acc: %f | train_acc: %f | time: %f" % (epoch_index, accuracy, train_accuracy, time() - epoch_start_time))
            saver.save(session, os.path.join("checkpoints/resnet_basic.ckpt"))
            batcher.prepare_epoch()
            step_index = 0
            epoch_start_time = time()
            if epoch_index == epochs:
                break
            epoch_index += 1
        images, labels = batcher.get_batch(batch_size)
        model.train_model(session, images, labels)
        # print("Step %i" % step_index)
        # step_index += 1
print("Training complete")
