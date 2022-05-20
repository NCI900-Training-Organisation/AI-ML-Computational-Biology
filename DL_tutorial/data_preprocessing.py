import os
import argparse
from data_utils import *
import tensorflow as tf


def get_args():
    parser = argparse.ArgumentParser(description="Transcription Factor Data Preprocessing")
    parser.add_argument('-tf', '--transcription_factor', default='CTCF', type=str, nargs='+',
                        help='transcript factor')
    parser.add_argument('-m', '--peak_calling_method', default='_labels_fimo', type=str,
                        help='peak calling method')
    parser.add_argument('-n', '--number_of_samples', default=20000, type=int,
                        help='number of samples in each draw (default draw 10000 examples)')
    parser.add_argument('-d', '--draw_frequency', default=1, type=int,
                        help='draw frequency (default draw 1 time)')
    parser.add_argument('-r', '--random_seed', default=1, type=int,
                        help='fix random seed')
    parser.add_argument('-f', '--path', default='/g/data/ik06/stark/NCI_Leopard/', type=str,
                        help='save data to path')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)

    the_tf = args.transcription_factor
    num_sample = args.number_of_samples
    draw_times = args.draw_frequency
    random_seed = args.random_seed
    save_path = args.path
    batch_size = 100

    train_data_name = "train_data_"
    val_data_name = "validation_data_"
    test_data_name = "test_data_"

    data_dir = save_path + "preprocessed_" + ''.join(the_tf) + "_fimo_data/"
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    for i in range(draw_times):
        train_file_name = train_data_name + str(i)
        val_file_name = val_data_name + str(i)
        test_file_name = test_data_name + str(i)

        print("TRAINING DATA")
        train_dataset = tf.data.Dataset.from_tensor_slices(
            generate_data_batch(if_train=True, the_tf=the_tf,
                                num_sample=num_sample,
                                random_seed=random_seed))
        #train_dataset = train_dataset.batch(batch_size)
        tf.data.experimental.save(train_dataset, os.path.join(data_dir + train_file_name))

        print("Validation DATA")
        val_dataset = tf.data.Dataset.from_tensor_slices(
            generate_data_batch(if_train=False, the_tf=the_tf,
                                num_sample=num_sample * 0.2,
                                random_seed=random_seed))
        #val_dataset = val_dataset.batch(batch_size)
        tf.data.experimental.save(val_dataset, os.path.join(data_dir + val_file_name))

        print("TEST DATA")
        test_dataset = tf.data.Dataset.from_tensor_slices(
            generate_data_batch(if_train=False, if_test=True, the_tf=the_tf,
                                num_sample=num_sample * 0.2, random_seed=random_seed))
        #test_dataset = test_dataset.batch(batch_size)
        tf.data.experimental.save(test_dataset, os.path.join(data_dir + test_file_name))


if __name__ == '__main__':
    main()
