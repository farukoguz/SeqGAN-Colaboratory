#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os
import random

from dataloader import dataset_for_generator, dataset_for_discriminator
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from rnnlm import RNNLM 


#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 16 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64
VOCAB_SIZE = 5000

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200 
positive_file = 'data/gan_train1.txt' # must be provided
negative_file = 'data/gan_fake.txt' # auto generated
eval_file = 'data/gan_test.txt' # auto generated
generated_num = 10000

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    assert START_TOKEN == 0

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)

    generator = Generator(VOCAB_SIZE, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    target_lstm = RNNLM(VOCAB_SIZE, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN) 
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=VOCAB_SIZE, embedding_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, dropout_keep_prob=dis_dropout_keep_prob,
                                  l2_reg_lambda=dis_l2_reg_lambda)
    
    gen_dataset = dataset_for_generator(positive_file, BATCH_SIZE)
    log = open('save/experiment-log.txt', 'w')
    #  pre-train generator
    if not os.path.exists("save/generator_pretrained.h5"):
        print('Start pre-training...')
        log.write('pre-training...\n')
        generator.pretrain(gen_dataset, target_lstm, PRE_EPOCH_NUM, generated_num // BATCH_SIZE, eval_file)
        generator.save("save/generator_pretrained.h5")
    else:
        generator.load("save/generator_pretrained.h5")

    if not os.path.exists("discriminator_pretrained.h5"):
        print('Start pre-training discriminator...')
        # Train 3 epoch on the generated data and do this for 50 times
        for _ in range(50):
            print("Dataset", _)
            generator.generate_samples(generated_num // BATCH_SIZE, negative_file)
            dis_dataset = dataset_for_discriminator(positive_file, negative_file, BATCH_SIZE)
            discriminator.train(dis_dataset, 3, (generated_num // BATCH_SIZE) * 2)
        discriminator.save("save/discriminator_pretrained.h5")
    else:
        discriminator.load("save/discriminator_pretrained.h5")

    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    
    for total_batch in range(TOTAL_BATCH):
        print("Generator", total_batch, 'of ', TOTAL_BATCH)
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate_one_batch()
            rewards = rollout.get_reward(samples, 16, discriminator)
            generator.train_step(samples, rewards)

        # Test
        if total_batch % 10 == 0 or total_batch == TOTAL_BATCH - 1:
            generator.generate_samples(generated_num // BATCH_SIZE, eval_file)
            likelihood_dataset = dataset_for_generator(eval_file, BATCH_SIZE)
            test_loss = target_lstm.target_loss(likelihood_dataset)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print('total_batch: ', total_batch, 'of: ', TOTAL_BATCH, 'test_loss: ', test_loss)
            generator.save(f"save/generator_{total_batch}.h5")
            discriminator.save(f"save/discriminator_{total_batch}.h5")
            log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        print("Discriminator", total_batch, 'of ', TOTAL_BATCH)
        # There will be 5 x 3 = 15 epochs in this loop
        for _ in range(5):
            generator.generate_samples(generated_num // BATCH_SIZE, negative_file)
            dis_dataset = dataset_for_discriminator(positive_file, negative_file, BATCH_SIZE)
            discriminator.train(dis_dataset, 3, (generated_num // BATCH_SIZE) * 2)
    generator.save(f"save/generator_{TOTAL_BATCH}.h5")
    discriminator.save(f"save/discriminator_{TOTAL_BATCH}.h5")

    log.close()


if __name__ == '__main__':
    main()
