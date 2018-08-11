import random
from collections import deque
import os

import numpy as np

import tensorflow as tf
import cv2

from flappy_bird.flappy_bird import FlappyBird

N_ACTION = 2
DROPOUT = 0.75

GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100000.  # timesteps to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.0001  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1


with tf.device('/gpu:0'):
    tf.reset_default_graph()

    weights = {
        'wc1': tf.Variable(tf.truncated_normal([8, 8, 4, 32])),
        'wc2': tf.Variable(tf.truncated_normal([4, 4, 32, 64])),
        'wc3': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
        'wf1': tf.Variable(tf.truncated_normal([256, 256])),
        'out': tf.Variable(tf.truncated_normal([256, N_ACTION]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bc3': tf.Variable(tf.random_normal([64])),
        'bf1': tf.Variable(tf.random_normal([256])),
        'out': tf.Variable(tf.random_normal([N_ACTION]))
    }

    s = tf.placeholder(tf.float32, [None, 80, 80, 4])
    keep_prob = tf.placeholder(tf.float32)
    a = tf.placeholder(tf.float32, [None, N_ACTION])
    y = tf.placeholder(tf.float32, [None])

    conv1 = tf.nn.conv2d(s, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, biases['bc1'])
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, biases['bc2'])
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv3 = tf.nn.conv2d(conv2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, biases['bc3'])
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    fc1 = tf.reshape(conv3, [-1, weights['wf1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wf1']), biases['bf1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, DROPOUT)

    readout = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    game_state = FlappyBird()

    D = deque()

    do_nothing = np.zeros(N_ACTION)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # 저장 불러오기 에러 개짱나네
    checkpoint = tf.train.get_checkpoint_state('./save')
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")


    # start training
    epsilon = INITIAL_EPSILON
    t = 0

    episode_length = 10000
    episode = 0
    while episode != episode_length:
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([N_ACTION])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(N_ACTION)
                a_t[random.randrange(N_ACTION)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        if terminal:
            episode += 1
            print(episode)

        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 1000 iterations
        if t % 1000 == 0:
            saver.save(sess, './save/flappy_bird-dqn', global_step=t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        #print("TIMESTEP", t, "/ STATE", state, \
        #      "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
        #      "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        #if t % 1000 <= 100:
            # a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            # h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
        #if True:
        #    cv2.imwrite(current_dir + 'logs/frame' + str(t) + '.png', x_t1)
