import tensorflow as tf
import numpy as np
import retro
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from collections import deque       # (Double Ended Queue) data type that removes the oldest element when new 1 is added
import random
import warnings
import dqn                          # Our implementation of the DQN
import mem                          # Our implementation of Memory
import argparse
import sys
import os
warnings.filterwarnings('ignore')
import logging

# Poor python coder's global vars
env = None
possible_actions = None
stacked_frames = None
state = None
new_state = None
memory = None
writer = None
write_op = None

# Environment params
stack_size = 4                      # Number of frames stacked to counter Temporal limitations
state_size = [110, 84, 4]           # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
action_size = None   # 8 possible actions
learning_rate = 0.00025             # Alpha (aka learning rate)

# Training params
total_episodes = 50                 # Total episodes for training
max_steps = 50000                   # Max possible steps in an episode
batch_size = 64                     # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0                 # exploration probability at start
explore_stop = 0.01                 # minimum exploration probability
decay_rate = 0.00001                # exponential decay rate for exploration prob

# Q learning hyper-parameters
gamma = 0.9                         # Discounting rate

# Memory params
pretrain_length = batch_size        # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000               # Number of experiences the Memory can keep

# MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = False

# Render the environment (novelty wears off after first couple of runs)
episode_render = True

sess = None
DQNetwork = None

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", required=False, type=bool, default=False,
                help="Set debug to tree to see debug messages, otherwise they are logged to log file only")
args = vars(ap.parse_args())


def preprocess_frame(frame):
    try:
        # convert to gray to eliminate depth (pixels go to 0-1 rather than 0-255)
        gray = rgb2gray(frame)

        # crop the scene to eliminate some pixels
        cropped_frame = gray[8:-12, 4:-12]

        # Normalize the gray scale values
        normalized_frame = cropped_frame/255.0
        preprocessed_frame = transform.resize(normalized_frame, [110, 84])
    except Exception as err:
        print("Error in preprocess_frame function: {}".format(err))
        pass

    return preprocessed_frame


# Due to temporal limitations - we need to create stacks of frames, rather than single frames
def stack_frames(stacked_frames, state, is_new_episode):
    try:
        frame = preprocess_frame(state)
        if is_new_episode:
            stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for _ in range(stack_size)], maxlen=4)

            # New episode means we take the first frame and copy it 4 times, the deque wil rotate in new frames soon
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)

            stacked_state = np.stack(stacked_frames, axis=2)

        else:
            # Add new frame and deque removes oldest frame
            stacked_frames.append(frame)

            stacked_state = np.stack(stacked_frames, axis=2)
    except Exception as err:
        print("Error in stack_frames function: {}".format(err))
        pass

    return stacked_state, stacked_frames


def engage_memory():
    global stacked_frames
    global possible_actions
    global state
    global new_state
    global memory
    global pretrain_length
    global env

    # Create some memory
    try:
        memory = mem.Memory(max_size=memory_size, batch_size=batch_size)
        for i in range(pretrain_length):
            if i is 0:
                state = env.reset()
                state, stacked_frames = stack_frames(stacked_frames, state, True)

            # Take a random action
            choice = random.randint(1, len(possible_actions)) - 1
            action = possible_actions[choice]
            next_state, reward, done, _ = env.step(action)

            if episode_render:
                env.render()

            # stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            # If the episode is complete and we have died 3 times
            if done:
                next_state = np.zeros(state.shape)

                # add the experience to memory
                memory.add((state, action, reward, next_state, done))

                # start a new episode
                state = env.reset()

                # stack the frames
                state, stacked_frames = stack_frames(stacked_frames, state, True)
            else:
                # add the experience to memory
                memory.add((state, action, reward, next_state, done))

                # update new_state to be current state
                state = next_state
    except Exception as err:
        print("Error in engage_memory function: {}".format(err))


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    global possible_actions
    global sess
    global DQNetwork

    action = None
    explore_probability = None
    try:
        # Epsilon greedy
        exp_exp_tradeoff = np.random.rand()

        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        if explore_probability > exp_exp_tradeoff:
            choice = random.randint(1, len(possible_actions))-1
            action = possible_actions[choice]
        else:
            # Get action from our  q-network
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

            # Take the biggest Q value
            choice = np.argmax(Qs)
            action = possible_actions[choice]
    except Exception as err:
        print("Error in function predict_action: {}".format(err))
        pass

    return action, explore_probability


def train():
    global total_episodes
    global stacked_frames
    global possible_actions
    global decay_rate
    global state
    global explore_start
    global explore_stop
    global env
    global memory
    global writer
    global write_op
    global DQNetwork

    try:
        saver = tf.train.Saver()
        with sess:
            sess.run(tf.global_variables_initializer())
            decay_step = 0
            for episode in range(total_episodes):
                step = 0
                episode_rewards = []
                state = env.reset()
                state, stacked_frames = stack_frames(stacked_frames, state, True)

                while step < max_steps:
                    step += 1

                    # Increase decay
                    decay_step += 1

                    # predict the action to take and take it
                    action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step,
                                                                 state, possible_actions)
                    print("Next predicted action is {}".format(action))

                    # Perform the action and get the next_state, reward and done info
                    next_state, reward, done, _ = env.step(action)

                    if episode_render:
                        env.render()

                    episode_rewards.append(reward)

                    if done:
                        next_state = np.zeros((110, 84), dtype=np.int)
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                        step = max_steps
                        total_reward = np.sum(episode_rewards)

                        print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Explore P: {:.4f}'.format(explore_probability),
                              'Training Loss {:.4f}'.format(DQNetwork.loss))


                        # Commit to memory
                        memory.add((state, action, reward, next_state, done))
                    else:
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                        # Commit to memory
                        memory.add((state, action, reward, next_state, done))
                        # next_state now becomes current state
                        state = next_state

                    batch = memory.sample(batch_size)


                    states_mb = np.vstack([np.expand_dims(x, 0) for x in batch])
                    #states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.vstack([np.expand_dims(x, 1) for x in batch])
                    #actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.vstack([np.expand_dims(x, 2) for x in batch])
                    #rewards_mb = np.array([each[2] for each in batch])
                    next_states_mb = np.vstack([np.expand_dims(x, 3) for x in batch])
                    #next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.vstack([np.expand_dims(x, 4) for x in batch])
                    #dones_mb = np.array([each[4] for each in batch])

                    target_Qs_batch  = []

                    # Debug
                    print("Network output: {}".format(DQNetwork.output.shape))
                    print("Network inputs: {}".format(DQNetwork.inputs_.shape))
                    print("Next_states_mb: {}".format(next_states_mb.shape))

                    # Get Q value prediction for the next_state
                    Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]

                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])
                        else:
                            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)

                    targets_mb = np.array([each[0] for each in target_Qs_batch])

                    loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                       feed_dict={DQNetwork.inputs_: states_mb,
                                                  DQNetwork.target_Q: targets_mb,
                                                  DQNetwork.actions_: actions_mb})

                    # Write TF Summaries
                    summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                            DQNetwork.target_Q: targets_mb,
                                                            DQNetwork.actions_: actions_mb})
                    writer.add_summary(summary, episode)
                    writer.flush()

                    # Save model every 5 episodes
                if episode % 5 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model Saved")

    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error in train function: {} \n {} \n {} \n {}".format(err, exc_type, fname, exc_tb.tb_lineno))


def start():
    global env
    global state_size
    global action_size
    global learning_rate
    global stacked_frames
    global possible_actions
    global DQNetwork
    global writer
    global write_op
    global sess
    try:

        print("Initializing new SpaceInvaders training")

        # Set it up
        env = retro.make(game='SpaceInvaders-Atari2600')
        print("Retro environment set up")
        action_size = env.action_space.n

        # initialize empty stacked_frames
        stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for _ in range(stack_size)], maxlen=4)
        possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
        print("Observable state space is: {}".format(env.observation_space))
        print("Observable action size is: {}".format(env.action_space.n))

        # Engage memory module
        engage_memory()
        print("Memory module engaged")

        print("Configuring TensorFlow...")
        tf.reset_default_graph()

        # Create the network
        DQNetwork = dqn.DQNetwork(state_size, action_size, learning_rate)
        print("Network created")
        sess = tf.Session()

        # Setup tensorboard
        writer = tf.summary.FileWriter("/tensorboard/dqn/1")
        tf.summary.scalar("Loss", DQNetwork.loss)
        write_op = tf.summary.merge_all()
        print("Tensorboard set, begin training...")

        train()

    except Exception as err:
        print("Error in start function: {}".format(err))


start()