#@Author - Christos Kouppas 
#@Secondary Author - Jordan Phillips
##For code written by Jordan is outlined in comments, imports were also overhauled

##The dependencies to ensure proper running of the code are listed within the readme accompanying this file.

##This code creates a DDPG agent based on 'rl.agents.ddpg', a model is created for the actor and critic of the DDPG agent as well as target networks respectively. 
##The DDPG agent is then compiled and fit to the model.
##After completion of the training, graphs are drawn from the training data.
##The agent is then tested using the new model just trained for 10 episodes where the output is displayed in the command line interface


from vrepper.core import vrepper
import time
import numpy as np
import math
import tensorflow as tf
from gym import spaces
import keras


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cProfile

from rl.agents.ddpg import DDPGAgent

import rl.memory
import tensorflow.keras.backend as K

from gym.utils import seeding

#Import the layers needed by the model to train and use
from tensorflow.keras.layers import Dense, Flatten, Concatenate, LSTM, Dropout
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate

#Use the Adam optimizer
from tensorflow.keras.optimizers import Adam

from rl.random import OrnsteinUhlenbeckProcess
import os

print(tf.__version__)

#Disable eager execution, since using tensorflow 2.2.0, it is enabled by default
tf.compat.v1.disable_eager_execution()

goal_steps = 500
score_requirement = 50
initial_games = 1000

epochs = 10  # Number or repeat of the whole Training
inner_epochs = 5  # Number of repeat of each training session
time_frame = 10  # The length of each time window
action_wait_time = 100
test_games = 10

batch_size = 32
batch_depth = 16
num_cores = 10

if False:
    num_GPU = 0
    num_CPU = 1
else:
    num_GPU = 0
    num_CPU = 1

#Configure GPUs if they are being used
config =  tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
        inter_op_parallelism_threads=num_cores, allow_soft_placement=False,
        device_count = {'CPU': num_CPU, 'GPU': num_GPU})
config.gpu_options.per_process_gpu_memory_fraction = 0.75

#Create the session for the code to run within
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

#############################Source 1: Christos Kouppas, Loughborough University PhD Student

#set up the environment
class CartPoleVREPEnv():
    def __init__(self,headless=False):
        self.venv = venv = vrepper(headless=headless)
        venv.start()
        time.sleep(0.1)
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Sarah_vortex_CPG.ttt')
        print(file_path)
        venv.load_scene(file_path)
        time.sleep(0.1)
        # Pressure Sensors
        self.L_Force_Front = venv.get_object_by_name('L_Force_F')
        self.L_Force_Back = venv.get_object_by_name('L_Force_B')
        self.R_Force_Front = venv.get_object_by_name('R_Force_F')
        self.R_Force_Back = venv.get_object_by_name('R_Force_B')

        # Angle Sensors / Actuators
        self.L_Abduction = venv.get_object_by_name('L_Abduction')
        self.R_Abduction = venv.get_object_by_name('R_Abduction')
        self.L_Hip_Angle = venv.get_object_by_name('L_Hip_Angle')
        self.R_Hip_Angle = venv.get_object_by_name('R_Hip_Angle')
        self.L_Linear = venv.get_object_by_name('L_Hip_Linear')
        self.R_Linear = venv.get_object_by_name('R_Hip_Linear')

        # Structural Components
        self.Center = venv.get_object_by_name('Central_Addon')
        print('(CartPoleVREP) initialized')
        obs = np.array([np.inf]*16)
        act = np.array([1.]*3)
        self.seed()

        self.action_space = spaces.Box(-act,act)
        self.observation_space = spaces.Box(-obs,obs)
        self.Frequency = 0
        self.Center_mass = 0
        self.observation = 0
        self.parameters = 0
        self.Time = 0

    def self_observe(self):
        # observe then assign
        LFF_Force, LFF_Torque = self.L_Force_Front.read_force_sensor()
        LFB_Force, LFB_Torque = self.L_Force_Back.read_force_sensor()
        RFF_Force, RFF_Torque = self.R_Force_Front.read_force_sensor()
        RFB_Force, RFB_Torque = self.R_Force_Back.read_force_sensor()
        LA_angle = self.L_Abduction.get_joint_angle()
        RA_angle = self.R_Abduction.get_joint_angle()
        LH_angle = self.L_Hip_Angle.get_joint_angle()
        RH_angle = self.R_Hip_Angle.get_joint_angle()
        LL_length = self.L_Linear.get_joint_angle()
        RL_length = self.R_Linear.get_joint_angle()

        IMU = np.array([0.]*6)
        IMU[0] = self.venv.get_float_signal('A_CA_X')
        IMU[1] = self.venv.get_float_signal('A_CA_Y')
        IMU[2] = self.venv.get_float_signal('A_CA_Z')
        IMU[3] = self.venv.get_float_signal('G_CA_X')
        IMU[4] = self.venv.get_float_signal('G_CA_Y')
        IMU[5] = self.venv.get_float_signal('G_CA_Z')
        self.observation = np.array([
            IMU[0], IMU[1], IMU[2], IMU[3], IMU[4], IMU[5],               # 6
            math.sqrt(LFF_Force[0]**2+LFF_Force[1]**2+LFF_Force[2]**2)/10,   # 1
            math.sqrt(LFB_Force[0]**2+LFB_Force[1]**2+LFB_Force[2]**2)/10,   # 1
            math.sqrt(RFF_Force[0]**2+RFF_Force[1]**2+RFF_Force[2]**2)/10,   # 1
            math.sqrt(RFB_Force[0]**2+RFB_Force[1]**2+RFB_Force[2]**2)/10,   # 1
            LA_angle, RA_angle,     # 2
            LH_angle, RH_angle,     # 2
            LL_length, RL_length    # 2
            ]).astype('float32')

    # The step function which is called for each step the robot takes during the testing and training
    def step(self, actions):
        self.parameters = np.around(actions*100, 0)
        self.venv.set_float_signal('Parameter_1', self.parameters[0])
        self.venv.set_float_signal('Parameter_2', self.parameters[1])
        self.venv.set_float_signal('Parameter_3', self.parameters[2])

        self.venv.step_blocking_simulation()


        # observe again
        self.self_observe()


        # cost
        self.Center_mass = self.Center.get_position() 
        self.Frequency = self.venv.get_float_signal('Oscillation_Frequency')
        
        # cost function
        cost = 2**(-(self.Frequency - 3)**2)
        if self.Time < 2:
            cost += 1e-6
        elif self.Time > 5:
            if self.Frequency < 0.5:
                 cost = self.Time/3000
            else:
                 cost *= self.Time/3000
        else:
            cost *= self.Time/30

        if (self.Center_mass[2] > 0.85) and (self.Center_mass[2] < 1) and (np.abs(self.Center_mass[0]) < 0.5) and \
                (np.abs(self.Center_mass[1]) < 0.5):
            done = 0
            if self.venv.get_integer_signal('Simulation_Stop') == 1:
                print('stopped')
                self.venv.stop_simulation()
                done = 1
                cost = 0
                print(self.parameters) #print the last step taken by the model on each episode
                
        else:
            done = 1
            cost = 0
            print(self.parameters) #print the last step taken by the model on each episode
            self.venv.stop_simulation()
        self.Time = self.venv.get_float_signal('Time')
        return self.observation, cost, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.venv.stop_simulation()
        time.sleep(0.01)
        self.venv.start_blocking_simulation()
        self.venv.step_blocking_simulation()
        self.venv.stream_integer_signal('Simulation_Stop')
        self.venv.stream_float_signal('A_CA_X')
        self.venv.stream_float_signal('A_CA_Y')
        self.venv.stream_float_signal('A_CA_Z')
        self.venv.stream_float_signal('G_CA_X')
        self.venv.stream_float_signal('G_CA_Y')
        self.venv.stream_float_signal('G_CA_Z')
        self.venv.stream_float_signal('Oscillation_Frequency')
        self.venv.stream_float_signal('Time')
        time.sleep(0.01)
        self.self_observe()

        return self.observation

    def destroy(self):
        self.venv.stop_simulation()
        self.venv.end()
        time.sleep(0.05)


def some_random_games_first():
    # Each of these is its own game.
    for episode in range(3):
        env.reset()
        # this is each frame, up to 200...but we wont make it that far.
        for t in range(100):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            # env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            if done:
                break


def neural_network_model(input_shape, trY):
    print('Creating um_Model...')
    model = Sequential()
    model.add(LSTM(nb_observation**2, dropout=0.1,
                batch_input_shape=(batch_size, input_shape[0], input_shape[1]), activation='tanh',
                   return_sequences=True, stateful=True))
    model.add(Dense(nb_observation**2, activation='sigmoid'))
    model.add(LSTM(nb_observation*5, dropout=0.1, activation='sigmoid', return_sequences=True, stateful=True))
    model.add(Dense(nb_observation*5, activation='sigmoid'))
    model.add(LSTM(nb_observation*5, dropout=0.1, activation='sigmoid',return_sequences=True,stateful=True))
    model.add(Dense(nb_observation*5, activation = 'sigmoid'))
    model.add(LSTM(nb_observation**2, dropout=0.1, activation='sigmoid', return_sequences=False,stateful=True))
    model.add(Dense(nb_observation**2, activation='sigmoid'))
    #model.add(LSTM(nb_observation**2, dropout=0.1, activation='sigmoid', return_sequences=False, stateful=True))
    model.add(Dense(trY, activation='linear', bias_initializer='ones'))
    # print(model.summary())
    return model
 

def V_model(input_shape):
    print('Creating V_Model...')
    V_model = Sequential()
    V_model.add(LSTM(nb_observation*5,  dropout=0.1,
                batch_input_shape=(batch_size, input_shape[0], input_shape[1]), activation='tanh',
                     return_sequences=True, stateful=True))
    V_model.add(Dense(nb_observation*5, activation='sigmoid'))
    V_model.add(LSTM(nb_observation*2, dropout=0.1, activation='sigmoid', return_sequences=True, stateful=True))
    V_model.add(Dense(nb_observation*2, activation = 'sigmoid'))
    V_model.add(LSTM(nb_observation*2, dropout=0.1, activation='sigmoid',return_sequences=True,stateful=True))
    V_model.add(Dense(nb_observation*2, activation = 'sigmoid'))
    V_model.add(LSTM(nb_observation*5, dropout=0.1, activation='sigmoid',return_sequences=False,stateful=True))
    V_model.add(Dense(nb_observation*5, activation='sigmoid'))
    #V_model.add(LSTM(nb_observation*5, dropout=0.1, activation='sigmoid', return_sequences=False, stateful=True))
    V_model.add(Dense(1, activation='linear'))
    # print(V_model.summary())
    return V_model


def L_model(input_shape, nb_actions):
    print('Creating L_Model...')
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(input_shape, name='observation_input')
    x = Concatenate()([action_input, Flatten()(observation_input)])
    # x = concatenate([Flatten()(observation_input),action_input])
    # x =Reshape((time_frame,nb_observation+nb_actions))(x)
    # x =LSTM(nb_observation*5,activation='sigmoid',return_sequences=True,stateful=False)(x)
    x = Dense(nb_observation, activation='tanh')(x)
    # x =LSTM(nb_observation*5,activation='sigmoid',return_sequences=True,stateful=False)(x)
    x = Dense(nb_observation, activation='sigmoid')(x)
    x = Dense(nb_observation, activation='sigmoid')(x)
    x = Dense(nb_observation, activation='sigmoid')(x)
    x = Dense(nb_observation, activation='sigmoid')(x)
    # x =LSTM(nb_observation*5,activation='sigmoid',return_sequences=False,stateful=False)(x)
    x = Dense((nb_actions * nb_actions + nb_actions) // 2, activation='linear')(x)  # ((nb_actions * nb_actions + nb_actions) / 2))
    L_model = Model(inputs=[action_input, observation_input], outputs=x)
    # print(L_model.summary())
    return L_model
    
##Clear the graph of the session to remove any mess left over by the session before - Jordan
K.clear_session()
tf.compat.v1.reset_default_graph()



##set up the VREP environment
env = CartPoleVREPEnv(headless=True)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

nb_observation = env.observation_space.shape[0]
some_random_games_first()


print("Random Played. Starting Population.")

memory = rl.memory.SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)

#########################################################End of source 1

# Create a placeholder for the input shape to ensure that this is enforced by the first layer of the actor - Jordan
input_shape_j = np.empty([16,1])

##########################################################
#BELOW IS CODE CREATED BY JORDAN PHILLIPS B623995#########
##########################################################

#######################ACTOR------START######################################
# The actor is created as a sequential model and all layers are configured, at the end, the model structure is outputted

actor = Sequential()

actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16))
actor.add(Activation('tanh'))
actor.add(Dense(32))
actor.add(Activation('tanh'))
actor.add(Dropout(0.05))
actor.add(Dense(32))
actor.add(Activation('tanh')) 
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))

print(actor.summary())


#######################ACTOR------END########################################

# Create the variables for the input to the critic network and flatten the observation input to match the shape of the network
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)


#######################CRITIC-------START####################################
# Create the critic network, ensure that the Reshape() comes from tensorflow, a workaround to a bug encountered during creation
# Print the summary of the network to the user in the command line interface


x = Input(batch_shape=(None, 16))
x = tf.keras.layers.Reshape((16,))(x)
x = Concatenate()([action_input, flattened_observation])
x = Dense(16)(x)
x = Activation('tanh')(x)
x = Dense(32)(x)
x = Activation('tanh')(x)
x = Dropout(0.05)(x)
x = Dense(32)(x)
x = Activation('tanh')(x)
x = Dense(1)(x)
x = Activation('tanh')(x)

critic = Model(inputs=[action_input, observation_input], outputs=x)

print(critic.summary())

#######################CRITIC-------END######################################

# Create the ddpg agent using the models that are defined above, define learning rate for target networks within and the discount factor, gamma
# gamma = 0.9 is the discount factor of future rewards
ddpg = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                         memory=memory, batch_size=32, nb_steps_warmup_critic=5000, nb_steps_warmup_actor=5000,
                         random_process=random_process, gamma=0.9, target_model_update=5e-3)
                         

# .compile() is used to configure the model with losses and metrics. 
# The learning rate of actor and critic are entered as arguments below respectfully 
ddpg.compile([Adam(lr=5e-4), Adam(lr=5e-3)], metrics=['mae'])

# show the metrics of the model that can be analysed in graphs
print(ddpg.metrics_names)


# .fit() is used to train the DDPG model
# 3000 max steps specified by Christos Kouppas
# Test the agent until it reaches 1 million total steps and print the output in a verbose 2 styling.
history = ddpg.fit(env, nb_steps = 1000000, visualize=False, verbose=2)

# set up variables to store agent data from training to be used in graphs
history_dict = history.history
print(history_dict.keys())
episode_rew = history.history['episode_reward']
episode_steps = history.history['nb_episode_steps']
number_steps = history.history['nb_steps']


# create a variable to store the amount of episodes completed in the 1 million step limit, 
# as this is a variable amount the range will need to be able to cater to this changing number
num_of_eps = len(episode_rew)
num_of_eps = range(0,num_of_eps)

print(num_of_eps)

#############################Print two 2D line graphs in a subplot of height = 2

plt.subplot(2, 1, 1)
plt.plot(num_of_eps, episode_rew, 'r')
plt.title('Graphs')
plt.ylabel('Episode Reward')

plt.subplot(2, 1, 2)
plt.plot(num_of_eps, episode_steps, 'g')
plt.xlabel('Episode')
plt.ylabel('Episode Steps')

plt.show()

#############################Print a 3D scatter graph

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_set = episode_rew
y_set = episode_steps
z_set = num_of_eps

ax.scatter(x_set, y_set, z_set, c='r', marker='o')

ax.set_xlabel('Episode Reward')
ax.set_ylabel('Episode Steps')
ax.set_zlabel('Episode')

plt.show()

#############################Print a 2D scatter graph

plt.scatter(episode_rew, episode_steps, c='r', alpha=0.5)
plt.title('A scatter graph of total steps taken with corresponding reward.')
plt.xlabel('Episode Reward')
plt.ylabel('Episode Steps')
plt.show()

plt.scatter(num_of_eps, episode_rew, c='b', alpha=0.5)
plt.title('A scatter graph of reward per episode.')
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.show()

#####################################################
##############Testing the model that has been trained
#####################################################
#The model is now tested for 10 episodes to see how it performs using the training it has undergone 


print("Online Testing")
env.destroy
time.sleep(1)
env2 = CartPoleVREPEnv(headless=True)
time.sleep(1)
ddpg.test(env2, nb_episodes=10, visualize=False, nb_max_episode_steps=3000)

# end simulation
print('simulation ended. leaving in 5 seconds...')
time.sleep(1)
