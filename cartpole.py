#ACIT4610-1 21H Evolutionary artificial intelligence and robotics
# The ressource:

# https://perma.cc/C9ZM-652R
# https://coneural.org/florian/papers/05_cart_pole.pdf
# http://incompleteideas.net/sutton/book/code/pole.c
# https://casmodeling.springeropen.com/articles/10.1186/2194-3206-1-2
# https://github.com/hsayama/PyCX/blob/master/README.md
# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

# https://github.com/shilz1007/shilz1007/blob/main/float2bin.ipynb

#Task:
#1. Familiarize yourself with the models (CA, networks).
#2. Implement in Python a cellular automaton which receives argument(s) to define its rule.
#3. Familiarize yourself with the cart-pole balancing environment. You can install and prepare this environment by following the instructions on this link: https://gym.openai.com/docs/
#4. Come up with a method to encode input (environment observations) into the CA and to decode the CA state into output (action).
#5. Come up with a fitness function that tracks the performance of the controller.
#6. Evolve the rule of the CA to improve its control of the cart.
#7. Expand it to a network model (a simple neural network model with binary neurons). Then, evolve its parameters to improve the controller.

#using an artificial neural network
#or a cellular automaton
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from random import seed
from random import random

"one dimentisonlist, and two dim array for testing"
Dimlist1 = [0, 1, 9, 2, 0, 8, 7, 0, 4]
Dimlist2 = [[1, 12, 5, 2], [0, 6,10], [10, 8, 12, 5], [12,15,8,6]]
ruleNo = 10


# Initialize an artificial neural network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation


class CartPoleEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 3}
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        # state updates By seconde
        self.tau = 0.01
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        screen_width = 900
        screen_height = 500

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def float_bin(self, number, places=3):
        whole, dec = str(number).split(".")
        whole = int(whole)
        dec = int(dec)
        res = bin(whole).lstrip("0b") + "."
        for x in range(places):
            whole, dec = str((decimal_converter(dec)) * 2).split(".")
            dec = int(dec)
            res += whole

        return res

    def decimal_converter(self, num):
        while num > 1:
            num /= 10
        return num

    # n = 1.12345

    def CartPole(self, x):
        env = gym.make('CartPole-v0')
        env.reset()
        for _ in range(1):
            ##    env.render()
            ##    env.step(env.action_space.sample()) # take a random action
            o, r, d, i = env.step(env.action_space.sample())
            env.close()
        # print('observation', o)
        # print('reward', r)
        # print('done', d)
        # print('info', i)
        p = 8
        f1 = float_bin(o[0], places=p)
        f2 = float_bin(o[1], places=p)
        f3 = float_bin(o[2], places=p)
        f4 = float_bin(o[3], places=p)
        print(f1)
        print(f2)
        print(f3)
        print(f4)
        # converting to whole number
        f1_whole = convert_whole(f1)
        f2_whole = convert_whole(f2)
        f3_whole = convert_whole(f3)
        f4_whole = convert_whole(f4)

        print(f1_whole)
        print(f2_whole)
        print(f3_whole)
        print(f4_whole)



for i_episode in range(ruleNo):
    env = CartPoleEnv()
    observation = env.reset()

# defien new network
    seed(1)
    network = initialize_network(2, 1, 2)
    for layer in network:
        print(layer)

    #initialize_network()
    #observation = env.CartPole()

    #observation = env.render()
    #observation = env.close()

    for t in range(100):
        env.render(Dimlist2)
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break