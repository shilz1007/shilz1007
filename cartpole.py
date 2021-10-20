#ACIT4610-1 21H Evolutionary artificial intelligence and robotics
# The ressource:

#https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
# https://perma.cc/C9ZM-652R
# https://coneural.org/florian/papers/05_cart_pole.pdf
# http://incompleteideas.net/sutton/book/code/pole.c
# https://casmodeling.springeropen.com/articles/10.1186/2194-3206-1-2
# https://github.com/hsayama/PyCX/blob/master/README.md
# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# https://towardsdatascience.com/comparing-optimal-control-and-reinforcement-learning-using-the-cart-pole-swing-up-openai-gym-772636bc48f4


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
from scipy import linalg


"one dimentisonlist, and two dim array for testing"
Dimlist1 = [0, 1, 9, 2, 0, 8, 7, 0, 4]
Dimlist2 = [[1, 12, 5, 2], [0, 6,10], [10, 8, 12, 5], [12,15,8,6]]
ruleNo = 10

#define the linearized dynamics of the system:
# state matrix
g= 54
mp= 72
lp = 15
mk = 0.5
mt =73
a = g/(lp*(4.0/3 - mp/(mp+mk)))
A = np.array([[0, 1, 0, 0],
              [0, 0, a, 0],
              [0, 0, 0, 1],
              [0, 0, a, 0]])

# input matrix
b = -1/(lp*(4.0/3 - mp/(mp+mk)))
B = np.array([[0], [1/mt], [0], [b]])

#calculate the optimal controller:
R = np.eye(1, dtype=int)          # choose R (weight for input)
Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)

# get riccati solver
from scipy import linalg

# solve ricatti equation
P = linalg.solve_continuous_are(A, B, Q, R)

# calculate optimal controller gain
K = np.dot(np.linalg.inv(R),  np.dot(B.T, P))

#define a function, to call to actually calculate the input force F during runtime:
def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left


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

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "done = True. You "
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def apply_state_controller(K, x):
        # feedback controller
        u = -np.dot(K, x)  # u = -Kx
        if u > 0:
            return 1, u  # if force_dem > 0 -> move cart right
        else:
            return 0, u  # if force_dem <= 0 -> move cart left

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



# get environment
env = gym.make('CartPole-v0')
env.env.seed(1)  # seed for reproducibility
obs = env.reset()




print("Test controller")
# get environment
env = gym.make('CartPole-v0')
env.env.seed(1)  # seed for reproducibility
obs = env.reset()

for i in range(1000):
    env.render()

    # get force direction (action) and force value (force)
    action, force = apply_state_controller(K, obs)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))

    # change magnitute of the applied force in CartPole
    env.env.force_mag = abs_force

    # apply action
    obs, reward, done, _ = env.step(action)
    if done:
        print(f'Terminated after {i + 1} iterations.')
        break

env.close()
