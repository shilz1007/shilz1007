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

import numpy as np
import math
import gym


#

# Function returns octal representation
def float_bin(number, places = 3):

	# split() seperates whole number and decimal
	# part and stores it in two seperate variables
	whole, dec = str(number).split(".")

	# Convert both whole number and decimal
	# part from string type to integer type
	whole = int(whole)
	dec = int (dec)

	# Convert the whole number part to it's
	# respective binary form and remove the
	# "0b" from it.
	res = bin(whole).lstrip("0b") + "."

	# Iterate the number of times, we want
	# the number of decimal places to be
	for x in range(places):

		# Multiply the decimal value by 2
		# and seperate the whole number part
		# and decimal part
		whole, dec = str((decimal_converter(dec)) * 2).split(".")

		# Convert the decimal part
		# to integer again
		dec = int(dec)

		# Keep adding the integer parts
		# receive to the result variable
		res += whole

	return res

# Function converts the value passed as
# parameter to it's decimal representation
def decimal_converter(num):
	while num > 1:
		num /= 10
	return num

#

def CA(*arguments):
    print("ok2")
    size = 50
    r = 1
    timeLim = 25
    size = timeLim * 2 + 1
    ## converting rule to binary
    ruleNo = int(rule)
    print("Role No:", ruleNo)
    binaryRule = format(ruleNo, '0' + str(np.power(2, (2 * r + 1))) + 'b')
    rules = list(binaryRule)  # split
    print("The output:", binaryRule)

    print("Good, Observation values passed")
    inp1 = obs1
    print("obs1: ", inp1)
    inp2 = obs2
    print("obs2: ", inp2)
    inp3 = obs3
    print("obs3: ", inp3)
    inp4 = obs4
    print("obs4: ", inp4)

    ## validating neighbour

    def validate(neighbors):

        #:param neighbors: array of neighbor cell state values
        #:return: next step cell state value
        b = ''
        for num in neighbors:
            b += str(num)
        index = np.power(2, (2 * r + 1)) - 1 - int(b, 2)
        return int(rules[index])

    ## Updating CA based on neighbours
    def update(u):
        #:param u: array of all cell state
        #:return: array of next step all cell state
        u_next = []
        for num in range(size):
            nbs = []
            for i in range(num - r, num + r + 1):
                nbs.append(u[i % size])
            u_next.append(validate(nbs))
        return u_next

    ## Implementing majority rule on the last timestep.
    def Majority(W_New):
        Zero_Add = 0
        One_Add = 0
        print(len(W_New))
        for y in range(len(W_New)):
            if W_New[y] == '0':
                # print('this is zero')
                # print(U[y])
                Zero_Add = Zero_Add + 1
            else:
                # print('this is 1')
                # print(U[y])
                One_Add = One_Add + 1
            if Zero_Add > One_Add:
                Action = 0
            else:
                Action = 1

        print('Zero_Add', Zero_Add)
        print('One_Add', One_Add)
        print('Action', Action)
        return Action

    ## Creating input to fill in the time step 0 of the CA
    def Create_Input(inp1, inp2, inp3, inp4):
        New_Arr = np.zeros(size, dtype=int)

        input1_whole = inp1 * 100000000
        input2_whole = inp2 * 100000000
        input3_whole = inp3 * 100000000
        input4_whole = inp4 * 100000000
        a_whole = math.trunc(input1_whole)
        b_whole = math.trunc(input2_whole)
        c_whole = math.trunc(input3_whole)
        d_whole = math.trunc(input4_whole)

        ## convert to list
        a_str = str(a_whole)
        b_str = str(b_whole)
        c_str = str(c_whole)
        d_str = str(d_whole)

        a_res = [int(X) for X in a_str]
        b_res = [int(X) for X in b_str]
        c_res = [int(X) for X in c_str]
        d_res = [int(X) for X in d_str]

        L1_Arr = np.asarray(a_res, dtype=np.int)
        L2_Arr = np.asarray(b_res, dtype=np.int)
        L3_Arr = np.asarray(c_res, dtype=np.int)
        L4_Arr = np.asarray(d_res, dtype=np.int)
        L5 = np.array([0, 0, 0])

        total_val = np.concatenate((L1_Arr, L5, L2_Arr, L5, L3_Arr, L5, L4_Arr), axis=None)

        if len(total_val) < len(U):
            for x in range(len(total_val)):
                New_Arr[x: len(total_val)] = total_val[x]

        print(New_Arr)
        return New_Arr

    U = np.zeros(size, dtype=np.int)
    U_input = Create_Input(inp1, inp2, inp3, inp4)
    print('value in U', U_input)
    U = U_input

    print('value in U', U)
    print('Length of U', len(U))

    W = np.array([U])
    print('w', W)
    # W_New = np.array([U])
    # W_Major = Majority(W_New)

    ## Runnig the CA for 25 time steps
    for j in range(timeLim):
        U = update(U)
        # if j == 25:
        #    print('length of U', len(U))
        #    print('length of W', len(W))
        W = np.vstack((W, U))
    W_New = np.array([U])
    W_Major = Majority(W_New)
    print('Final W_New', W_New)


#*************************
env = gym.make('CartPole-v0')
print("Plesase, Entner role number")
rule = input()
for i_episode in range(20):
    observation = env.reset()
    print("ok1")
    # *******
    for number in range(3):
        obs1 = float_bin(observation[0], 6)
        obs2 = float_bin(observation[1], 6)
        obs3 = float_bin(observation[2], 6)
        obs4 = float_bin(observation[3], 6)

    for t in range(100):
        env.render()
        #
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        CA(obs1, obs2, obs3, obs4, rule)

env.close()

