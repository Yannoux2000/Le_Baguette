import tensorflow as tf
import numpy as np

from os import listdir
SMRY_DIR = "/tmp/FUNModel/"
SMRY_DIR = SMRY_DIR + str(len(listdir(SMRY_DIR))) + "/"

GAMMA = 0.9					# reward discount factor
A_LR = 0.0001				# learning rate for actor
C_LR = 0.0002				# learning rate for critic
UPDATE_STEP = 10			# loop update operation n-steps
EPSILON = 0.2				# for clipping surrogate objective
SUB_DIM = 8					# number of subgoals dimension
GOAL_DIM = 4 				# dimention for each subgoals (4=> XYZ and interest)


class Model():

	def Policy_Net(self, name, A_DIM = 8, trainable = True, hidden = [200]):
		"""
		Policy for continuous actions decisions
		"""
		with tf.variable_scope(name):
			l1 = self.t_states

			for h in hidden:
				l1 = tf.layers.dense(l1, h, tf.nn.relu,
					kernel_initializer=tf.random_normal_initializer(stddev=0.5,mean=0.0), trainable=trainable)

			mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh,
				kernel_initializer=tf.random_normal_initializer(stddev=1e-5,mean=0.5), trainable=trainable)

			sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus,
				kernel_initializer=tf.random_normal_initializer(stddev=1e-6,mean=1e-5), trainable=trainable)

			sigma = tf.clip_by_value(sigma, -50, 50)

			norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

			tf.summary.histogram("mu", mu)
			tf.summary.histogram("sigma", sigma)

		params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
		return norm_dist, params

	def Policy_PPO_losses(self, pi, pi_params, oldpi):
		with tf.variable_scope("PPO_losses"):

			# ratio = tf.exp(pi.log_prob(self.t_actions) - oldpi.log_prob(self.t_actions))
			ratio = pi.prob(self.t_actions) / (oldpi.prob(self.t_actions) + 1e-5)
			surr = ratio * self.t_advantages # surrogate loss

			loss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.t_advantages))


			tf.summary.scalar("aloss", loss)
			tf.summary.histogram("ratio", ratio)

			# train_op = tf.train.AdamOptimizer(A_LR).minimize(loss)
			optzr = tf.train.AdamOptimizer(A_LR)
			grads = optzr.compute_gradients(loss,var_list=pi_params)

			grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
			smry_grad = tf.summary.merge([tf.summary.histogram("{}_grad".format(var.name), grad) for grad, var in grads])
			train_op = optzr.apply_gradients(grads)

		return train_op

	def Policy_A2C_losses(self, pi, pi_params):
		with tf.variable_scope("A2C_losses"):

			loss = -tf.reduce_mean(pi.log_prob(self.t_actions) * self.t_advantages)

			tf.summary.scalar("aloss", loss)

			# train_op = tf.train.AdamOptimizer(A_LR).minimize(loss)
			optzr = tf.train.AdamOptimizer(A_LR)
			grads = optzr.compute_gradients(loss,var_list=pi_params)

			grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
			smry_grad = tf.summary.merge([tf.summary.histogram("{}_grad".format(var.name), grad) for grad, var in grads])

			train_op = optzr.apply_gradients(grads)

		return train_op

	def Value_Net(self, name, hidden=[100]):

		with tf.variable_scope(name):
			# critic
			l1 = self.t_states

			for h in hidden:
				l1 = tf.layers.dense(l1, h, tf.nn.relu)

			v = tf.layers.dense(l1, 1)
			advantage = self.t_updateVal - v

			loss = tf.reduce_mean(tf.square(advantage))
			train_op = tf.train.AdamOptimizer(C_LR).minimize(loss)

			tf.summary.scalar("smry_value", tf.reduce_mean(v))
			tf.summary.scalar("closs", loss)

		return v, advantage, train_op

	def __init__(self , S_DIM, A_DIM): #to keep the S_Dim, A_Dim format in LeBaguette.py

		self.sess = tf.Session()

		self.ERROR = False	#As debugger is simple, use this as a anti-error-spam flag

		self.t_states = tf.placeholder(tf.float32, [None, S_DIM], 'states')
		self.t_updateVal = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

		tf.summary.scalar("mean_dr", tf.reduce_mean(self.t_updateVal))
		tf.summary.histogram("discounted_r", self.t_updateVal)

		self.t_subgoals = tf.placeholder(tf.float32, [None, SUB_DIM * GOAL_DIM], 'subgoal')

		self.t_actions = tf.placeholder(tf.float32, [None, A_DIM], 'action')
		self.t_advantages = tf.placeholder(tf.float32, [None, 1], 'advantage')

		# critic
		self.v, self.advantage, self.ctrain_op = self.Value_Net('value')

		# policy
		pi, pi_params = self.Policy_Net('pi', trainable=True)
		oldpi, oldpi_params = self.Policy_Net('oldpi', trainable=False)

		self.print_param = pi_params #debugs saving and other

		self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action

		self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

		self.atrain_op = self.Policy_PPO_losses(pi, pi_params, oldpi)

		#Stats
		self.step = tf.Variable(0)
		self.inc = tf.assign(self.step, self.step+1)

		self.smry = tf.summary.merge_all()
		
		self.w = tf.summary.FileWriter(SMRY_DIR)
		self.w.add_graph(self.sess.graph)
		#Stats

		print("Hello")

		self.sess.run(tf.global_variables_initializer())


	def update(self, s, a, r):

		adv = self.sess.run(self.advantage, {self.t_states: s, self.t_updateVal: r})

		smry = self.sess.run(self.smry, {self.t_states: s, self.t_actions: a, self.t_advantages: adv, self.t_updateVal: r})
		self.w.add_summary(smry, self.sess.run(self.step))
		self.sess.run(self.inc)

		# update actor and critic in a update loop
		[self.sess.run(self.atrain_op, {self.t_states: s, self.t_actions: a, self.t_advantages: adv}) for _ in range(UPDATE_STEP)]
		[self.sess.run(self.ctrain_op, {self.t_states: s, self.t_updateVal: r}) for _ in range(UPDATE_STEP)]

		self.sess.run(self.update_oldpi_op)	 # copy pi to old pi


	def choose_action(self, s):
		a = self.sess.run(self.sample_op, {self.t_states: [s]})[0]
		return np.clip(a, -1, 1)

	def get_v(self, s):
		return self.sess.run(self.v, {self.t_states: s})[0, 0]


		##Encapsulate Acting so the agent's model is hot-swappable
	def act(self, s):
		act = self.choose_action(s)

		for i,a in enumerate(act):
			if np.isnan(a):
				if not self.ERROR:
					self.ERROR = True
					print("ERROR") # Debuging
					print(act)
				act[i] = 0

		#Return actions to be outputed, and values to be reinputed in the training algorithm
		return act, act

		##Encapsulate
	def train(self, buffer_s, buffer_p, buffer_a, buffer_r):

		#Get current values
		v_s_ = self.get_v(buffer_s)

		#Discounted Rewards Calculus
		discounted_r = []
		for r in buffer_r[::-1]:
			v_s_ = r + GAMMA * v_s_
			discounted_r.append([v_s_])
		discounted_r.reverse()

		self.update(buffer_s, buffer_p, discounted_r) # [-20:]

		##Encapsulates supervised train
	def train_s(self, buffer_s, buffer_p, buffer_a, buffer_r):

		buffer_r = [[r] for r in buffer_r] ## Needs counter examples presented
		self.update(buffer_s, buffer_p, buffer_r) # [-20:]

	def continuous_action_to_probs(self, acts):
		return acts

		##SaverStuff to keep a save
	def init_saver(self):
		self.saver = tf.train.Saver(max_to_keep=24)

	def save(self, name):
		self.saver.save(self.sess, "./agents/LeBaguette/saves/{}".format(name)) # aim at this directory/saves/
		print(self.sess.run(self.print_param)[0]) #Confirm Load

	def load(self, name):
		self.saver.restore(self.sess, "./agents/LeBaguette/saves/{}".format(name)) # aim at this directory/saves/
		print(self.sess.run(self.print_param)[0]) #Confirm Load