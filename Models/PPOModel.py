import tensorflow as tf
import numpy as np

GAMMA = 0.9					# reward discount factor
A_LR = 0.0001				# learning rate for actor
C_LR = 0.0002				# learning rate for critic
UPDATE_STEP = 10			# loop update operation n-steps
EPSILON = 0.2				# for clipping surrogate objective
A_DIM = 8					# action dimension

class Model():

	def Policy_Net(self, name, trainable, hidden=[200]):
		with tf.variable_scope(name):
			l1 = self.t_states
			for h in hidden:
				l1 = tf.layers.dense(l1, h, tf.nn.relu,
					kernel_initializer=tf.random_normal_initializer(stddev=0.5,mean=0.0), trainable=trainable)

			mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh,
				kernel_initializer=tf.random_normal_initializer(stddev=0.1,mean=0.5), trainable=trainable)

			sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus,
				kernel_initializer=tf.random_normal_initializer(stddev=0.1,mean=0.5), trainable=trainable)

			norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

		params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
		return norm_dist, params

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

		return v, advantage, train_op

	def __init__(self , S_DIM, _): #to keep the S_Dim, A_Dim format in LeBaguette.py

		self.sess = tf.Session()

		self.ERROR = False	#As debugger is simple, use this as a anti-error-spam flag

		self.t_states = tf.placeholder(tf.float32, [None, S_DIM], 'states')
		self.t_updateVal = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

		self.t_actions = tf.placeholder(tf.float32, [None, A_DIM], 'action')
		self.t_advantages = tf.placeholder(tf.float32, [None, 1], 'advantage')

		# critic
		self.v, self.advantage, self.ctrain_op = self.Value_Net('value')

		# actor
		pi, pi_params = self.Policy_Net('pi', trainable=True)
		oldpi, oldpi_params = self.Policy_Net('oldpi', trainable=False)

		self.print_param = pi_params #debugs saving and other

		self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
		self.sample_old_op = tf.squeeze(oldpi.sample(1), axis=0)  # operation of choosing action

		self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

		# ratio = tf.exp(pi.log_prob(self.t_actions) - oldpi.log_prob(self.t_actions))
		ratio = pi.prob(self.t_actions) / (oldpi.prob(self.t_actions) + 1e-5)
		surr = ratio * self.t_advantages # surrogate loss

		self.aloss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.t_advantages))
		
		self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
		self.sess.run(tf.global_variables_initializer())

	def update(self, s, a, r):

		self.sess.run(self.update_oldpi_op)	 # copy pi to old pi

		adv = self.sess.run(self.advantage, {self.t_states: s, self.t_updateVal: r})
		# update actor and critic in a update loop
		[self.sess.run(self.atrain_op, {self.t_states: s, self.t_actions: a, self.t_advantages: adv}) for _ in range(UPDATE_STEP)]
		[self.sess.run(self.ctrain_op, {self.t_states: s, self.t_updateVal: r}) for _ in range(UPDATE_STEP)]


	def choose_action(self, s):
		a = self.sess.run(self.sample_op, {self.t_states: [s]})[0]
		return np.clip(a, -1, 1)

	def old_action(self,s):
		a = self.sess.run(self.sample_old_op, {self.t_states: [s]})[0]
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

	def old_act(self,s):
		act = self.old_action(s)

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


		##SaverStuff to keep a save :thinking:
	def init_saver(self):
		self.saver = tf.train.Saver(max_to_keep=24, keep_checkpoint_every_n_hours=1)

	def save(self, name):
		self.saver.save(self.sess, "./agents/LeBaguette/TF_Model_Saves/{}".format(name)) # aim at this directory/TF_Model_Saves/
		print(self.sess.run(self.print_param)[0]) #Confirm Load


	def load(self, name):
		self.saver.restore(self.sess, "./agents/LeBaguette/TF_Model_Saves/{}".format(name)) # aim at this directory/TF_Model_Saves/
		print(self.sess.run(self.print_param)[0]) #Confirm Load