import tensorflow as tf
print("tensorflow imported")
import numpy as np

class Model:
	def Predictor(self, t_obs, t_prev_obs, t_act, prev_size, obs_size, act_shape, hidden=[200,100,20], lr=0.01):

		last_layer = tf.reshape(t_prev_obs,[-1, prev_size * obs_size])
		last_layer = tf.concat([last_layer, t_act],1)
		for h_size in hidden:
			last_layer = tf.nn.relu(tf.layers.dense(last_layer,h_size))

		predicted = tf.layers.dense(last_layer,obs_size)

		loss = tf.square(predicted - t_obs)
		optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

		return (predicted, optimizer)

	def Policy(self, t_obs, t_prob, t_adv, act_shape, name="Main", hidden=[200,100,20], lr=0.01, b=0):
		with tf.variable_scope("policy" + name):

			last_layer = t_obs
			for h_size in hidden:
				last_layer = tf.nn.relu(tf.layers.dense(last_layer,h_size))
			linear = tf.layers.dense(last_layer,sum(act_shape))

			splited_probs = tf.split(linear,act_shape,axis=1)
			
			output = []
			for splitout in splited_probs: # apply softmax to each actions to maximize one value for each actions
				output.append(tf.nn.softmax(splitout))
			probabilities = tf.concat(output,axis=1)

			with tf.name_scope("RL_diffs_and_loss"):

				splited_probs = tf.split(t_prob,act_shape,axis=1)
				rl_loss = sl_loss = 0
				for out,prob in zip(output,splited_probs):

					esperance = tf.reduce_sum(tf.multiply(probabilities, t_prob),reduction_indices=[1])
					eligibility = tf.log(esperance) * t_adv
					rl_loss = rl_loss + tf.reduce_mean(eligibility)
					sl_loss = sl_loss + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=prob))

				# esperance = tf.reduce_sum(tf.multiply(probabilities, t_prob),reduction_indices=[1])
				# eligibility = tf.log(esperance) * t_adv
				# rl_loss = -tf.reduce_mean(eligibility)

				rl_optimizer = tf.train.AdamOptimizer(lr).minimize(rl_loss)

			with tf.name_scope("SL_diffs_and_loss"):
				loss = -tf.reduce_mean(tf.square(probabilities - t_prob))

			mimic_optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
		return (probabilities, rl_optimizer, mimic_optimizer)

	def Value(self,t_obs,t_newvalue, hidden=[200,100,20],lr=0.01):
		with tf.variable_scope("value"):
			
			prev = t_obs
			for hid_size in hidden:
				prev = tf.nn.relu(tf.layers.dense(prev,hid_size))
			value = tf.layers.dropout(tf.layers.dense(prev,1))

			with tf.name_scope("diffs_and_loss"):
				advantages = value - t_newvalue
				loss = tf.square(advantages)

			optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
		return (value, optimizer, advantages)


	def __init__(self,obs_size,act_shape, prev_size = 5, predict_n = 10):
		
		self.obs_size = obs_size
		self.act_size = sum(act_shape)
		self.act_shape = act_shape
		self.prev_size = prev_size
		self.predict_n = predict_n

		self.t_obs = tf.placeholder(tf.float32, [None,self.obs_size])
		self.t_newvalue = tf.placeholder(tf.float32, [None, 1])

		self.t_prob = tf.placeholder(tf.float32, [None,self.act_size])

		self.t_prev_obs = tf.placeholder(tf.float32, [None,self.prev_size,self.obs_size])
		self.t_act = tf.placeholder(tf.float32, [None,len(self.act_shape)])

		self.value, self.opt_v, t_adv = self.Value(self.t_obs, self.t_newvalue)
		self.probs, self.opt_p, self.opt_mp = self.Policy(self.t_obs, self.t_prob, tf.stop_gradient(t_adv),self.act_shape)
		self.preds, self.opt_f = self.Predictor(self.t_obs,self.t_prev_obs,self.t_act, self.prev_size, self.obs_size, self.act_shape)

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)


	def ff_Policy(self, obs):
		return self.sess.run(self.probs, feed_dict={self.t_obs: [obs]})[0]

	def ff_Value(self, obs):
		return self.sess.run(self.value, feed_dict={self.t_obs: [obs]})[0]

	def ff_Predict(self, prev_obs, act):
		return self.sess.run(self.preds, feed_dict={self.t_prev_obs: [prev_obs[-self.prev_size:]], self.t_act: [act]})[0]

	def ff_Imaginary(self, prev_obs):
		if len(prev_obs)>self.prev_size:
			obs_vect = prev_obs[-self.prev_size:]
		else:
			obs_vect = [[0] * self.obs_size] * (self.prev_size + 2) 
		all_path = []
		all_val = []
		path = []

		sum_val = 0

		for i in range(self.predict_n):

			policy = self.act(obs_vect[-1])
			sum_val += self.ff_Value(obs_vect[-1])
			imag_obs = self.ff_Predict(obs_vect, policy[0])

			path.append(policy)
			obs_vect.append(imag_obs)

		all_path.append(path)
		all_val.append(sum_val)

		return all_path[np.argmax(all_val)][0]

	def act(self, obs):
		return self.probs_to_continuous_action(self.ff_Policy(obs))

	def New_MonteCarlo(self, rewards):
		
		delay = 1
		gamma = 0.97
		val = 0
		newvalues = [[0]] * delay
		
		for rew in rewards[::-1]:
			val = val * gamma + rew
			newvalues.append([val])
		return newvalues[delay:][::-1]

	def train(self, epi_obs, epi_prob, epi_act, epi_rew):
		newvals = self.New_MonteCarlo(epi_rew)

		self.train_Policy(epi_obs, epi_prob, newvals)
		self.train_Value(epi_obs, newvals)
		# self.train_Predict(epi_obs, epi_act)

	def train_Policy(self, epi_obs, epi_prob, newvals):
		# update policy function
		self.sess.run(self.opt_p, feed_dict={self.t_obs: epi_obs, self.t_prob: epi_prob, self.t_newvalue: newvals})

	def train_Value(self, epi_obs, newvals):
		# update value function
		self.sess.run(self.opt_v, feed_dict={self.t_obs: epi_obs, self.t_newvalue: newvals})

	def train_Mimic(self, epi_obs, epi_prob):
		# update policy function
		self.sess.run(self.opt_mp, feed_dict={self.t_obs : epi_obs, self.t_prob: epi_prob})

	def train_Predict(self, epi_obs, epi_act):
		batch_obs = [epi_obs[i:i + self.prev_size] for i,obs in enumerate(epi_obs) if i+1<len(epi_obs)][:-2]
		batch_act = epi_act[self.prev_size:]
		batch_label_obs = epi_obs[self.prev_size+1:]

		self.sess.run(self.opt_f,feed_dict={self.t_prev_obs: batch_obs, self.t_act: batch_act, self.t_obs : batch_label_obs})

	#Change Group of normalized values into continues
	#Quantification makes discrete decisions apply to continous
	def probs_to_continuous_action(self, probs):
		actions = []
		ret_space = []

		start = 0
		for shape in self.act_shape:
			prob = probs[start:start + shape]
			start += shape

			if len(prob)==2:
				discreteOut = [0,1]
			else:
				discreteOut = np.linspace(-1,1,len(prob))

			if len(prob)!=0:

				i = np.random.choice(range(shape),p=prob)
				actions.append(discreteOut[i])
				ret_space = ret_space + [int(i==j) for j in range(len(prob))]
		return actions,ret_space

	#Quantifies actions into onehot vectors
	def continuous_action_to_probs(self, actions):
		ret_probs = []

		for i,act in enumerate(actions):
			if self.act_shape[i] <= 2:
				#Bool case
				i_one = act
			else:
				i_one = (act + 1) * (self.act_shape[i]-1)/2

			onehot = [0] * self.act_shape[i]
			onehot[int(i_one)] = 1

			ret_probs = ret_probs + onehot

		return ret_probs