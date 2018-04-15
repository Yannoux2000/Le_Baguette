print("numpy")
import numpy as np
print("DataExtractors")
from Helpers.DataExtractors import *
from Helpers.LeMaths import *
print("Model")
# from Models.PPOModel import Model
from Models.FUNModel import Model
print("InnerAgent")
from Agents.atba import Agent as Atbagent
print("Tracer")
from quicktracer import trace

#LEBAGUETTE IS ML
# BLANK_ACTION = [0,0,0,0,0 ,0,0,0]
ACTION_SHAPE = [5,5,5,5,5 ,2,2,2]
ACTION_SIZE = 8
OBS_SIZE = 60

EXP_VAL = 20

class Agent:
	def __init__(self, name, team, index, bot_parameters=None):
		self.name = name
		self.team = team  # 0 towards positive goal, 1 towards negative goal.
		self.index = index

		#Observator giving Obs vectors for our model
		self.Observer = Obs_Extractor(self.team, self.index, Obs_Size=OBS_SIZE,
				Gen_Obs = Obs_Extractor.Final_Obs)

		#Rewarder giving Rewards vectors for our model
		self.Rewarder = Rewards_Signal(self.team, self.index,
				Gen_Rews = Rewards_Signal.distance_to_ball_Rews)#, Deriv=False, Clip=True, C_val= 20.0)

		self.debug = True if index==0 else False

		self.debug = False

		self.model = Model(OBS_SIZE, ACTION_SIZE)
		## internal Data used for stepwise applications
		self.state = {'Step' : -100, 'MaxStep' : 250, 'TrainStep' : 0, 'SaveAt' : 20}

		##For supervised learning
		self.CopyAgent = Atbagent(name, team, index)

		self.max_trans_n = 1000
		self.empty_transition()

		print('Init_saver:')
		self.model.init_saver()

		##Transitions stores rewards actions states and probs in same lengthed arrays
	def empty_transition(self):

		self.epi_rewards = []
		self.epi_actions = []
		self.epi_states = []
		self.epi_probs = []

	def save_transition(self, obs, prob, act, rew):

		self.epi_rewards.append(rew)
		self.epi_actions.append(act)
		self.epi_states.append(obs)
		self.epi_probs.append(prob)

		self.epi_rewards = self.epi_rewards[-self.max_trans_n:]
		self.epi_actions = self.epi_actions[-self.max_trans_n:]
		self.epi_states = self.epi_states[-self.max_trans_n:]
		self.epi_probs = self.epi_probs[-self.max_trans_n:]

	def empty_old_actions(self):

		self.epi_old_actions = []

	def save_old_actions(self,old_act):

		self.epi_old_actions.append(old_act)

		self.epi_old_actions = self.epi_old_actions[-self.max_trans_n:]

	def Format_Output(self,acts):
		ACTS_OUT = []

			#clip all inputs between -1 and 1
		for act in acts:
			ACTS_OUT.append(np.clip(act,-1,1))

			#Last 3 outputs are thresholded to convert into bools
		for i,act in enumerate(acts[-3:]):
			ACTS_OUT[5+i] = (act > 0.8)

		return ACTS_OUT

	def get_output_vector(self, game_tick_packet):

			#Comment to switch training mode
		output = self.reinforced_play(game_tick_packet)
		# output = self.supervised_play(game_tick_packet)
		# output = self.obvious_play(game_tick_packet)

			##Debuging every 50 steps
		if self.debug and (self.state['Step']%10)==0:
			trace(np.mean(self.epi_rewards))
			trace(Car(Get_car(game_tick_packet,self.index)).loc.c_2d())

			##Count loops, at MaxStep will reset, also train every maxsteps
		if self.state['Step'] >= self.state['MaxStep']:

				#Comment to switch training mode
			self.reinforced_train()
			# self.supervised_train()
			# self.obvious_train()


			self.state['TrainStep'] += 1
			self.state['Step'] = 0

			if self.state['TrainStep'] >= self.state['SaveAt']:
				self.model.save("V5_{}".format(self.name))
				print("Saving")
				self.state['TrainStep'] = 0

		else:
			self.state['Step'] += 1

		return self.Format_Output(output)

	def obvious_play(self,GTP):
		obs = self.Observer(GTP)
		rews = self.Rewarder(GTP)

		outputs = [0] * 8
		outputs[0] = 1.0
		outputs[1] = obs[0]

		self.epi_rewards.append(rews)
		self.epi_rewards = self.epi_rewards[-self.max_trans_n:]

		return outputs

	def obvious_train(self):
		pass

	def reinforced_play(self,GTP):

		rews = self.Rewarder(GTP)
		obs = self.Observer(GTP)
		# rews += self.FUCKUReward(obs)

		similarity = 0

		for o in self.epi_states:
			similarity += np.mean([abs(oa - ob) for oa,ob in zip(o,obs)])

		similarity = similarity/ (5000 * EXP_VAL)


		#try to predict what the future is like
		# acts, probs = self.model.ff_Imaginary(self.epi_states)

		# #if there is not enouth previous actions then you can't imagine
		acts, probs = self.model.act(obs)

		rews = float(acts[0]) * 5
		rews += similarity
		# #Favor forward movement
		# if acts[0]<0.0:
		# 	rews -= 0.01

		# rews /= 20

		for a in acts:
			if np.isnan(a):
				print(acts)
				print(obs)
				print(rews)

		outputs = [0] * 8
		outputs = [acts[0],acts[1],acts[2],acts[3],acts[4],0,0,0]

		# outputs = acts

		self.save_transition(obs,outputs,acts,rews)

			##Debuging every 50 steps
		if self.debug and (self.state['Step']%50)==0:
			dbg_mess = ""
			dbg_mess += "Sim : {:8.2f}".format(similarity)

			print(dbg_mess)
		# 	trace(np.mean(self.epi_rewards))
		return outputs

	def reinforced_train(self):
		self.model.train(self.epi_states,self.epi_probs,self.epi_actions,self.epi_rewards)
		# self.empty_transition()

	def supervised_play(self, GTP):

		rews = 1 # Because copy agent is always right
		obs = self.Observer(GTP)

		acts = self.CopyAgent.get_output_vector(GTP)
		probs = self.model.continuous_action_to_probs(acts)

		output, _ = self.model.act(obs)
		#Here probs is used as labels
		self.save_transition(obs,probs,output,rews)

		if self.debug and (self.state['Step']%50)==0:
			dbg_mess = ""
			dbg_mess += "[{:6}]".format(self.state['Step'])
			mean = np.mean(np.absolute(np.subtract(self.epi_probs,self.epi_actions)))
			dbg_mess += "Differences {}".format(mean)
			print(dbg_mess)

		return output

	def supervised_train(self):
		self.model.train_s(self.epi_states,self.epi_probs,self.epi_actions,self.epi_rewards)
		# self.empty_transition()