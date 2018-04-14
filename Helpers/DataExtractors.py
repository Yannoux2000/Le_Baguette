import numpy as np
from Helpers.LeMaths import *

class Obs_Extractor:
	def __init__(self, team, index, Obs_Size = 60, Gen_Obs = None):
		self.team = team
		self.index = index

		self.obs_size = Obs_Size
		self.home = Vector3(0, 5000 * (self.team * 2 - 1),0)
		self.goal = Vector3(0,-5000 * (self.team * 2 - 1),0)

		if Gen_Obs == None:
			self.Gen_Obs = Obs_Extractor.Polar_Obs
		else:
			self.Gen_Obs = Gen_Obs

	def __call__(self, GTP):
		return self._format_to_obs_size(self.Gen_Obs(self, GTP))

	def _format_to_obs_size(self,this_obs):
		"""
		Make sure the obs are the same size from start to finish
		"""

		for i,o in enumerate(this_obs):
			if np.isnan(o):
				print("NaNFoundInObs")
				this_obs[i] = 0.0


		obs = [0] * (self.obs_size)
		obs[:len(this_obs)] = this_obs
		return obs

	def Polar_Obs(self,GTP):
		"""
		Here is the same, but for Observations
		Polar outputs polar coords, only locations
		"""

		car = Car(Get_car(GTP,self.index))
		ball_loc = Vectorize_Loc(GTP.gameball)

		obs = car.p_array() + ball_loc.p_array()


		return obs

	def Pure_Obs(self,GTP):
		"""
		Pure outputs Cartesian coords, only locations
		"""

		car = Car(Get_car(GTP,self.index))
		ball_loc = Vectorize_Loc(GTP.gameball)

		obs = car.c_array() + ball_loc.c_array()


		return obs

	def Better_Obs(self,GTP):
		"""
		Better outputs Polar Locations and Velocities
		"""

		car = Car(Get_car(GTP,self.index))
		ball_loc = Vectorize_Loc(GTP.gameball)

		obs = car.array() + ball_loc.array()

		return obs

	def Obvious_Obs(self,GTP):
		"""
		Giving Directly the correct value, test algorithm learning ability
		"""

		car = Car(Get_car(GTP,self.index))
		ball_loc = Vectorize_Loc(GTP.gameball)

		obs = car.p_array_to(ball_loc)

		return obs

	def Final_Obs(self,GTP):
		"""
		Giving most important informations to the model
		"""

		car = Car(Get_car(GTP,self.index))

		obs = car.array()

		targets = []
		targets.append(Vectorize_Loc(GTP.gameball))

		for t in targets:
			obs += car.p_array_to(t)
		return obs




class Rewards_Signal:
	def __init__(self, team, index, Gen_Rews = None, Deriv = False, Clip = False, C_val = 2.0):

		self.team = team
		self.index = index

		self.deriv = Deriv
		self.prev_rew = 0

		self.clip = Clip
		self.c_val = C_val

		if Gen_Rews == None:
			self.Gen_Rews = EnvInterface.Normalv2_Rews
		else:
			self.Gen_Rews = Gen_Rews

	def __call__(self, GTP):
		rew = self.Gen_Rews(self,GTP)

		rew = self._derivate_rew(rew) if self.deriv else rew

		rew = self._clip_rew(rew) if self.clip else rew

		return np.clip(rew,-50,50) #no wild values allowed here

	def _clip_rew(self,rew):
		return np.clip(rew,-self.c_val, self.c_val)

	#only keep stepwise variations
	def _derivate_rew(self,rew):
		step_rew = rew - self.prev_rew
		self.prev_rew = rew
		return step_rew

	def Supervised_Rews(self,GTP):
		"""
		This reward is for supervised only, outputs 1 to take into account each actions from innerAgents
		"""
		return 1

	def Scores_Rews(self, GTP):
		"""
		Simply extracting scores from each car and adding it to the reward
		if this car is in the same team than LeBaguette's team
		"""
		rew = 0.0

		for car in GTP.gamecars:
			rew += car.Score.Score * ((self.team * 2 - 1) * (car.Team * 2 -1))

		return rew

	def Normalv2_Rews(self, GTP):
		"""
		Simply extracting scores from each car and adding it to the reward
		if this car is in the same team than LeBaguette's team
		"""
		rew = self.Scores_Rews(GTP)
		rew += self.distance_to_ball_Rews(GTP)
		rew += self.SuperSonic_Rews(GTP)
		return rew

	def SuperSonic_Rews(self, GTP):
		car = Get_car(GTP,self.index)
		rew = 10 if car.bSuperSonic else -0.01

		return rew

		#1st trial on ball touch problem, i tried giving it the ball distance + ball angle together
	def Continuous_touch_Rews(self, GTP):
		rew = 0.0

		car = Car(Get_car(GTP,self.index))
		ball_loc = Vectorize_Loc(GTP.gameball)

		ball_angle = abs(car.steer_to(ball_loc))
		ball_distance = car.distance(ball_loc)

		if car.reached(ball_loc):
			bonus_touch = 50
		else:
			bonus_touch = -5

		rew = 10/ball_angle + 20/ball_distance + bonus_touch

		return rew

		#2nd trial on ball touch problem, this time only giving distance
	def distance_to_Rews(self, GTP, vector):
		return 3000 - Car(Get_car(GTP,self.index)).distance(vector)

	def distance_to_ball_Rews(self, GTP):
		current_distance = Vectorize_Loc(Get_car(GTP,self.index)).distance(Vectorize_Loc(GTP.gameball)) / 500
		divided_value = (50 / (current_distance + 1e-10))
		return divided_value

		#3rd trial on ball touch problem, only giving angle
	def angle2d_to_Rews(self, GTP, vector):
		my_car = Get_car(GTP,self.index)
		car_loc, car_dir, car_vel, car_anv = Car_To_Vec(my_car)
		car_to_vec = vector - car_loc

		angle = Rad_clip(car_to_vec.yaw - car_dir.yaw)
		return 0.5 - abs(angle)

		#Last Trial on distance, i probably shoud think of something else
	def Reach_vec_Rews(self,GTP,vector):
		car = Car(Get_car(GTP,self.index))

		if car.reached(vector):
			return 10
		else:
			value = (10 / car.distance(vector))
			if value < 0.5:
				return -1
			else:
				return value

	def Reach_Ball_Rews(self, GTP):
		ball_loc = Vectorize_Loc(GTP.gameball)
		return self.Reach_vec_Rews(GTP,ball_loc)

		# Reward on speed
	def Speed_Rews(self, GTP):
		car = Car(Get_car(GTP,self.index))
		return (car.vel.magnitude / 50)