#This class serves as a cheaper debugger than RL can be, instead of waiting RL and stuff on my laptop you no...

from LeBaguette import Agent
import math
import tensorflow as tf
import numpy as np
import gym
import time


class Vector3():
	def __init__(self):
		self.X = 0.0#", ctypes.c_float),
		self.Y = 0.0#", ctypes.c_float),
		self.Z = 0.0#", ctypes.c_float)]

		#Debuging
	def __str__(self):
		return "({:6.2f}, {:6.2f}, {:6.2f})".format(self.X,self.Y,self.Z)


class Rotator():
	def __init__(self):
		self.Pitch = 0#", ctypes.c_int),
		self.Yaw = 0#", ctypes.c_int),
		self.Roll = 0#", ctypes.c_int)]


class Touch():
	def __init__(self):
		self.wPlayerName = "ABotsName"#", ctypes.c_wchar * MAX_NAME_LENGTH),
		self.fTimeSeconds = 0.0#", ctypes.c_float),
		self.sHitLocation = Vector3()#", Vector3),
		self.sHitNormal = Vector3()#", Vector3)]

class ScoreInfo():
	def __init__(self):
		self.Score = 0#", ctypes.c_int),
		self.Goals = 0#", ctypes.c_int),
		self.OwnGoals = 0#", ctypes.c_int),
		self.Assists = 0#", ctypes.c_int),
		self.Saves = 0#", ctypes.c_int),
		self.Shots = 0#", ctypes.c_int),
		self.Demolitions = 0#", ctypes.c_int)]


class PlayerInfo():
	def __init__(self):
		self.Location = Vector3()#", Vector3),
		self.Rotation = Rotator()#", Rotator),
		self.Velocity = Vector3()#", Vector3),
		self.AngularVelocity = Vector3()#", Vector3),
		self.Score = ScoreInfo()#", ScoreInfo),
		self.bDemolished = False#", ctypes.c_bool),
				# True if your wheels are on the ground, the wall, or the ceiling. False if you're midair or turtling.
		self.bOnGround = False#", ctypes.c_bool),
		self.bSuperSonic = False#", ctypes.c_bool),
		self.bBot = False#", ctypes.c_bool),
				# True if the player has jumped. Falling off the ceiling / driving off the goal post does not count.
		self.bJumped = False#", ctypes.c_bool),
				# True if player has double jumped. False does not mean you have a jump remaining, because the
				# aerial timer can run out, and that doesn't affect this flag.
		self.bDoubleJumped = False#", ctypes.c_bool),
		self.wName = "ABotsName"#", ctypes.c_wchar * MAX_NAME_LENGTH),
		self.Team = 0#", ctypes.c_ubyte),
		self.Boost = 0#", ctypes.c_int)]
	def __str__(self):
		return "POS : " + str(self.Location) + "  Score : " + str(self.Score.Score)


class BallInfo():
	def __init__(self):
		self.Location = Vector3()#", Vector3),
		self.Rotation = Rotator()#", Rotator),
		self.Velocity = Vector3()#", Vector3),
		self.AngularVelocity = Vector3()#", Vector3),
		self.Acceleration = Vector3()#", Vector3),
		self.LatestTouch = Touch()#", Touch)]

	def __str__(self):
		return "POS : " + str(self.Location)


class BoostInfo():
	def __init__(self):
		self.Location = Vector3()#", Vector3),
		self.bActive = False#", ctypes.c_bool),
		self.Timer = 0#", ctypes.c_int)]


class GameInfo():
	def __init__(self):
		self.TimeSeconds = 0.0#", ctypes.c_float),
		self.GameTimeRemaining = 0.0#", ctypes.c_float),
		self.bOverTime = False#", ctypes.c_bool),
		self.bUnlimitedTime = False#", ctypes.c_bool),
				# True when cars are allowed to move, and during the pause menu. False during replays.
		self.bRoundActive = False#", ctypes.c_bool),
				# Only false during a kickoff, when the car is allowed to move, and the ball has not been hit,
				# and the game clock has not started yet. If both players sit still, game clock will eventually
				# start and this will become true.
		self.bBallHasBeenHit = False#", ctypes.c_bool),
				# Turns true after final replay, the moment the 'winner' screen appears. Remains true during next match
				# countdown. Turns false again the moment the 'choose team' screen appears.
		self.bMatchEnded = False#", ctypes.c_bool)]


# On the c++ side this struct has a long at the beginning for locking.  This flag is removed from this struct so it isn't visible to users.
class GameTickPacket():
	def __init__(self):
		self.gamecars = [PlayerInfo() for i in range(10)]#", PlayerInfo * MAX_PLAYERS),
		self.numCars = 1#", ctypes.c_int),
		self.gameBoosts = [BoostInfo() for i in range(10)]#", BoostInfo * MAX_BOOSTS),
		self.numBoosts = 1#", ctypes.c_int),
		self.gameball = BallInfo()#", BallInfo),
		self.gameInfo = GameInfo()#", GameInfo)]

	def __str__(self):
		mess = "GTP : \n"
		mess += "\t car_count : {}".format(self.numCars)
		for car in self.gamecars:
			mess += "\n\t\t" + str(car)
		mess += "\n\t Ball :\n\t\t" + str(self.gameball)

		return mess

# Fully matching c++ struct
class GameTickPacketWithLock():
	def __init__(self):
		self.lock = 0#", ctypes.c_long),
		self.iLastError = 0#", ctypes.c_int),
		self.gamecars = [PlayerInfo() for i in range(10)]#", PlayerInfo * MAX_PLAYERS),
		self.numCars = 10#", ctypes.c_int),
		self.gameBoosts = [BoostInfo() for i in range(10)]#", BoostInfo * MAX_BOOSTS),
		self.numBoosts = 10#", ctypes.c_int),
		self.gameball = BallInfo()#", BallInfo),
		self.gameInfo = GameInfo()#", GameInfo)]

def generate_random_packet():
	game_tick_packet = GameTickPacket()

	for car in game_tick_packet.gamecars:
		car.Location.X = np.random.uniform(-200, 200)
		car.Location.Y = np.random.uniform(-200, 200)
		car.Location.Z = np.random.uniform(-200, 200)

		car.Velocity.X = np.random.uniform(-50, 50)
		car.Velocity.Y = np.random.uniform(-50, 50)
		car.Velocity.Z = np.random.uniform(-50, 50)

		car.Rotation.X = int(np.random.uniform(-32768, 32768))
		car.Rotation.Y = int(np.random.uniform(-32768, 32768))
		car.Rotation.Z = int(np.random.uniform(-32768, 32768))

	game_tick_packet.numCars = len(game_tick_packet.gamecars)

	ball = game_tick_packet.gameball

	ball.Location.X = np.random.uniform(-200, 200)
	ball.Location.Y = np.random.uniform(-200, 200)
	ball.Location.Z = np.random.uniform(-200, 200)

	ball.Velocity.X = np.random.uniform(-50, 50)
	ball.Velocity.Y = np.random.uniform(-50, 50)
	ball.Velocity.Z = np.random.uniform(-50, 50)

	ball.Rotation.X = int(np.random.uniform(-32768, 32768))
	ball.Rotation.Y = int(np.random.uniform(-32768, 32768))
	ball.Rotation.Z = int(np.random.uniform(-32768, 32768))

	ball.Acceleration.X = np.random.uniform(-75, 75)
	ball.Acceleration.Y = np.random.uniform(-75, 75)
	ball.Acceleration.Z = np.random.uniform(-75, 75)

	for boost in game_tick_packet.gameBoosts:

		boost.Location.X = np.random.uniform(-200, 200)
		boost.Location.Y = np.random.uniform(-200, 200)
		boost.Location.Z = np.random.uniform(-200, 200)
		boost.bActive = np.random.choice([True,False])

	return game_tick_packet


#Static faker for syntax error
class EnvFaker():

	def __init__(self):

		self.env = gym.make("Pendulum-v0")
		self.totrew = 0
		self.all_totrew = []

	def FakeObs(self,obs,rews):
		GTP = GameTickPacket()

		GTP.gamecars[0].Location.X = obs[0]
		GTP.gamecars[0].Location.Y = obs[1]

		GTP.gamecars[0].Score.Score = rews

		GTP.gameball.Location.X = obs[2]
		# GTP.gameball.Location.Y = obs[3]

		return GTP

	def Reset_epi(self):
		obs = self.env.reset()
		return self.FakeObs(obs,0)

	def run_step(self,inact):

		act = [inact[0]]
		obs, rew, done , _ = self.env.step(act)
		self.totrew += rew

		if done:
			rew = -20
			self.all_totrew.append(self.totrew)
			self.all_totrew = self.all_totrew[-100:]
			print("{}, µ = {}".format(self.totrew,np.mean(self.all_totrew)))
			self.totrew = 0
			obs = self.env.reset()

		return self.FakeObs(obs,rew)

env = EnvFaker()

GTP = env.Reset_epi()

agent = Agent("MYBOT", 0, 0)
agent.debug = False

for i in range(2000000):
	acts = agent.get_output_vector(GTP)
	GTP = env.run_step(acts)
	# if (x%100)==0:
	# 	print("[{:6d}]: {}".format(x,acts))
