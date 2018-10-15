from LeCommon.Vector import *
from LeCommon.Objs import *
from LeCommon.ConstVec import *

class Line():

	def __init__(self, vecA, vecB):
		self.vecA = vecA
		self.vecB = vecB

	def __str__(self):
		return "[ "+str(self.vecA)+" - "+str(self.vecB)+" ]"

"""
This class is based on a radial area principle.
Subdividing the areas into 6 zones, starting from the ball, should help define weither an agent can be threatening
or in disadvantage.
	  ________________
	 /        |       \
	 |-----___|___----|
	[      ___X___     ]
	 |-----   |   ----|
	 \________|______ /

this drawing is perfect
the map would be subdevided depending on the goal_Pole to ball segments, 
and depending on the ball to side walls segments

from theses segment the bot would define weither an agent is correctly placed (if an agent is in one zone)
or is he a possible threat (alined with the ball)

"""

class BallMetaArea():
	def __init__(self):
		self.ball = Vec3()

		self.Goal_sideL = Vec3()
		self.Goal_sideR = Vec3()

		self.Home_SideL = Vec3()
		self.Home_SideR = Vec3()

		self.Wall_L = Vec3()
		self.Wall_R = Vec3()


	def update(self, ball_loc, team):
		self.ball = ball_loc

		Pl = ConstVec.get('Goal_L',team)
		Pr = ConstVec.get('Goal_R',team)

		self.Goal_sideL = -(Pl - ball_loc)
		self.Goal_sideR = -(Pr - ball_loc)

		Pl = ConstVec.get('Home_L',team)
		Pr = ConstVec.get('Home_R',team)

		self.Home_sideL = -(Pl - ball_loc)
		self.Home_sideR = -(Pr - ball_loc)

		Wl = ConstVec.get('Wall_L', team)

		self.Wall_sideL = -(Pl - ball_loc)
		self.Wall_sideR = -(Pr - ball_loc)

	def inFrontZone(self, loc):
		return BallMetaArea.inZone(self.ball, loc, self.Goal_sideL, self.Goal_sideR)

	def inShotZone(self, loc):
		return BallMetaArea.inZone(self.ball, loc, self.Home_sideL, self.Home_sideR)

	# def inShotZone(self, loc):
	# 	return BallMetaArea.inZone(self.ball, loc, self.Goal_sideL, self.Goal_sideR)

	# def inShotZone(self, loc):
	# 	return BallMetaArea.inZone(self.ball, loc, self.Goal_sideL, self.Goal_sideR)

	@staticmethod
	def inZone(ref, loc, pL, pR):

		w1 = pL[0]/pL[1]
		w2 = pR[0]/pR[1]

		wl = loc[0]/loc[1]

		print(w1, wl ,w2)

		if(w1 > w2) : 
			return w2 <= wl and wl <= w1
		else:
			return w2 >= wl and wl >= w1 
		# return w1 >= 0 and w2 >= 0
