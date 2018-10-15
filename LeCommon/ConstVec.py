from LeCommon.Vector import *

class ConstVec():
	#Center
	Center	=			Vec3()

	#Center of a goal. positives Y are for the orange team.
	Pole	=			Vec3([      0	, 5120	,    321.3875])
	#Left pole when facing the goal
	Pole_L	=			Vec3([    893	, 5120	,    0])
	#Right pole when facing the goal
	Pole_R	=			Vec3([   -893	, 5120	,    0])
	#Top pole when facing the goal
	Pole_T	=			Vec3([   -0		, 5120	,    642.775])

	#Left Full Boost pad when facing the goal
	Boost_Corner_L =	Vec3([   3072	, 4096	,    73])
	#Left Full Boost pad when facing the goal
	Boost_Corner_R =	Vec3([  -3072	, 4096	,    73])

	#Left Full Boost pad when facing the goal
	Boost_Center_L =	Vec3([   3584   ,    0  ,    73])
	#Left Full Boost pad when facing the goal
	Boost_Center_R =	Vec3([  -3584   ,    0  ,    73])

	#Wall X
	Wall_L = 			Vec3([   4096   ,    0  ,    0])
	Wall_R = 			Vec3([  -4096   ,    0  ,    0])

	def get(name, team): #team => 1 is orange, 0 is blue
		if name=='Center':
			return ConstVec.Center

		if name=='Goal':
			return ConstVec.Pole * (team*2 - 1)
		if name=='Goal_L':
			return ConstVec.Pole_L * (team*2 - 1)
		if name=='Goal_R':
			return ConstVec.Pole_R * (team*2 - 1)
		if name=='Goal_T':
			return ConstVec.Pole_T * (team*2 - 1)

		if name=='Home':
			return ConstVec.Pole * (team*(-2) + 1)
		if name=='Home_L':
			return ConstVec.Pole_L * (team*(-2) + 1)
		if name=='Home_R':
			return ConstVec.Pole_R * (team*(-2) + 1)
		if name=='Home_T':
			return ConstVec.Pole_T * (team*(-2) + 1)

		if name=='Boost_Goal_L':
			return ConstVec.Boost_Corner_L * (team*2 - 1)
		if name=='Boost_Goal_R':
			return ConstVec.Boost_Corner_R * (team*2 - 1)

		if name=='Boost_Center_L':
			return ConstVec.Boost_Center_L * (team*2 - 1)
		if name=='Boost_Center_R':
			return ConstVec.Boost_Center_R * (team*2 - 1)

		if name=='Boost_Home_L':
			return ConstVec.Boost_Corner_L * (team*(-2) + 1)
		if name=='Boost_Home_R':
			return ConstVec.Boost_Corner_R * (team*(-2) + 1)

		if name=='Wall_L':
			return ConstVec.Wall_L * (team*2 - 1)
		if name=='Wall_R':
			return ConstVec.Wall_R * (team*2 - 1)


"""
Boost pads coords
[    0.0, -4240.0, 70.0] (0)
[-1792.0, -4184.0, 70.0] (1)
[ 1792.0, -4184.0, 70.0] (2)
[-3072.0, -4096.0, 73.0] (3)  Full
[ 3072.0, -4096.0, 73.0] (4)  Full
[- 940.0, -3308.0, 70.0] (5)
[  940.0, -3308.0, 70.0] (6)
[    0.0, -2816.0, 70.0] (7)
[-3584.0, -2484.0, 70.0] (8)
[ 3584.0, -2484.0, 70.0] (9)
[-1788.0, -2300.0, 70.0] (10)
[ 1788.0, -2300.0, 70.0] (11)
[-2048.0, -1036.0, 70.0] (12)
[    0.0, -1024.0, 70.0] (13)
[ 2048.0, -1036.0, 70.0] (14)
[-3584.0,     0.0, 73.0] (15) Full
[-1024.0,     0.0, 70.0] (16)
[ 1024.0,     0.0, 70.0] (17)
[ 3584.0,     0.0, 73.0] (18) Full
[-2048.0,  1036.0, 70.0] (19)
[    0.0,  1024.0, 70.0] (20)
[ 2048.0,  1036.0, 70.0] (21)
[-1788.0,  2300.0, 70.0] (22)
[ 1788.0,  2300.0, 70.0] (23)
[-3584.0,  2484.0, 70.0] (24)
[ 3584.0,  2484.0, 70.0] (25)
[    0.0,  2816.0, 70.0] (26)
[- 940.0,  3310.0, 70.0] (27)
[  940.0,  3308.0, 70.0] (28)
[-3072.0,  4096.0, 73.0] (29) Full
[ 3072.0,  4096.0, 73.0] (30) Full
[-1792.0,  4184.0, 70.0] (31)
[ 1792.0,  4184.0, 70.0] (32)
[    0.0,  4240.0, 70.0] (33)
"""
