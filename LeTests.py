from LeCommon.Vector import *
from LeCommon.ConstVec import *
from LeCommon.Objs import *
from LeCommon.Areas import *

import sys
import os
sys.path.insert(0, os.path.realpath(os.path.dirname(os.path.abspath(__file__)) + '/../../src/main/python/'))

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

def assert_message(expected, actual, assert_name=""):
	print("{:^32} :{:^48}\t==\t{:^48}".format(assert_name, str(expected), str(actual)))

def asserter(expected, actual, assert_name):
	assert_message(expected, actual, assert_name)
	assert expected == actual
	

class mock_physics:

	def __init__(self, loc={'x' : 0, 'y' : 0, 'z' : 0},vel={'x' : 0, 'y' : 0, 'z' : 0},rot={'pitch' : 0, 'yaw' : 0, 'roll' : 0},ang={'x' : 0, 'y' : 0, 'z' : 0}):
		self.location = type('obj', (object,), loc)
		self.velocity = type('obj', (object,), vel)
		self.rotation = type('obj', (object,), rot)
		self.angular_velocity = type('obj', (object,), ang)

# TESTS Vec3

asserter("(x:   0.00,y:   0.00,z:   0.00)"	,str(Vec3()),						"Vec3 Str")
asserter("(x:   1.25,y:   7.26,z:  -4.16)"	,str(Vec3([1.25,7.26,-4.156])),		"Vec3 Str")

v = Vec3([5,-5,200.5])
asserter(v 						,v,												"Vec3 Equal")
asserter(Vec3([5,0,-0.5])		,Vec3([5,0,-0.5]),								"Vec3 Equal")

asserter(Vec3([0,0,0])			,Vec3()					, 						"Vec3 init")
asserter(Vec3([1,2,3])			,Vec3([1,2,3])			, 						"Vec3 init")

asserter(Vec3([5,-5,200])		,Vec3([0,0,-0.5])		 + Vec3([5,-5,200.5]), 	"Vec3 add")
asserter(Vec3([5,-5,200])		,Vec3([5,-5,200.5])		 + Vec3([0,0,-0.5]), 	"Vec3 add")
asserter(Vec3([51,-25,80.5])	,Vec3([1,-75,30.5])		 + 50, 					"Vec3 add")

asserter(Vec3([0,5,-11])		,Vec3([5,0,-0.5])		 - Vec3([5,-5,10.5]), 	"Vec3 sub")
asserter(Vec3([0,-5,11])		,Vec3([5,-5,10.5])		 - Vec3([5,0,-0.5]), 	"Vec3 sub")
asserter(Vec3([-45,-55,-39.5])	,Vec3([5,-5,10.5])		 - 50, 					"Vec3 sub")

asserter(Vec3([0,0,-100.25])	,Vec3([0,0,-0.5])		 * Vec3([5,-5,200.5]), 	"Vec3 mul")
asserter(Vec3([0,0,-100.25])	,Vec3([5,-5,200.5])		 * Vec3([0,0,-0.5]), 	"Vec3 mul")
asserter(Vec3([-25,25,0.5*-5])	,Vec3([5,-5,0.5])		 * -5, 					"Vec3 mul")

v = Vec3.process_Vec(type('obj', (object,), {'x': 1,'y':-1.25,'z':123}))
asserter(Vec3([1,-1.25,123]),v,"Vec3 distance")

v = Vec3.process_Rot(type('obj', (object,), {'pitch' : 1, 'yaw' : -1.25, 'roll' : 123}))
asserter(Vec3([1,-1.25,123]),v,"Vec3 distance")

asserter(20	,Vec3([0,0,20]).distance(Vec3([0,20,20])),							"Vec3 distance")
asserter(0	,Vec3([15,24,32]).distance(Vec3([15,24,32])),						"Vec3 distance")
asserter(64	,Vec3([15,24,32]).distance(Vec3([15,24,-32])),						"Vec3 distance")

asserter(Vec3([5,-5,0])			,Vec3([5,-5,0.5]).Gnd(),						"Vec3 GND")

asserter(-100.25				,Vec3([5,-5,200.5]).dot(Vec3([0,0,-0.5])),		"Vec3 dot")
asserter(25 + 25 + (20**2)		,Vec3([5,-5,20]).dot(Vec3([5,-5,20])),			"Vec3 dot")
asserter(-75					,Vec3([5,5,5]).dot(Vec3([-5,-5,-5])),			"Vec3 dot")
asserter(3						,Vec3([2,3,0]).dot(Vec3([3,-1,0])),				"Vec3 dot")
asserter(-3						,Vec3([2,3,0]).dot(Vec3([-3,1,0])),				"Vec3 dot")

print("ConstVec")

print(ConstVec.get('Center', 0))
print(ConstVec.get('Goal', 0))
print(ConstVec.get('Goal', 1))
print(ConstVec.get('Goal_L', 0))
print(ConstVec.get('Goal_L', 1))
print(ConstVec.get('Goal_R', 0))
print(ConstVec.get('Goal_R', 1))
print(ConstVec.get('Goal_T', 0))
print(ConstVec.get('Goal_T', 1))
print(ConstVec.get('Home', 0))
print(ConstVec.get('Home', 1))
print(ConstVec.get('Home_L', 0))
print(ConstVec.get('Home_L', 1))
print(ConstVec.get('Home_R', 0))
print(ConstVec.get('Home_R', 1))
print(ConstVec.get('Home_T', 0))
print(ConstVec.get('Home_T', 1))
print(ConstVec.get('Boost_Goal_L', 0))
print(ConstVec.get('Boost_Goal_L', 1))
print(ConstVec.get('Boost_Goal_R', 0))
print(ConstVec.get('Boost_Goal_R', 1))
print(ConstVec.get('Boost_Center_L', 0))
print(ConstVec.get('Boost_Center_L', 1))
print(ConstVec.get('Boost_Center_R', 0))
print(ConstVec.get('Boost_Center_R', 1))

asserter(Vec3(),Obj().loc,"Obj init")

bma = BallMetaArea()

print(ConstVec.get('Goal_L', 1))
print(ConstVec.get('Goal_R', 1))

ball = Vec3([0,0,0])

a = - (ConstVec.get('Goal_L', 1) - ball)
b = - (ConstVec.get('Goal_R', 1) - ball)
print()
print(a)
print(b)

bma.update(ball, 1)

asserter(False	,bma.inShotZone(Vec3([-893,-5119,-321.39]))	, "Area")
asserter(False	,bma.inShotZone(Vec3([-893,-5119, 321.39]))	, "Area")
asserter(True	,bma.inShotZone(Vec3([-893,-5121,-321.39]))	, "Area")
asserter(True	,bma.inShotZone(Vec3([-893,-5121, 321.39]))	, "Area")
asserter(True	,bma.inShotZone(Vec3([-892,-5120,-321.39]))	, "Area")
asserter(True	,bma.inShotZone(Vec3([-892,-5120, 321.39]))	, "Area")
asserter(False	,bma.inShotZone(Vec3([-894,-5120,-321.39]))	, "Area")
asserter(False	,bma.inShotZone(Vec3([-894,-5120, 321.39]))	, "Area")

asserter(False	,bma.inShotZone(Vec3([893,-5119,-321.39]))	, "Area")
asserter(False	,bma.inShotZone(Vec3([893,-5119, 321.39]))	, "Area")
asserter(True	,bma.inShotZone(Vec3([893,-5121,-321.39]))	, "Area")
asserter(True	,bma.inShotZone(Vec3([893,-5121, 321.39]))	, "Area")
asserter(True	,bma.inShotZone(Vec3([892,-5120,-321.39]))	, "Area")
asserter(True	,bma.inShotZone(Vec3([892,-5120, 321.39]))	, "Area")
asserter(False	,bma.inShotZone(Vec3([894,-5120,-321.39]))	, "Area")
asserter(False	,bma.inShotZone(Vec3([894,-5120, 321.39]))	, "Area")


ball = Vec3([-500,-3000,0])

a = ConstVec.get('Home_L', 1)
b = ConstVec.get('Home_R', 1)

bma.update(ball, 1)

asserter(False	,bma.inShotZone(Vec3([a[0],a[1]+1,a[2]]))	, "Area")
asserter(True	,bma.inShotZone(Vec3([a[0],a[1]-1,a[2]]))	, "Area")
asserter(True	,bma.inShotZone(Vec3([a[0]+1,a[1],a[2]]))	, "Area")
asserter(False	,bma.inShotZone(Vec3([a[0]-1,a[1],a[2]]))	, "Area")

asserter(False	,bma.inShotZone(Vec3([b[0],b[1]+1,b[2]]))	, "Area")
asserter(True	,bma.inShotZone(Vec3([b[0],b[1]-1,b[2]]))	, "Area")
asserter(True	,bma.inShotZone(Vec3([b[0]-1,b[1],b[2]]))	, "Area")
asserter(False	,bma.inShotZone(Vec3([b[0]+1,b[1],b[2]]))	, "Area")

from LeCroissant2 import *
