import numpy as np

URotationToRadians = np.pi / float(32768)

def URot_to_Degree(val):
	return val % 65536 / 65536 * 360

def URot_to_Rad(val):
	return val * URotationToRadians

def Rad_to_Degree(val):
	return val / np.pi * 180

def Rad_clip(val):
		# Make sure we go the 'short way'
		if abs(val) > np.pi:
			val += (2 * np.pi) if val < 0 else (- 2 * np.pi)
		return val

class Vec3:
	def __init__(self, data = [0,0,0]):
		#cartesian coords
		self.vec = data


	@staticmethod
	def process_Vec(vector):
		return Vec3([vector.x,vector.y,vector.z])

	@staticmethod
	def process_Rot(rot):
		return Vec3([rot.pitch,rot.yaw,rot.roll])

	def __getitem__(self,i):
		return self.vec[i]

	def __neg__(self):
		return Vec3([-self[0], -self[1], -self[2]])

	def __add__(self, other):
		if (isinstance(other, Vec3)):
			return Vec3([self[0] + other[0], self[1] + other[1], self[2] + other[2]])
		else :
			return Vec3([self[0] + other, self[1] + other, self[2] + other])

	def __sub__(self, other):

		if (isinstance(other, Vec3)):
			return Vec3([self[0] - other[0], self[1] - other[1], self[2] - other[2]])
		else:
			return Vec3([self[0] - other, self[1] - other, self[2] - other])

	def __mul__(self, other):

		if (isinstance(other, Vec3)):
			return Vec3([self[0] * other[0], self[1] * other[1], self[2] * other[2]])
		else:
			return Vec3([self[0] * other, self[1] * other, self[2] * other])

	def __div__(self, other):

		if (isinstance(other, Vec3)):
			return Vec3([self[0] / other[0], self[1] / other[1], self[2] / other[2]])
		else:
			return Vec3([self[0] / other, self[1] / other, self[2] / other])

	def __eq__(self, other):
		return (self[0] == other[0] and self[1] == other[1] and self[2] == other[2])

	def __str__(self):
		return "(x: {0:6.2f},y: {1:6.2f},z: {2:6.2f})".format(*self.vec)

	def __format__(self, args):
		return str(self)

	def dot(self, other):
		# return sum([a * b for a,b in zip(self.vec,other.vec)])
		return (self[0] * other[0] + self[1] * other[1] + self[2] * other[2])

	def cross(self, other):
		return 	Vec3([self[1]*other[2]-self[2]*other[1], self[2]*other[0]-self[0]*other[2], self[0]*other[1]-self[1]*other[0]])

	def Gnd(self):
		return Vec3([self[0]] + [self[1]] + [0])

	def magnitude(self):
		return np.sqrt(self.dot(self))

	def distance(self, other):
		return (self - other).magnitude()

	def np(self,w=1):
		return np.array([self.vec+[w]])

	# #length calculation
	# def __abs__(self):
	# 	return np.sqrt(self.x**2 + self.y**2 + self.z**2)

	def normalize(self):
		return self / self.magnitude()

	def c_2d(self):
		return [self[0]/50,self[1]/50]

	def c_array(self):
		return self.vec

	# def p_array(self):
	# 	return [self.yaw,self.pitch,self.magnitude]

	def array(self):
		return self.c_array() + self.p_array()

	def to_polar(self):
		# The in-game axes are left handed, so use -x
			# ret pitch , yaw
		return Rad_clip(np.arcsin(self[2] / self.magnitude)), Rad_clip(np.arctan2(self[1], -self[0]))

	def correction2d_to(self, ideal):
		return Rad_clip(ideal.yaw - self.yaw)
