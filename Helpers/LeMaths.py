import numpy as np

URotationToRadians = np.pi / float(32768)

GRAVITY = 13
GROUND_Z_AXIS = 1.8555
COR = 0.8
AIR_RESISTANCE = 0.3

def URot_to_Degree(val):
	return val % 65536 / 65536 * 360
def URot_to_Rad(val):
	return val * URotationToRadians
def Rad_to_Degree(val):
	return val / np.pi * 180

class Vector3:
	def __init__(self, x = 0, y = 0, z = 0):
		#cartesian coords
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)

		#polar coords
		self.magnitude = self.dot(self)
		self.pitch, self.yaw = self.to_polar()


	def __add__(self, other):
		return Vector3( self.x + other.x, self.y + other.y, self.z + other.z)

	def __sub__(self, other):
		return Vector3( self.x - other.x, self.y - other.y, self.z - other.z)

	def __str__(self):
		return "(x: {:6.2f},y: {:6.2f},z: {:6.2f})".format(self.x,self.y,self.z)

	#length calculation
	def __abs__(self):
		return np.sqrt(self.x**2 + self.y**2 + self.z**2)

	def Zp(self,z):
		return Vector3(self.x,self.y,self.z + z)

	def dot(self, other):
		return np.sqrt(self.x * other.x + self.y * other.y + self.z * other.z)

	def cross(self, other):
		return Vector3(self.y*other.z - self.z*other.y, self.z*other.x - self.x*other.z, self.x*other.y-self.y*other.x)

	def normalize(self):
		return Vector3(self.x/self.magnitude, self.y/self.magnitude, self.z/self.magnitude)

	def Gnd(self):
		return Vector3(self.x,self.y)

	#Norm of a difference = distance
	def distance(self, other):
		return (self - other).magnitude

	def np(self,w=1):
		return np.array([[self.x],[self.y],[self.z],[w]])

	def c_2d(self):
		return [self.x/50,self.y/50]

	def c_array(self):
		return [self.x,self.y,self.z]

	def p_array(self):
		return [self.yaw,self.pitch,self.magnitude]

	def array(self):
		return self.c_array() + self.p_array()

	def to_polar(self):
		# The in-game axes are left handed, so use -x
			# ret pitch , yaw
		return Rad_clip(np.arcsin(self.z / self.magnitude)), Rad_clip(np.arctan2(self.y, -self.x))

	def correction2d_to(self, ideal):
		return Rad_clip(ideal.yaw - self.yaw)

def Vectorize_Np(array):
	return Vector3(array[0],array[1],array[2])

def Vectorize_Loc(entity):
	return Vector3(entity.Location.X,entity.Location.Y,entity.Location.Z)

def Vectorize_Vel(entity):
	return Vector3(entity.Velocity.X,entity.Velocity.Y,entity.Velocity.Z)

def Vectorize_Avl(entity):
	return Vector3(entity.AngularVelocity.X,entity.AngularVelocity.Y,entity.AngularVelocity.Z)

	#Extract Car entity from the GTP, as an easier object to manipulate
def Get_car(GTP, index):
	return GTP.gamecars[index]

class Car():
	def __init__(self, car_entity):
		
		self.ent = car_entity

		self.pitch = float(car_entity.Rotation.Pitch) * URotationToRadians
		self.yaw = float(car_entity.Rotation.Yaw) * URotationToRadians
		self.roll = float(car_entity.Rotation.Roll) * URotationToRadians

		self.loc = Vectorize_Loc(car_entity)
		self.vel = Vectorize_Vel(car_entity)
		self.avl = Vectorize_Avl(car_entity)

		self.dir = self.Forward()

	def to(self, t_loc):
		return t_loc - self.loc

	def steer_to(self, t_loc):
		return - Rad_clip(self.to(t_loc).yaw - self.dir.yaw)

	def distance(self, t_loc):
		return self.loc.distance(t_loc)

	def reached(self, t_loc, threshold = 500):
		return (self.distance(t_loc) < threshold)

	def Forward(self):

		facing_x = np.cos(self.pitch) *  np.cos(self.yaw)
		facing_y = np.cos(self.pitch) *  np.sin(self.yaw)
		facing_z = np.sin(self.pitch)

		#double check normilizations of vectors
		return Vector3(facing_x, facing_y, facing_z).normalize()

	def Left(self):

		left_x = np.cos(self.pitch) *(-np.sin(self.yaw))
		left_y = np.cos(self.pitch) *  np.cos(self.yaw)
		left_z = np.sin(self.pitch)

		return Vector3(left_x, left_y, left_z).normalize()

	def TMat(self):
		return Get_TMat(self.Forward(), self.Left(), self.loc)

	def localize(self, vec):
		#Inverted from the local to global matrix generated
		matM = np.linalg.inv(self.TMat())
		outvec = Vectorize_Np(np.dot(matM,vec.np(1)))

		return outvec

	def globalize(self, vec):

		matM = self.TMat()
		outvec = Vectorize_Np(np.dot(matM,vec.np(1)))

		return outvec

	def p_array_to(self, t_loc):
		t_local = self.localize(t_loc)

		ret = t_local.p_array()
		ret[0] = Rad_clip(np.pi - ret[0])
		
		return ret

	def array_to(self, t_loc):
		t_local = self.localize(t_loc)

		ret = t_local.array()
		ret[0] = Rad_clip(np.pi - ret[0])
		
		return ret

	def p_array(self):
		return self.loc.p_array() + self.dir.p_array() + self.vel.p_array() + self.avl.p_array()

	def c_array(self):
		return self.loc.c_array() + self.dir.c_array() + self.vel.c_array() + self.avl.c_array()

	def array(self):
		return self.loc.array() + self.dir.array() + self.vel.array() + self.avl.array()


def Car_To_Vec(car):
	return Vectorize_Loc(car), Car_Forward(car), Vectorize_Vel(car), Vectorize_Avl(car)

def Rad_clip(val):
		# Make sure we go the 'short way'
		if abs(val) > np.pi:
			val += (2 * np.pi) if val < 0 else (- 2 * np.pi)
		return val


def Car_Forward(car):

	pitch = float(car.Rotation.Pitch) * URotationToRadians
	yaw = float(car.Rotation.Yaw) * URotationToRadians
	roll = float(car.Rotation.Roll) * URotationToRadians

	facing_x = np.cos(pitch) *  np.cos(yaw)
	facing_y = np.cos(pitch) *  np.sin(yaw)
	facing_z = np.sin(pitch)

	#double check normilizations of vectors
	return Vector3(facing_x, facing_y, facing_z).normalize()

def Car_Left(car):

	pitch = float(car.Rotation.Pitch) * URotationToRadians
	yaw = float(car.Rotation.Yaw) * URotationToRadians + np.pi/2
	roll = float(car.Rotation.Roll) * URotationToRadians

	facing_x = np.cos(pitch) *(-np.sin(yaw))
	facing_y = np.cos(pitch) *  np.cos(yaw)
	facing_z = np.sin(pitch)

	#double check normilizations of vectors
	return Vector3(facing_x, facing_y, facing_z).normalize()

def Car_TMat(car):
	return Get_TMat(Car_Forward(car), Car_Left(car), Vectorize_Loc(car))

def Get_TMat(f, l, t):
	#Generate matrix transform
	#X Forward, Y = Left, Z = Up

	#Y = M . X
	#M = Y . X-1
	#X = indentity because of f, l, u corresponding to unit vectors

	u = f.cross(l)

	#the vector t is for translations

	#This matrix convert from local to global
	return np.concatenate((f.np(0), l.np(0), u.np(0), t.np(1)), axis=1)

def to_local(car, vec):
	#Inverted from the local to global matrix generated
	matM = np.linalg.inv(Car_TMat(car))
	outvec = Vectorize_Np(np.dot(matM,vec.np(1)))

	return outvec

def to_global(car, vec):

	matM = Get_TMat(car)
	outvec = Vectorize_Np(np.dot(matM,vec.np(1)))

	return outvec

def predict(pos, vel, time=5, timestep=0.0166):
	path = [pos]
	vel_path = [vel]
	for _ in range(int(time / timestep)):
		acceleration = Vector3(0, 0, GRAVITY) - vel_path[-1] * AIR_RESISTANCE
		predicted_vel = vel_path[-1] + acceleration * timestep
		predicted_pos = path[-1] + vel_path[-1] + 0.5 * acceleration * timestep**2

		if predicted_pos.z < GROUND_Z_AXIS:
			predicted_pos.z = GROUND_Z_AXIS
			predicted_vel.z *= COR
			predicted_vel.z = abs(predicted_vel.z)

		path.append(predicted_pos)
		vel_path.append(predicted_vel)

	return path
