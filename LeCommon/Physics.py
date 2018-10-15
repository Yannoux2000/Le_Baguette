import numpy as np
from LeCommon.Vector import *
from LeCommon.Objs import *


#inputs are vectors
def bounce(ball_pos, ball_vel, ball_avl, normal):

	v_perp = ball_pos.dot(normal) * normal
	v_para = ball_vel - v_perp
	v_spin = R * normal.cross(ball_avl)
	s = v_para + v_spin
	
	ratio = v_perp.magnitude / s.magnitude
	
	delta_v_perp = - (1.0 + C_R) * v_perp
	delta_v_para = - min(1.0, Y * ratio) * mu * s

	return ball_pos, ball_vel + delta_v_perp + delta_v_para, ball_avl + A * R * delta_v_para.cross(normal)

#Copied from discord not used.
def predict(pos, vel, avl, time=5, timestep=0.0166):
	path = [pos]
	vel_path = [vel]
	for _ in range(int(time / timestep)):
		acceleration = Vector3(0, 0, GRAVITY) - vel_path[-1] * AIR_RESISTANCE
		predicted_vel = vel_path[-1] + acceleration * timestep
		predicted_pos = path[-1] + vel_path[-1] + 0.5 * acceleration * timestep**2

		if predicted_pos.z < GROUND_Z_AXIS:
			bounce(pos, vel, avl, normal)

		path.append(predicted_pos)
		vel_path.append(predicted_vel)

	return path
