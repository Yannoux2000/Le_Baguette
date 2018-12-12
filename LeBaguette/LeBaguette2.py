from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator

from LeFramework.exercices.AirExercice import FixedBallExercice, AirRotateExercice
from LeFramework.exercices.GroundExercice import RandomPlaceExercice

from LeFramework.common.Objs import Car, Ball, Target
from LeFramework.common.GameMap import MapInfo
# from LeFramework.common.Areas import *
from LeFramework.common.Vector import Vec3, UNIT_X, UNIT_Y, UNIT_Z
from LeFramework.common.Regulators import Timer, RESET_ALL_TIMERS
# from LeFramework.common.ConstVec import *

DT_STEP = 0.016666
ALL_TIMERS = []

class LeBaguette2(BaseAgent):
	def initialize_agent(self):
		RESET_ALL_TIMERS()
		self.side = (self.team * 2) - 1

		self.prev_game_s = 0

		#reference to the important parts
		self.time = 0
		self.dt = 0
		
		#load exercice
		self.exercice = RandomPlaceExercice(self.index, self.team)
		self.set_game_state(self.exercice.reset())

	def render(self):
		self.renderer.begin_rendering()
		# self.renderer.draw_line_3d([*self.me.loc],[*(Vec3([1,0,0]) * 100 + self.me.loc)], self.renderer.red())
		# self.renderer.draw_line_3d([*self.me.loc],[*(self.me.to_local(self.ball) + self.me.loc)], self.renderer.green())
		# self.renderer.draw_line_3d([*self.me.loc],[*((self.actions.airial.t_f * 300) + self.me.loc)], self.renderer.blue())
		# self.renderer.draw_line_3d([*self.me.loc],[*self.me.step(5).loc], self.renderer.red())
		# self.renderer.draw_line_3d([*self.me.loc],[*(self.me.ang * 50 + self.me.loc)], self.renderer.blue())
		self.renderer.end_rendering()
		# pass

	def preprocess(self, packet):
		new_t = packet.game_info.seconds_elapsed
		self.dt = new_t - self.time
		self.time = new_t


	def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
		self.preprocess(packet)
		self.timers_update()

		curr_game_s = packet.game_info.seconds_elapsed
		if self.prev_game_s == curr_game_s:
			self.prev_game_s = curr_game_s
			return SimpleControllerState()

		print(self.exercice.reward(packet))
		self.set_game_state(self.exercice(packet))
		# print(self.dt)

		return SimpleControllerState()

	def timers_update(self):
		for t in ALL_TIMERS:
			t.step(self.dt)
