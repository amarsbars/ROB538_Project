__author__ = "Ovunc Tuzel"

import random, math, utils
from NN_Unsupervised import NeuralNet
from operator import attrgetter
import copy

class RoverDomain(object):


	# Constructor
	def __init__(self, min_sensor_dist, max_sensor_dist, num_rovers, num_pois, world_width=240, world_height=240):
		self.min_sensor_dist	= min_sensor_dist
		self.max_sensor_dist	= max_sensor_dist
		self.num_rovers			= num_rovers
		self.num_pois			= num_pois
		self.world_width		= world_width
		self.world_height		= world_height
		self.rover_list			= []
		self.poi_list			= []


	# Initialize world
	def init_world(self, poi_min_vel, poi_max_vel):

		# Initializing rovers
		for i in range(self.num_rovers):
			self.add_rover(random.randint(0, self.world_height), random.randint(0, self.world_width), random.randint(0, 360))

		# Initializing POIs
		for i in range(self.num_pois):
			self.add_poi(random.randint(0, self.world_height), random.randint(0, self.world_width), random.randint(0, 360))

		# Setting POIs initial velocities
		for poi in self.poi_list:
			poi.heading = random.randint(0, 360)
			poi.set_vel_lin(random.uniform(poi_min_vel, poi_max_vel))


	# Iterate the world simulation
	def sim_step(self):

		# POIs step
		for poi in self.poi_list:
			poi.sim_step(self.world_width,self.world_height)

		# Rovers step
		for rover in self.rover_list:
			inputs = self.return_NN_inputs(rover)
			outputs = random.choice(rover.population).forward(inputs)
			rover.sim_step(outputs)

	# Initialize NNs for each rover
	def initRoverNNs(self, pop_size, inputLayers, outputLayers, hiddenLayers):
		for rover in self.rover_list:
			for i in range(pop_size):
				rover.population.append(NeuralNet(inputLayers, outputLayers, hiddenLayers))


	# Initialize NNs for each rover
	def mutateNNs(self, perturbation):
		for rover in self.rover_list:
			mutantlist = []
			for nn in rover.population:
				mutant = copy.deepcopy(nn)
				mutant.perturb_weights(perturbation)
				mutantlist.append(mutant)
			rover.population += mutantlist


	# Selecting best NNs
	def select(self):

		# Retain k best
		for rover in self.rover_list:
			rover.population = self.remove_worst(rover.population)


	# Removing the worst performing rover
	def remove_worst(self, nn_list, k=None):
		if k == None:
			k = len(nn_list)/2
		for i in range(k):
			random.shuffle(nn_list)
			worst = min(nn_list, key=attrgetter('performance'))
			nn_list.remove(worst)
		return nn_list

	# Registering new POI
	def add_poi(self, x=0, y=0, heading=0, value=1.0):
		self.poi_list.append(Poi(x, y, heading, value))


	# Registering new rover
	def add_rover(self, x=0, y=0, heading=0):
		self.rover_list.append(Rover(x, y, heading))


	# Resetting agents to random or initial starting position
	def reset_agents(self, rnd_pois = 1, rnd_rovers = 1):

		# Resetting POIs
		for i in self.poi_list:
			i.pos = i.init_pos if not rnd_pois else (random.randint(0,self.world_width), random.randint(0,self.world_height))

		# Resetting Rovers
		for i in self.rover_list:
			i.pos = i.init_pos if not rnd_pois else (random.randint(0,self.world_width), random.randint(0,self.world_height))

	# Computing sensor measurement
	def return_sensor(self, agentList, quadrant, rover):
		sum = 0
		for agent in agentList:
			vect = utils.vect_sub(agent.pos, rover.pos)
			dist = utils.get_norm(vect)
			angle = utils.get_angle(vect)
			relative_angle = (angle - rover.heading + math.pi/2) % (2*math.pi)
			if dist < self.max_sensor_dist and utils.check_quadrant(relative_angle, quadrant):
				sum += agent.value / max(dist**2, self.min_sensor_dist**2)
		return sum
	
	def return_POI_vel(self, poiList, max_dist = 500):
                min_dist = 10
                sum = np.zeros([4,2])
                count = np.zeros(4)
                for poi in poiList:

                    #get quarant of POI
                    vect = utils.vect_sub(poi.pos, self.pos)
                    dist = utils.get_norm(vect)
                    angle = utils.get_angle(vect) % (2*math.pi ) # Between 0 to 2pi
                    relative_angle = (angle - self.heading + math.pi/2) % (2*math.pi)
                    q = utils.get_quadrant(relative_angle) - 1
                    
                    #get relative velocity of POI to agent.
                    poi_vel_vect = np.array([math.cos(poi.heading), math.sin(poi.heading)])*poi.speed
                    rover_vel_vect = np.array([math.cos(self.heading), math.sin(self.heading)])*self.speed
                    rel_vel_vect = rover_vel_vect - poi_vel_vect
                    if q == 2:
                        rel_vel_vect[1] *= -1
                    elif q == 3:
                        rel_vel_vect[0] *= -1

                    
                    #update average velocity vector
                    count[q] += 1
                    sum[q][0] = (sum[q][0] * (count[q] - 1) + rel_vel_vect[0]) / count[q]
                    sum[q][1] = (sum[q][1] * (count[q] - 1) + rel_vel_vect[1]) / count[q]

                return sum
	
	
	
	# Gathering all sensor measurements
	def return_NN_inputs(self, rover):

		inputs = []

		# Sensing rovers
		for i in range(4):
			inputs.append(self.return_sensor(self.rover_list, i, rover))

		# Sensing POIs
		for i in range(4):
			inputs.append(self.return_sensor(self.poi_list, i, rover))

		return inputs

# General agent that models POIs and rovers
class Agent(object):

	# Constructor
	def __init__(self, posx, posy, heading, value = 1.0):
		self.init_pos	= (posx, posy)		# Starting position
		self.pos		= (posx, posy)		# Current position
		self.vel_lin	= (0.0,0.0)			# Linear velocity
		self.vel_ang	= 0.0				# Angular velocity (rad/sec)
		self.heading	= heading			# Heading direction (rad)	
		self.value		= value				# Utility value	

	# Update heading using angular velocity
	def update_heading(self):
		self.heading += self.vel_ang
		self.set_vel_lin(self.get_vel_lin())

	# Update position using linear velocity
	def update_pos(self):
		self.pos = utils.vect_sum(self.pos, self.vel_lin)

	# Set heading and update velocity accordingly
	def set_heading(self, heading):
		self.heading = heading
		self.set_vel_lin(self.get_vel_lin())

	# Get absolute velocity
	def get_vel_lin(self):
		return utils.get_norm(self.vel_lin)

	# Set absolute velocity
	def set_vel_lin(self,vel_lin_abs):
		self.vel_lin = vel_lin_abs*math.cos(self.heading), vel_lin_abs*math.sin(self.heading)

	# Wall bouncing
	def bounce_walls(self, world_width, world_height):

		# Check left-right wall collisions
		if self.pos[0] > world_width or self.pos[0] < 0:
			self.set_heading((1*math.pi - self.heading) % (math.pi * 2))

		# Check top-bottom wall collisions
		if self.pos[1] > world_height or self.pos[1] < 0:
			self.set_heading((2*math.pi - self.heading) % (math.pi * 2))

# POI agent
class Poi(Agent):
	def __init__(self, posx, posy, heading, value):
		Agent.__init__(self, posx, posy, heading, value)

	# Simulation step for the POIs
	def sim_step(self, world_width, world_height):
		self.update_heading();
		self.update_pos();
		self.bounce_walls(world_width, world_height)

# Rover agent
class Rover(Agent):

	def __init__(self, posx, posy, heading):
		Agent.__init__(self, posx, posy, heading, 1.0)
		self.population = []

	# Simulation step for the rovers
	def sim_step(self, nn_outputs):
		self.vel_ang = nn_outputs[0]
		self.set_vel_lin(nn_outputs[1])
		self.update_heading();
		self.update_pos();
