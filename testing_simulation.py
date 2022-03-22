import traci
import numpy as np
import random
import timeit
import os

from training_simulation import PHASE_ALL_RED

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7
PHASE_ALL_RED = 8 

TL = "cluster_2775882079_2775882101_J10_J11_J12_J7_J8_J9"

class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, red_duration, num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._red_duration = red_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []


    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        old_action = -1 # dummy init
        last_phase_yellow = False

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)
                last_phase_yellow = True 

            if old_action == 1 and action == 0:
                last_phase_yellow = False
            elif old_action == 3 and action == 2:
                last_phase_yellow = False 
            elif old_action == action:
                last_phase_yellow = False 

            if last_phase_yellow == True:
                self._set_red_phase() 
                self._simulate(self._red_duration)
                last_phase_yellow = False
            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        #print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time


    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length() 
            self._queue_length_episode.append(queue_length)

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "E2TL1", "N2TL", "N2TL1", "N2TL2", "W2TL", "W2TL1", "W2TL2", "S2TL", "S2TL1"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time



    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.argmax(self._Model.predict_one(state))

    def _set_red_phase(self):
        traci.trafficlight.setPhase(TL, PHASE_ALL_RED)

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase(TL, yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """


        if action_number == 0:
            traci.trafficlight.setPhase(TL, PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase(TL, PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase(TL, PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase(TL, PHASE_EWL_GREEN)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL") + traci.edge.getLastStepHaltingNumber("N2TL1") + traci.edge.getLastStepHaltingNumber("N2TL2")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL") + traci.edge.getLastStepHaltingNumber("S2TL1")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL") + traci.edge.getLastStepHaltingNumber("E2TL1")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL") + traci.edge.getLastStepHaltingNumber("W2TL1") + traci.edge.getLastStepHaltingNumber("W2TL2")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        EAST_LEG = ["E2TL", "E2TL1"]
        WEST_LEG = ["W2TL", "W2TL1", "W2TL2"]
        NORTH_LEG = ["N2TL", "N2TL1", "N2TL2"]
        SOUTH_LEG = ["S2TL", "S2TL1"]

        E_TH_LANES = ["E2TL_0", "E2TL_1", "E2TL1_0", "E2TL1_1"]
        E_LT_LANES = ["E2TL_2"]
        W_TH_LANES = ["W2TL_0", "W2TL_1", "W2TL1_0", "W2TL1_1"]
        W_LT_LANES = ["W2TL_2"]
        N_TH_LANES = ["N2TL_0", "N2TL_1", "N2TL1_0", "N2TL1_1"]
        N_LT_LANES = ["N2TL_2", "N2TL1_2"]
        S_TH_LANES = ["S2TL_0", "S2TL_1", "S2TL_2", "S2TL1_0", "S2TL1_1", "E2TL1_2"]
        S_LT_LANES = ["S2TL_3"]

        car_pos = {}

        veh_list = traci.vehicle.getIDList()
        for veh in veh_list:
            lane_pos = traci.vehicle.getLanePosition(veh)
            lane_id = traci.vehicle.getLaneID(veh)
            road_id = traci.vehicle.getRoadID(veh)
            lane_cell = 5
            #WEST LEG #LANE GROUP 0 AND 1 
            if road_id in WEST_LEG:
                if road_id == WEST_LEG[0]:
                    lane_length = traci.lane.getLength(WEST_LEG[0]+"_1")
                    lane_pos = lane_length - lane_pos
                    if lane_id in W_LT_LANES:
                        if lane_pos < 7:
                            lane_cell = 0 
                        elif lane_pos < 14:
                            lane_cell = 1
                        elif lane_pos < 21:
                            lane_cell = 2
                        elif lane_pos < 28: 
                            lane_cell = 3
                        elif lane_pos < 35:
                            lane_cell = 4 
                        elif lane_pos < 42:
                            lane_cell = 5 
                        elif lane_pos < 63:
                            lane_cell = 6 
                        elif lane_pos < 84:
                            lane_cell = 7 
                        elif lane_pos < 105:
                            lane_cell = 8 
                        elif lane_pos < 134:
                            lane_cell = 9 
                    elif lane_id in W_TH_LANES:
                        if lane_pos < 7:
                            lane_cell = 0 
                        elif lane_pos < 14:
                            lane_cell = 1
                        elif lane_pos < 21:
                            lane_cell = 2
                        elif lane_pos < 28: 
                            lane_cell = 3
                        elif lane_pos < 35:
                            lane_cell = 4 
                        elif lane_pos < 50:
                            lane_cell = 5 
                        elif lane_pos < 75:
                            lane_cell = 6 
                        elif lane_pos < 100:
                            lane_cell = 7 
                elif road_id == ":WTL_0":
                    lane_cell = 7 
                elif road_id == WEST_LEG[1]:
                    lane_length = traci.lane.getLength(WEST_LEG[1]+"_1")
                    lane_pos = lane_length - lane_pos 
                    if lane_pos < 150 - traci.lane.getLength(WEST_LEG[0]+"_1"):
                        lane_cell = 8 
                    else:
                        lane_length = traci.lane.getLength(WEST_LEG[1]+"_1")
                        lane_pos = lane_length - lane_pos 
                        lane_cell = 9 
                elif road_id == WEST_LEG[2]:
                    lane_length = traci.lane.getLength(WEST_LEG[2]+"_1")
                    lane_pos = lane_length - lane_pos 
                    lane_cell = 9     
            #NORTH LEG            
            elif road_id in NORTH_LEG:
                if road_id == NORTH_LEG[0]:
                    lane_length = traci.lane.getLength(NORTH_LEG[0]+"_1")
                    lane_pos = lane_length - lane_pos
                    if lane_id in N_LT_LANES:
                        if lane_pos < 7:
                            lane_cell = 0 
                        elif lane_pos < 14:
                            lane_cell = 1
                        elif lane_pos < 21:
                            lane_cell = 2   
                    elif lane_id in N_TH_LANES:
                        if lane_pos < 7:
                            lane_cell = 0 
                        elif lane_pos < 14:
                            lane_cell = 1
                        elif lane_pos < 21:
                            lane_cell = 2                  
                elif road_id == ":NTL_0":
                    lane_cell = 2
                elif road_id == NORTH_LEG[1]:                 
                    lane_length = traci.lane.getLength(NORTH_LEG[1]+"_1")
                    lane_pos = lane_length - lane_pos
                    if lane_id in N_LT_LANES:
                        if lane_pos < 28 - traci.lane.getLength(NORTH_LEG[0]+"_1"): 
                            lane_cell = 3
                        elif lane_pos < 35 - traci.lane.getLength(NORTH_LEG[0]+"_1"):
                            lane_cell = 4 
                        elif lane_pos < 42 - traci.lane.getLength(NORTH_LEG[0]+"_1"):
                            lane_cell = 5 
                        elif lane_pos < 49 - traci.lane.getLength(NORTH_LEG[0]+"_1"):
                            lane_cell = 6  
                        elif lane_pos < 56 - traci.lane.getLength(NORTH_LEG[0]+"_1"):
                            lane_cell = 7
                        elif lane_pos < 63 - traci.lane.getLength(NORTH_LEG[0]+"_1"):
                            lane_cell = 8   
                        elif lane_pos < 70 - traci.lane.getLength(NORTH_LEG[0]+"_1"):
                            lane_cell = 9  
                    elif lane_id in N_TH_LANES:
                        if lane_pos < 28 - traci.lane.getLength(NORTH_LEG[0]+"_1"): 
                            lane_cell = 3
                        elif lane_pos < 35 - traci.lane.getLength(NORTH_LEG[0]+"_1"):
                            lane_cell = 4 
                        elif lane_pos < 50 - traci.lane.getLength(NORTH_LEG[0]+"_1"):
                            lane_cell = 5 
                        elif lane_pos < 75 - traci.lane.getLength(NORTH_LEG[0]+"_1"):
                            lane_cell = 6           
                elif road_id == ":NTL1_0":
                    lane_cell = 6
                elif road_id == NORTH_LEG[2]:
                    lane_length = traci.lane.getLength(NORTH_LEG[2]+"_1")
                    lane_pos = lane_length - lane_pos
                    if lane_pos < 120 - traci.lane.getLength(NORTH_LEG[0]+"_1") - traci.lane.getLength(NORTH_LEG[1]+"_1"): 
                        lane_cell = 7
                    elif lane_pos < 200 - traci.lane.getLength(NORTH_LEG[0]+"_1") - traci.lane.getLength(NORTH_LEG[1]+"_1"): 
                        lane_cell = 8
                    else:
                        lane_cell = 9                    
            #EAST LEG 
            elif road_id in EAST_LEG:
                if road_id == EAST_LEG[0]:
                    lane_length = traci.lane.getLength(EAST_LEG[0]+"_1")
                    lane_pos = lane_length - lane_pos
                    if lane_id in E_LT_LANES:
                        if lane_pos < 7:
                            lane_cell = 0 
                        elif lane_pos < 14:
                            lane_cell = 1
                        elif lane_pos < 21:
                            lane_cell = 2
                        elif lane_pos < 28: 
                            lane_cell = 3
                        elif lane_pos < 35:
                            lane_cell = 4 
                        elif lane_pos < 42:
                            lane_cell = 5 
                        elif lane_pos < 49:
                            lane_cell = 6 
                        elif lane_pos < 70:
                            lane_cell = 7 
                        elif lane_pos < 91:
                            lane_cell = 8 
                        elif lane_pos < 118:
                            lane_cell = 9   
                    elif lane_id in E_TH_LANES:                      
                        if lane_pos < 7:
                            lane_cell = 0 
                        elif lane_pos < 14:
                            lane_cell = 1
                        elif lane_pos < 21:
                            lane_cell = 2
                        elif lane_pos < 28: 
                            lane_cell = 3
                        elif lane_pos < 35:
                            lane_cell = 4 
                        elif lane_pos < 50:
                            lane_cell = 5 
                        elif lane_pos < 75:
                            lane_cell = 6 
                        elif lane_pos < 100:
                            lane_cell = 7 
                elif road_id == ":ETL_0":
                    lane_cell = 7 
                elif road_id == EAST_LEG[1]:
                    lane_length = traci.lane.getLength(EAST_LEG[1]+"_1")
                    lane_pos = lane_length - lane_pos
                    if lane_pos < 150 - traci.lane.getLength(EAST_LEG[0]+"_1"):
                        lane_cell = 8 
                    else:
                        lane_cell = 9 
            
             #SOUTH LEG    
            elif road_id in SOUTH_LEG:
                if road_id == SOUTH_LEG[0]:
                    lane_length = traci.lane.getLength(SOUTH_LEG[0]+"_1")
                    lane_pos = lane_length - lane_pos
                    if lane_id in S_LT_LANES:
                        if lane_pos < 7:
                            lane_cell = 0 
                        elif lane_pos < 14:
                            lane_cell = 1
                        elif lane_pos < 21:
                            lane_cell = 2
                        elif lane_pos < 28: 
                            lane_cell = 3
                        elif lane_pos < 35:
                            lane_cell = 4 
                        elif lane_pos < 42:
                            lane_cell = 5 
                        elif lane_pos < 49:
                            lane_cell = 6 
                        elif lane_pos < 56:
                            lane_cell = 7 
                        elif lane_pos < 63:
                            lane_cell = 8 
                        elif lane_pos < 83:
                            lane_cell = 9   
                    elif lane_id in S_TH_LANES:                     
                        if lane_pos < 7:
                            lane_cell = 0 
                        elif lane_pos < 14:
                            lane_cell = 1
                        elif lane_pos < 21:
                            lane_cell = 2
                        elif lane_pos < 28: 
                            lane_cell = 3
                        elif lane_pos < 35:
                            lane_cell = 4 
                        elif lane_pos < 50:
                            lane_cell = 5 
                        elif lane_pos < 75:
                            lane_cell = 6 
                    #print(veh, lane_cell, lane_pos)
                elif road_id == ":STL_0":
                    lane_cell = 6
                elif road_id == SOUTH_LEG[1]:
                    lane_length = traci.lane.getLength(SOUTH_LEG[1]+"_1")
                    lane_pos = lane_length - lane_pos 
                    if lane_pos < 100 - traci.lane.getLength(SOUTH_LEG[0]+"_1"):
                        lane_cell = 7 
                    elif lane_pos < 150 - traci.lane.getLength(SOUTH_LEG[0]+"_1"):
                        lane_cell = 8 
                        #print(veh, lane_cell, lane_pos)
                    else:
                        lane_cell = 9 
                        #print(veh, lane_cell, lane_pos)


            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL1_0" or lane_id == "W2TL1_1" or lane_id == "W2TL2_0" or lane_id == "W2TL2_1":
                lane_group = 0
            elif lane_id == "W2TL_2":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL1_0" or lane_id == "N2TL1_1" or lane_id == "N2TL2_0" or lane_id == "N2TL2_1":
                lane_group = 2
            elif lane_id == "N2TL_2" or lane_id == "N2TL1_2":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id =="E2TL1_0" or lane_id == "E2TL1_1":
                lane_group = 4
            elif lane_id == "E2TL_2":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2" or lane_id == "S2TL1_0" or lane_id == "S2TL1_1":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"
        return state


    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode



