import random
import json
import os
import time
import traceback
from utils.vector import Vector
#from map.map import Map
from simworld.utils.logger import Logger
from agent.nav_agent import NavAgent
from nav_llm.nav_llm import NavLLM
from concurrent.futures import ThreadPoolExecutor
from threading import Event

class TaskManager:
    def __init__(self, config, communicator):
        random.seed(config['simworld.seed'])

        self.config = config
        self.communicator = communicator
        
        self.exit_event = Event()
        self.agents = self.initialize_agent(config)

    def initialize_agent(self, config:str):
        with open(self.config['navReq.tasks_file'], 'r') as f:
            tasks_file = json.load(f)
        
        selected_task = tasks_file[self.config['navReq.task_num']]

        start_position = Vector(selected_task['origin']['x'],selected_task['origin']['y'])
        end_position = Vector(selected_task['destination']['x'], selected_task['destination']['y'])

        with open(self.config['navReq.agent_config'], 'r') as f:
            agent_config = json.load(f)

        isMultiagent = self.config['navReq.multi_agent']
        agent_model_type = self.config['navReq.agent_model_type']
        agents = []
        if not isMultiagent:
            for agent_name, agent_config in agent_config.items():
                if agent_name == agent_model_type and agent_config['provider'] == 'openai':
                    nav_agent_llm = NavLLM(agent_config['model'], agent_config['url'], os.getenv('OPENAI_API_KEY'))
                    nav_agent = NavAgent(
                        position = start_position, 
                        direction = Vector(0, 0), 
                        communicator = self.communicator,
                        nav_llm = nav_agent_llm,
                        destination = end_position,
                        config = self.config)
                    agents.append(nav_agent)
                else:
                    continue 
            return agents
    
    def create_world(self):
        world_json = self.config['worldGen.world_json']
        ue_assets_path = self.config['worldGen.ue_assets_path']
        self.communicator.generate_world(world_json, ue_assets_path)
    
    def load_agent(self):
        for agent in self.agents:
            agent_model_path = self.config['navReq.agent_model_path']
            agent_speed = self.config['navReq.agent_speed']
            self.communicator.spawn_agent(agent, agent_model_path)
            self.communicator.agent_set_speed(agent.id, agent_speed)

    def update_physical_states(self):
        agent_ids = [agent.id for agent in self.agents]
        result = self.communicator.get_position_and_direction(
            vehicle_ids = [],
            pedestrian_ids = [],
            traffic_signal_ids = [],
            agent_ids = agent_ids)
        
        for (type, object_id), values in result.items():
            if type == 'agent':
                position, direction = values
                self.agents[object_id].position = position
                self.agents[object_id].direction = direction

    def run_task(self):
        num_threads = self.config['navReq.num_threads']
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            try:
                futures = []
                for agent in self.agents:
                    future = executor.submit(agent.run, self.exit_event)
                    futures.append(future)

                while True:
                    if all(future.done() for future in futures):
                        print("Agent has reached its final destination")
                        break
                    self.update_physical_states()

                    time.sleep(self.dt)
            
            except KeyboardInterrupt:
                print("User stopped it")
                self.exit_event.set()
            except Exception as e:
                print(f"An error occurred: {e}")
                self.exit_event.set()
            finally:
                print("Waiting for all agents to finish...")
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error in thread: {e}")

                print("Simulation fully stopped.")
            
                   


