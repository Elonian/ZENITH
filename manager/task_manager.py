import random
import json
import os
import time
import traceback
from simworld.utils.vector import Vector
from map.map import Map
from simworld.utils.logger import Logger
from agent.nav_agent import NavAgent
from llm.nav_llm import NavLLM
from concurrent.futures import ThreadPoolExecutor
from threading import Event

class TaskManager:
    def __init__(self, config, communicator):
        random.seed(config['simworld.seed'])

        self.config = config
        self.communicator = communicator
        
        self.exit_event = Event()
        self.agent = self.initialize_agent(config)

    def initialize_agent(self, config:str):
        with open(self.config['navReq.tasks_file'], 'r') as f:
            tasks_file = json.load(f)
        
        selected_task = tasks_file[self.config['navReq.task_num']]

        start_position = Vector(selected_task['origin'['x']],selected_task['origin']['y'])
        end_position = Vector(selected_task['destination']['x', selected_task['destination']['y']])

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
                    agents.append(agent)
                else:
                    continue 
            return agents
    
    def create_world(self):
        world_json = self.config['worldGen.world_json']
        ue_assets_path = self.config['ue_assets_path']
        self.communicator.generate_world(world_json, ue_assets_path)
    
    def load_agent(self):
        for agent in self.agents:
            agent_model_path = self.config['navReq.agent_model_path']
            agent_speed = self.config['navReq.agent_speed']
            self.communicator.spawn_agent(agent, agent_model_path)
            self.communicator.agent_set_speed(agent.id, agent_speed)
                   


