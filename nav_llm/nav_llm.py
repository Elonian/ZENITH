import base64
import io
from json import loads
import cv2
from PIL import Image
import numpy as np
from pydantic import BaseModel

from simworld.llm.base_llm import BaseLLM
# from agent.action_space import ActionSpace

# define LLM output JSON format
class Waypoint(BaseModel):
    x: int
    y: int
class WaypointList(BaseModel):
    waypoints: list[Waypoint]

class NavLLM(BaseLLM):
    def __init__(self, model_name, url, api_key):
        super().__init__(model_name,url, api_key)

    def generate_waypoints_openai(self, image, depth_map, system_prompt, waypoint_prompt, max_tokens=None,
                                    temperature=0.7, top_p = 1.0, response_format=WaypointList):
        user_content = []
        user_content.append({"type": "text", "text": waypoint_prompt})

        rgb_image_data = self._process_image_to_base64(image)
        depth_data = self._process_image_to_base64(depth_map)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{rgb_image_data}"}
        })
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{depth_data}"}
        })

        print('user_content', user_content)

        try:
            response = self.client.beta.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generate_waypoints_openai: {e}")
            return None

    def select_waypoints_openai(image, waypoints, system_prompt, waypoint_prompt, max_tokens=None,
                                    temperature=0.7, top_p = 1.0, response_format=WaypointList):
        # image: PIL Image
        # waypoints: list [(x1,y1), (x2,y2), ..., (x12,y12)] of integers
    
        # add waypoints to image
        color_map = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462",
                     "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f",]
    
        waypoint_image = image.copy()
        legend = []
        wi_draw = ImageDraw.Draw(waypoint_image)
        for p,c in zip(waypoints, color_map):
            legend.append({'x': p[0], 'y': p[1], 'color': c})
            wi_draw.circle(p, 3, fill=c)
        #display(waypoint_image)

        # build prompt
        image_url = self._process_image_to_base64(waypoint_image)
        user_content = []
        user_content.append({"type": "text", "text": waypoint_prompt})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })
        user_content.append({"type": "text", "text": str(legend)})
        #print('user_content', user_content)

        # run prompt
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generate_waypoints_openai: {e}")
            return None

