import base64
import io
from json import loads
import cv2
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import numpy as np
from pydantic import BaseModel

# from agent.action_space import ActionSpace
from simworld.llm.base_llm import BaseLLM
from utils.prompt_utils import (
    WAYPOINT_SYSTEM_PROMPT, WAYPOINT_GENERATION_PROMPT, WAYPOINT_VERIFICATION_PROMPT, WAYPOINT_SELECTION_PROMPT,
    WAYPOINT_NO_REASONING, WAYPOINT_REASONING, WAYPOINT_LIST_NO_REASONING, WAYPOINT_LIST_REASONING
)

# define LLM output JSON format
class WaypointBase(BaseModel):
    # json {"x":<x>, "y":<y>}
    x: int
    y: int
class Waypoint(BaseModel):
    # json {"waypoint": {"x":<x>, "y":<y>}}
    waypoint: WaypointBase
class WaypointList(BaseModel):
    # json {"waypoints": [{"x":<x1>,"y":<y1>}, ..., {"x":<xn>,"y":<yn>}]}
    waypoints: list[WaypointBase]
class ReasonedWaypoint(BaseModel):
    # json {"reasoning": "<reasoning>", "waypoint": {"x":<x>, "y":<y>}}
    reasoning: str
    waypoint: WaypointBase
class ReasonedWaypointList(BaseModel):
    # json {"reasoning": "<reasoning>", "waypoints": [{"x":<x1>,"y":<y1>}, ..., {"x":<xn>,"y":<yn>}]}
    reasoning: str
    waypoints: list[WaypointBase]

class NavLLM(BaseLLM):
    def __init__(self, model_name, url):
        super().__init__(model_name,url)
    
    def _process_image_to_base64(self, image: np.ndarray) -> str:
        """Convert a NumPy RGB image to base64-encoded PNG string."""
        if isinstance(image, np.ndarray):
            # Convert to PIL Image if necessary
            img_pil = Image.fromarray(image.astype('uint8'))
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_base64
        else:
            raise ValueError("Input image must be a NumPy array")

    def _make_system_prompt(self, response_format) -> str:
        """ Write the system prompt for the given response format """
        prompt = WAYPOINT_SYSTEM_PROMPT
        if response_format is Waypoint:
            prompt += WAYPOINT_NO_REASONING
        elif response_format is ReasonedWaypoint:
            prompt += WAYPOINT_REASONING
        elif response_format is WaypointList:
            prompt += WAYPOINT_LIST_NO_REASONING
        elif response_format is ReasonedWaypointList:
            prompt += WAYPOINT_LIST_REASONING
        return prompt
        
    def generate_segmented_overlay(self, image):
        try:
            segmentation_prompt = """
            Analyze this image and identify key navigational elements like:
            - Walkable paths
            - Obstacles
            - Walls and boundaries
            - Points of interest
            Return the segmentation information in a structured format.
            """

            image_data = self._process_image_to_base64(image)
    
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a vision assistant that helps with scene segmentation."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": segmentation_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                temperature=0.2
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generate_segmented_overlay: {e}")
            return None

    def generate_waypoints_openai(self, image, depth_map, seg_mask, reasoning=False,
                                  max_tokens=None, temperature=0.7, top_p = 1.0,
                                  verbose=True):
        
        response_format = ReasonedWaypointList if reasoning else WaypointList
        
        print("Generate waypoints openai functin in navllm started.")
        user_content = []
        user_content.append({"type": "text", "text": WAYPOINT_GENERATION_PROMPT})

        rgb_image_data = self._process_image_to_base64(image)
        depth_data = self._process_image_to_base64(depth_map)
        seg_data = self._process_image_to_base64(seg_mask)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{rgb_image_data}"}
        })
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{depth_data}"}
        })
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{seg_data}"}
        })

        if verbose:
            print('user_content', user_content)

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[{"role": "system", "content": self._make_system_prompt(response_format)},
                        {"role": "user", "content": user_content}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=response_format,
            )
            print("generate waypoint openai has recieved response.")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generate_waypoints_openai: {e}")
            return None

    def validate_waypoints_openai(self, image, waypoints, reasoning=False,
                                max_tokens=None, temperature=0.7, top_p = 1.0,
                                verbose=True):
        # image: PIL Image
        # waypoints: list [(x1,y1), (x2,y2), ..., (x12,y12)] of integers

        response_format = ReasonedWaypointList if reasoning else WaypointList
    
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
        user_content.append({"type": "text", "text": WAYPOINT_VERIFICATION_PROMPT})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_url}"}
        })
        user_content.append({"type": "text", "text": str(legend)})

        if verbose:
            print('user_content', user_content)

        # run prompt
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[{"role": "system", "content": self._make_system_prompt(response_format)},
                        {"role": "user", "content": user_content}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in select_waypoints_openai: {e}")
            return None
        
    def select_best_waypoint(self, image, waypoints, 
                             current_pos, destination, history, distances_from_current, distances_to_destination,
                             reasoning=False,
                             max_tokens=None, temperature=0.7, top_p=1.0,
                             verbose=False):

        response_format = ReasonedWaypoint if reasoning else Waypoint

        image_data = self._process_image_to_base64(image)
        scene_data = self.generate_segmented_overlay(image)
        waypoint_data = "\n".join([f"- {label}: {coords}" for label, coords in waypoints.items()])
        text_inputs = {
            "SCENE_ANALYSIS": scene_data,
            "CURRENT_POS": current_pos,
            "DESTINATION": destination,
            "PREVIOUS_POS": history[-5:], #{history[-5:] if len(history)>5 else history},
            "DISTANCES_FROM_CURRENT": distances_from_current,
            "DISTANCES_TO_DESTINATION": distances_to_destination,
            "WAYPOINT_TEXT": waypoint_data,
        }
        prompt = WAYPOINT_SELECTION_PROMPT.format(**text_inputs)

        if verbose:
            print('user_content', prompt)

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._make_system_prompt(response_format)},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=response_format,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in select_best_waypoint: {e}")
            return None