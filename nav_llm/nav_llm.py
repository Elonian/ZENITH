import base64
import io
import cv2
from PIL import Image
import numpy as np

from simworld.llm.base_llm import BaseLLM
# from agent.action_space import ActionSpace

class NavLLM(BaseLLM):
    def __init__(self, model_name, url, api_key):
        super().__init__(model_name,url, api_key)

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
    
            response = self.client.chat.completions.create(
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

    def generate_waypoints_openai(self, image, depth_map, system_prompt, waypoint_prompt, max_tokens=None,
                                    temperature=0.7, top_p = 1.0, response_format="json"):
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
        
    def select_best_waypoint(self, image, waypoints, current_pos, destination, history):
        scene_analysis = self.generate_segmented_overlay(image)
        
        image_data = self._process_image_to_base64(image)

        # waypoint_text = "\n".join([f"- {label}: {coords}" for label, coords in waypoints.items()])
        prompt = f"""
        Scene Analysis: {scene_analysis}
        
        Current position: {current_pos}
        Destination: {destination}
        Previous positions: {history[-5:] if len(history) > 5 else history}
        
        Available waypoints:
        {waypoint_text}
        
        Choose the best waypoint considering:
        1. Distance to destination
        2. Avoiding previously visited areas
        3. Clear path without obstacles
        4. Natural movement flow
        
        Return only the waypoint label (A, B, C, etc).
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a navigation assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                temperature=0.2
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in select_best_waypoint: {e}")
            return None

