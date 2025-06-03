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

    def generate_waypoints_openai(self, image, system_prompt, waypoint_prompt, max_tokens=None,
                                    temperature=0.7, top_p = 1.0):
        user_content = []
        user_content.append({"type": "text", "text": waypoint_prompt})

        rgb_image_data = self._process_image_to_base64(image)
        # depth_data = self._process_image_to_base64(depth_map)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{rgb_image_data}"}
        })
        # user_content.append({
        #     "type": "image_url",
        #     "image_url": {"url": f"data:image/jpeg;base64,{depth_data}"}
        # })

        print('user_content', user_content)

        try:
            response = self.client.beta.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format="json",
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generate_waypoints_openai: {e}")
            return None

