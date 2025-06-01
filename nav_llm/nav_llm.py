import base64
import io
import cv2
from PIL import Image
import numpy as np

from simworld.llm.base_llm import BaseLLM
#from agent.action_space import ActionSpace

class NavLLM(BaseLLM):
    def __init__(self, model_name, url, api_key):
        super().__init__(model_name, url, api_key)

    # def generate_nav_instructions_openai(self, image, system_prompt, user_prompt, max_tokens=None, 
    #                                      temperature=0.7, top_p=1.0, response_format=ActionSpace):
    #     user_content = []
    #     user_content.append({"type": "text", "text": user_prompt})

    #     img_data = self._process_image_to_base64(image)
    #     user_content.append({
    #         "type": "image_url",
    #         "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
    #     })
    #     print('user_content', user_content)

    #     try:
    #         response = self.client.beta.chat.completions.parse(
    #             model=self.model_name,
    #             messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
    #             max_tokens=max_tokens,
    #             temperature=temperature,
    #             top_p=top_p,
    #             response_format=response_format,
    #         )
    #         return response.choices[0].message.content
    #     except Exception as e:
    #         print(f"Error in generate_nav_instructions_openai: {e}")
    #         return None
        
    


    # def generate_nav_instructions_openrouter(self, origin, destination):
    #     prompt = f"Generate navigation instructions from {origin} to {destination}"
    #     response = self.generate_text(prompt)
    #     return response
        

    # def _process_image_to_base64(self, image: np.ndarray) -> str:
    #     """Convert numpy array image to base64 string.

    #     Args:
    #         image (np.ndarray): Image array (1 or 3 channels)

    #     Returns:
    #         str: Base64 encoded image string
    #     """
    #     # Convert single channel to 3 channels if needed
    #     if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
    #         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    #     # Ensure uint8 type
    #     if image.dtype != np.uint8:
    #         image = (image * 255).astype(np.uint8)

    #     # Convert to PIL Image
    #     pil_image = Image.fromarray(image)

    #     # Convert to base64
    #     buffered = io.BytesIO()
    #     pil_image.save(buffered, format="JPEG")
    #     img_str = base64.b64encode(buffered.getvalue()).decode()

    #     return img_str
