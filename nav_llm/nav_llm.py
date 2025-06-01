import base64
import io
import cv2
from PIL import Image
import numpy as np

from simworld.llm.base_llm import BaseLLM
from agent.action_space import ActionSpace