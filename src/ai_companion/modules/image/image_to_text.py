import os
import base64
from groq import Groq
import logging
from typing import Optional, Union

from ai_companion.core.exceptions import ImageToTextError
from settings import settings

class ImageToText:
    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]

    def __init__(self):
        self._validate_env_vars()
        self._client: Optional[Groq] = None
        self.logger = logging.getLogger(__name__)

    def _validate_env_vars(self):
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing env variables: {', '.join(missing_vars)}")
        
    @property
    def client(self):
        if self._client is None:
            self._client = Groq(api_key=settings.GROQ_API_KEY)
        return self._client
        
    def analyze_image(self, image_data: Union[str, bytes], prompt:str = "") -> str:
        try:
            if isinstance(image_data, str):
                if not os.path.exists(image_data):
                    raise ValueError(f"Image file path does not exist at {image_data}")
                else:
                    with open(image_data, "rb") as img_file:
                        image_bytes = img_file.read()

            elif isinstance(image_data, bytes):
                image_bytes = image_data

            if not image_bytes:
                raise ValueError("Image data cannot be empty")
            
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            if not prompt:
                prompt = "Describe the content of the image in detail."

            messages = [
                {"role":"user", "content": [
                    {"type":"text", "text": prompt},
                    {"type":"image_base64", "image_base64": image_base64}
                ]}
            ]

            response = self.client.chat.completions.create(
                model=settings.ITT_MODEL_NAME,
                messages=messages,
                max_tokens=1000,
            )

            if not response.choices or not response.choices[0].message.content:
                raise ImageToTextError("No content returned from image analysis")
            
            description = response.choices[0].message.content
            self.logger.info(f"Generated image description: {description}")

            return description
        
        except Exception as e:
            raise ImageToTextError(f"Image to text conversion failed: {str(e)}") from e