import os
import base64
from groq import Groq
import logging
from typing import Optional, Union

from ai_companion.core.exceptions import TextToImageError
from ai_companion.core.prompts import IMAGE_ENHANCEMENT_PROMPT, IMAGE_SCENARIO_PROMPT
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from settings import settings
from together import Together

class ScenarioPrompt(BaseModel):
    narrative: str = Field(..., description="The AI's narrative response to the question")
    image_prompt: str = Field(..., description="The visual prompt to generate an image representing the scene")

class EnhancedPrompt(BaseModel):
    content: str = Field(..., description="The enhanced and detailed image generation prompt")


class TextToImage:
    REQUIRED_ENV_VARS = ["GROQ_API_KEY", "TOGETHER_API_KEY"]

    def __init__(self):
        self._validate_env_vars()
        self._together_client: Optional[Together] = None
        self.logger = logging.getLogger(__name__)

    def _validate_env_vars(self):
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing env variables: {', '.join(missing_vars)}")
        
    @property
    def client(self):
        if self._together_client is None:
            self._together_client = Groq(api_key=settings.GROQ_API_KEY)
        return self._together_client
    
    async def generate_image(self, prompt:str, output_path:str)->bytes:
        if not prompt.strip():
            raise ValueError("Input prompt cannot be empty")
        
        try:
            self.logger.info(f"Generating image for prompt: {prompt}")
            response = self.together_client.images.generate(
                model=settings.TTI_MODEL_NAME,
                prompt=prompt,
                width=1024,
                height=768,
                steps=4,
                n=1,
                response_format="b64_json",
            )

            image_data = base64.b64decode(response['data'][0].b64_json)

            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as img_file:
                    img_file.write(image_data)
                self.logger.info(f"Image saved to {output_path}")
            return image_data
        except Exception as e:
            raise TextToImageError(f"Text-to-image generation failed: {str(e)}") from e
        
    async def create_scenario(self, chat_history: list= None) -> ScenarioPrompt:
        try:
            formatted_history = "\n".join([f"{msg.type.title()}:{msg.content}" for msg in chat_history])
            self.logger.info(f"Creating scenario with chat history")

            llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model=settings.TEXT_MODEL_NAME,
                temperature=0.4,
                max_retries=2
                )
            
            structured_llm = llm.with_structured_output(ScenarioPrompt)

            chain = (
                    PromptTemplate(
                        input_variables=["chat_history"],
                        template=IMAGE_SCENARIO_PROMPT
                    )
                | structured_llm
            )

            scenario = chain.invoke({"chat_history": formatted_history})
            self.logger.info(f"Created scenario: {scenario}")

            return scenario
        
        except Exception as e:
            raise TextToImageError(f"Scenario creation failed: {str(e)}") from e
        
    async def enhance_prompt(self, base_prompt:str) -> EnhancedPrompt:
        try:
            self.logger.info(f"Enhancing prompt: {base_prompt}")

            llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model=settings.TEXT_MODEL_NAME,
                temperature=0.4,
                max_retries=2
                )
            
            structured_llm = llm.with_structured_output(EnhancedPrompt)

            chain = (
                    PromptTemplate(
                        input_variables=["base_prompt"],
                        template=IMAGE_ENHANCEMENT_PROMPT
                    )
                | structured_llm
            )

            enhanced_prompt = chain.invoke({"base_prompt": base_prompt})
            self.logger.info(f"Enhanced prompt: {enhanced_prompt}")

            return enhanced_prompt
        
        except Exception as e:
            raise TextToImageError(f"Prompt enhancement failed: {str(e)}") from e