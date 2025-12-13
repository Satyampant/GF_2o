from ai_companion.settings import settings
from ai_companion.core.exceptions import SpeechToTextError
from groq import Groq
import os
from typing import Optional
import tempfile

class SpeechToText:
    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]

    def __init__(self):
        self._validate_env_vars()
        self._client: Optional[Groq] = None

    def validate_env_vars(self):
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {','.joint(missing_vars)}")
        
    @property
    def client(self) -> Groq:
        if self._client is None:
            self._client = Groq(api_key=settings.GROQ_API_KEY)
        return self._client
    

    def transcribe(self, audio_data: str) -> str:
        if not audio_data:
            raise ValueError("Audio data cannot be empty.")
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            try:
                with open(temp_file_path, "rb") as audio_file:
                    transcription = self.client.audio.transcriptions.create(
                        file = audio_file,
                        model=settings.STT_MODEL_NAME,
                        language="en",
                        response_format="text",
                    )
                if not transcription:
                    raise SpeechToTextError("Transcription result is empty.")
                
                return transcription

            finally:
                os.unlink(temp_file_path)
        
        except Exception as e:
            raise SpeechToTextError(f"Speech to text conversion failed: {str(e)}") from e
