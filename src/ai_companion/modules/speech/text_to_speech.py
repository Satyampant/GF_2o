import os
from typing import Optional
from groq import Groq
from settings import settings
from ai_companion.core.exceptions import TextToSpeechError
from elevenlabs import ElevenLabs, Voice, VoiceSettings

class TextToSpeech:
    REQUIRED_ENV_VARS = ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]

    def __init__(self):
        self._validate_env_vars()
        self._client: Optional[ElevenLabs] = None 

    def _validate_env_vars(self):
        missing_vars = [var for var in vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing env variables: {''.join(missing_vars)}")
        
    @property
    def client(self):
        if self._client is None:
            self._client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        return self._client
    
    def synthesize(self, text:str) -> bytes:
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        if len(text)>5000:
            raise ValueError("Input text exceeds maximum length of 5000 characters")
        
        try:
            audio_generator = self.client.text_to_speech.convert(
                voice_id= settings.ELEVENLABS_VOICE_ID,
                text = text,
                model_id= settings.TTS_MODEL_NAME,
                voice_settings = VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.5
                )
            )

            # Convert generator to bytes
            audio_bytes = b"".join(audio_generator)
            if not audio_bytes:
                raise TextToSpeechError("Generated audio is empty")

            return audio_bytes

        except Exception as e:
            raise TextToSpeechError(f"Text-to-speech conversion failed: {str(e)}") from e
