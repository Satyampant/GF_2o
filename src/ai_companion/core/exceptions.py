from exceptions import Exception

class SpeechToTextError(Exception):
    """Exception raised for errors in the speech-to-text conversion process."""
    pass

class TextToSpeechError(Exception):
    """Exception raised for errors in the text-to-speech conversion process."""
    pass

class TextToImageError(Exception):
    """Exception raised for errors in the text-to-image generation process."""
    pass

class ImageToTextError(Exception):
    """Exception raised for errors in the image-to-text conversion process."""
    pass

