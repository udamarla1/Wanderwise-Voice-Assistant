"""Audio"""

import os
import wave
from io import BytesIO

import numpy as np
import torch

import speech_recognition as sr
import pyttsx3
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from transformers import pipeline
from st_audiorec import st_audiorec # does not have audio processing

# Add libs to PATH for pydub/ffmpeg
libs_path = os.path.join(os.getcwd(), "libs")
if libs_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + libs_path

# AudioSegment.converter = os.path.join(libs_path, "ffmpeg.exe") 
# AudioSegment.ffmpeg    = os.path.join(libs_path, "ffmpeg.exe")
# AudioSegment.ffprobe   = os.path.join(libs_path, "ffprobe.exe")


class Audio:
    """Audio Class"""

    def __init__(self) -> None:
        """Initialize speech recognition object"""
        self.recognizer = sr.Recognizer()
        self.microphone = None

        # Disable mic by default
        self.mic_enabled = False

    def initialize_microphone(self, device_index):
        """Initialize microphone object with appropriate device

        device_index: int indicating the index of the microphone
        """
        self.microphone = sr.Microphone(device_index)
        self.mic_enabled = True

    def communicate(self, phrase="You forgot to pass the text"):
        """Audio approach that saves to a file and then plays it.
        Could be sped up by doing a sentence at a time.

        phrase: the string to convert to speech
        """

        print(f"Bot said: {phrase}")
        
        # Generate unique filenames to avoid permission errors if multiple threads run
        import uuid
        unique_id = str(uuid.uuid4())
        temp_file = f"temp_speech_{unique_id}.mp3"
        temp_wav = f"temp_speech_{unique_id}.wav"
        
        # 1. Convert text to audio data
        print("Generating audio with gTTS...")
        try:
            tts = gTTS(text=phrase, lang='en')
            
            # 2. Save it to a temporary file
            print(f"Saving to {temp_file}...")
            tts.save(temp_file)
            
            # 3. Load the file
            print("Loading audio file with pydub...")
            talk = AudioSegment.from_mp3(temp_file)
            
            # 4. Play it!
            print("Playing audio with winsound...")
            talk.export(temp_wav, format="wav")
            
            import winsound
            winsound.PlaySound(temp_wav, winsound.SND_FILENAME)
            print("Audio finished playing.")
            
        except Exception as e:
            print(f"Error in communicate: {e}")
        finally:
            # Cleanup (Optional but good practice)
            # Remove the temporary files to keep the filesystem clean.
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception as cleanup_error:
                    print(f"Warning: Could not remove temp file {temp_wav}: {cleanup_error}")

    def recognize_speech_from_mic(self):
        """Transcribes speech from a microphone

        Returns a dictionary with the following keys:
            "success": A boolean indicating whether or not the request was successful
            "error":   'None' if successful, otherwise a string containing an error message
            "transcription": A string containing the transcribed text or 'None' if speech was
            unrecognizable
        """

        # Adjust the recognizer sensitivity for ambient noise and listen to the microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)

        # Initialize response object
        response = {"success": True, "error": None, "transcription": None}

        # Try to recognize the speech and handle exceptions accordingly
        try:
            response["transcription"] = self.recognizer.recognize_google(audio)
        except sr.RequestError:
            # API was unreachable or unresponsive
            response["success"] = False
            response["error"] = "API unavailable"
        except sr.UnknownValueError:
            # Speech was unintelligible
            response["success"] = False
            response["error"] = "Unable to recognize speech"

        return response

    def get_streamlit_audio(self):
        """
        Uses streamlit component to get the audio data
        https://github.com/stefanrmmr/streamlit-audio-recorder
        """
        try:
            audio_wave_bytes = st_audiorec()
        except Exception as e:
            raise ValueError("Unable to capture audio from browser") from e
        return self.convert_streamlit_audio_to_gradio_format(audio_wave_bytes)

    def convert_streamlit_audio_to_gradio_format(self, audio_wave_bytes):
        """Takes audio wave bytes and returns it in the format of gradio audio object
        sampling_rate, raw_audio_data = audio
        """
        if not audio_wave_bytes:
            raise ValueError("No audio wave bytes received.")
        with wave.open(BytesIO(audio_wave_bytes), "rb") as wf:
            params = wf.getparams()
            sampling_rate = params.framerate
            num_channels = params.nchannels
            num_frames = params.nframes
            raw_audio_data = np.frombuffer(wf.readframes(num_frames), dtype=np.int16)

            if num_channels > 1:
                raw_audio_data = raw_audio_data.reshape(-1, num_channels)
        return (sampling_rate, raw_audio_data)

    def transcribe_from_transformer(
        self, audio, model_name_and_version="openai/whisper-base.en"
    ):
        """Convert audio data to text using transformers"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transcriber = pipeline(
            task="automatic-speech-recognition",
            model=model_name_and_version,
            device=device,
        )
        try:
            sampling_rate, raw_audio_data = audio
        except TypeError as e:
            raise TypeError("No audio data received. Please speak louder.") from e

        # Convert to mono if stereo
        if raw_audio_data.ndim > 1:
            raw_audio_data = raw_audio_data.mean(axis=1)

        raw_audio_data = raw_audio_data.astype(np.float32)
        raw_audio_data /= np.max(np.abs(raw_audio_data))

        prompt = transcriber({"sampling_rate": sampling_rate, "raw": raw_audio_data})[
            "text"
        ]
        return prompt

    def get_prompt_from_gradio_audio(self, audio):
        """
        Converts audio captured from gradio to text.
        See https://www.gradio.app/guides/real-time-speech-recognition for more info.
        audio: object containing sampling frequency and raw audio data

        """
        print("Initializing Whisper model pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base.en",
            device=device,
        )
        print("Whisper pipeline ready. Transcribing...")
        try:
            sampling_rate, raw_audio_data = audio
        except TypeError as e:
            raise TypeError("No audio data received. Please speak louder.") from e

        # Convert to mono if stereo
        if raw_audio_data.ndim > 1:
            raw_audio_data = raw_audio_data.mean(axis=1)

        raw_audio_data = raw_audio_data.astype(np.float32)
        raw_audio_data /= np.max(np.abs(raw_audio_data))

        prompt = transcriber({"sampling_rate": sampling_rate, "raw": raw_audio_data})[
            "text"
        ]
        return prompt

    def get_prompt_from_file(self, file):
        """Get Prompt from audio file"""
        try:
            speech = sr.AudioFile(file)
        except Exception as e:
            raise IOError(f"Unable to read the audio file: {e}") from e
        with speech as source:
            speech = self.recognizer.record(source)
        text = self.recognizer.recognize_google(speech)
        return text


if __name__ == "__main__":
    recognized_mics = {}
    test_audio = Audio()
    for i, mic in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{i}: {mic}")
        recognized_mics.update({mic: i})
    built_in_idx = recognized_mics['Built-in Microphone']
    print(recognized_mics)
    test_audio.initialize_microphone(built_in_idx)
    test_audio.communicate("Hello class.")
    print(test_audio.recognize_speech_from_mic())
