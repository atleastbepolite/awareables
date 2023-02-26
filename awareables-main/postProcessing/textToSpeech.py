import os
import postProcessing.audioFile as af
# import pyaudio
import google.cloud.texttospeech as tts

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './postProcessing/awareables-17d2d16d8394.json'

class TextToSpeech():
    def __init__(self):
        self.client = tts.TextToSpeechClient()
        self.voice = tts.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Standard-C",
            ssml_gender=tts.SsmlVoiceGender.FEMALE,
        )
        self.audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16
        )

    def unique_languages_from_voices(voices):
        language_set = set()
        for voice in voices:
            for language_code in voice.language_codes:
                language_set.add(language_code)
        return language_set

    """
    Function that lists all possible language choices for the Google API
    """
    def list_languages(self):
        response = self.client.list_voices()
        languages = self.unique_languages_from_voices(response.voices)

        print(f" Languages: {len(languages)} ".center(60, "-"))
        for i, language in enumerate(sorted(languages)):
            print(f"{language:>10}", end="\n" if i % 5 == 4 else "")

    """
    Function that lists all possible voices available for the Google API
    """
    def list_voices(self):
        response = self.client.list_voices()
        for i, voice in enumerate(response.voices):
            print(f"{voice.name:>20}", end="\n" if i % 5 == 4 else "")

    """
    Synthesize_speech generates an audio output directly from the string inputted as text.
    This works by first generating a wav file that is then read as a stream. 
    """
    def synthesize_speech(self, text):
        self.synthesize_text(text)
        # input_text = tts.SynthesisInput(text=text)
        
        # response = self.client.synthesize_speech(
        #     request={"input": input_text, "voice": self.voice, "audio_config": self.audio_config}
        # )
        audio = af.AudioFile("output.wav")
        audio.play()
        audio.close()

    """
    Synthesize_text generates a wav file for a basic audio config that captures the desired
    text. 
    """
    def synthesize_text(self, text):
        # Synthesizes speech from the input string of text.

        input_text = tts.SynthesisInput(text=text)
        
        response = self.client.synthesize_speech(
            request={"input": input_text, "voice": self.voice, "audio_config": self.audio_config}
        )

        # The response's audio_content is binary.
        with open("output.wav", "wb") as out:
            out.write(response.audio_content)
            print('Audio content written to file "output.wav"')

if __name__=='__main__':
    ### Examples
    text = """The goal of the Aware-able product is to aid in the reading 
              and understanding of braille literature for both visually impaired 
              individuals as well as the generally braille-illiterate sighted 
              population. This product will be beneficial in improving overall 
              braille literacy rates, and therefore information symmetry, among 
              all individuals. This product can also be used as a teaching tool 
              for non-visually impaired persons to learn the language."""
    speech = TextToSpeech()
    speech.synthesize_speech(text)
