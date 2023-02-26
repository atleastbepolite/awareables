import pyaudio
import wave

# Adapted from PyAudio documentation: https://people.csail.mit.edu/hubert/pyaudio/docs/
# and 
# stackoverflow.com: https://stackoverflow.com/questions/6951046/how-to-play-an-audiofile-with-pyaudio

class AudioFile():
    chunk = 1024

    def __init__(self, file):
        """ Init audio stream """ 
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )

    """
    Play function writes data in chunks as audio output until
    file is complete. 
    """
    def play(self):

        data = self.wf.readframes(self.chunk)
        while data != b'':
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    """
    Close function manually closes stream and terminates pyaudio
    """
    def close(self):
        self.stream.close()
        self.p.terminate()
