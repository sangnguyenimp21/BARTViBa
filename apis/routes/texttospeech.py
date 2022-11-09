from GraphTranslation.apis.routes.base_route import BaseRoute

from objects.data import DataSpeech, OutDataSpeech
from TTS.main import generator, dct, generator_fm, dct_fm, hifigan, infer, output_sampling_rate
import io
from scipy.io.wavfile import write
import base64
import torch

class SpeakRoute(BaseRoute):
    def __init__(self):
        super(SpeakRoute, self).__init__(prefix="/speak")
        
    def make_audio(self, y):
        with torch.no_grad():
            audio = hifigan.forward(y).cpu().squeeze().clamp(-1, 1).detach().numpy()

        audio = audio * 4
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        write(byte_io, output_sampling_rate, audio)
        wav_bytes = byte_io.read()

        audio_data = base64.b64encode(wav_bytes).decode('UTF-8')
        return audio_data

    def translate_func(self, data: DataSpeech):
        
        input_text = data.text
        
        if data.gender:
            gender = data.gender
        else:
            gender = "both"

        # generate_wav_file should take a file as parameter and write a wav in it
        if gender == "male":
            y = infer(input_text, generator, dct)
        elif gender == "female":
            y = infer(input_text, generator_fm, dct_fm)
        else:
            y = infer(input_text, generator, dct)
            y_fm = infer(input_text, generator_fm, dct_fm)

        audio_data = self.make_audio(y)

        if gender == "both":
            audio_data_fm = self.make_audio(y_fm)
            return OutDataSpeech(speech=audio_data, speech_fm=audio_data_fm)
    
        return OutDataSpeech(speech=audio_data)

    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def translate(data: DataSpeech):
            return await self.wait(self.translate_func, data)
