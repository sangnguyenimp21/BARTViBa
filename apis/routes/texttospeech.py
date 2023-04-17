from GraphTranslation.apis.routes.base_route import BaseRoute

from objects.data import DataSpeech, OutDataSpeech
from TTS.main import generator, dct, generator_fm, dct_fm, hifigan, infer, output_sampling_rate
import io
from scipy.io.wavfile import write
import base64
import torch

import threading
import concurrent.futures

MAX_THREADS = 4

class SpeakRoute(BaseRoute):
    def __init__(self):
        super(SpeakRoute, self).__init__(prefix="/speak")

    def partition_input(self, input_text):
        """we are only allowed to use up to 4 threads"""
        split_input = input_text.split()
        len_of_input = len(split_input)
        chunk_size = len_of_input // 4
        partitioned = []
        if chunk_size > 0:
            for i in range(MAX_THREADS):
                start = i * chunk_size
                end = (i+1) * chunk_size
                if i == MAX_THREADS - 1:
                    end = len_of_input
                partitioned.append(' '.join(split_input[start:end]))
        else:
            partitioned = split_input
        return partitioned

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

        # generate_wav_file should take a wav file as argument
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
            print(self.partition_input(data.text))
            return await self.wait(self.translate_func, data)
