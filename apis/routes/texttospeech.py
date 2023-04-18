from GraphTranslation.apis.routes.base_route import BaseRoute

from objects.data import DataSpeech, OutDataSpeech
from TTS.main import generator, dct, generator_fm, dct_fm, hifigan, infer, output_sampling_rate
import io
from scipy.io.wavfile import write
import base64
import torch

import threading
import concurrent.futures
import nltk

MAX_THREADS = 4


class SpeakRoute(BaseRoute):
    def __init__(self):
        super(SpeakRoute, self).__init__(prefix="/speak")

    # improvement: use a thread pool to process the input text
    # partition the input text into 4 chunks and process them in parallel
    def partition_input(self, input_text):
        # Split the input text into sentences using the nltk library
        # nltk.download('punkt')
        sentences = nltk.sent_tokenize(input_text)
        num_sentences = len(sentences)
        print(num_sentences)

        # Determine the chunk size based on the number of sentences
        chunk_size = num_sentences // 4
        remainder = num_sentences % 4

        # Initialize the partitioned array
        partitioned = []

        # Divide the sentences into 4 tuples and append them to the partitioned array
        start = 0
        for i in range(4):
            end = start + chunk_size
            if i < remainder:
                end += 1
            partitioned.append(tuple(sentences[start:end]))
            start = end

        # print(partitioned)
        return partitioned

    def make_audio(self, y):
        with torch.no_grad():
            audio = hifigan.forward(
                y).cpu().squeeze().clamp(-1, 1).detach().numpy()
        audio = audio * 4
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        write(byte_io, output_sampling_rate, audio)
        wav_bytes = byte_io.read()

        audio_data = base64.b64encode(wav_bytes).decode('UTF-8')
        return audio_data

    def translate_func(self, data: DataSpeech):
        processed_text = self.partition_input(data.text)
        print(processed_text)

        if data.gender:
            gender = data.gender
        else:
            gender = "both"

        # process input_text into 4 chunks (multithreading)
        output_data = self.process_audio(processed_text, gender)

        return output_data

    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def translate(data: DataSpeech):
            # self.partition_input(data.text)
            return await self.wait(self.translate_func, data)

    def process_audio(self, processed_text, gender):
        threads_m = []
        threads_fm = []

        results = []

        if gender == "male":
            

        elif gender == "female":
            
        else:
            

        if gender == "both":
            
        else:
            