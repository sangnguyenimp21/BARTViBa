from GraphTranslation.apis.routes.base_route import BaseRoute

from objects.data import DataSpeech, OutDataSpeech
from TTS.main import generator, dct, generator_fm, dct_fm, hifigan, infer, output_sampling_rate
import io
from scipy.io.wavfile import write
import base64
import torch
import numpy as np

import threading
import nltk
import queue


MAX_THREADS = 4

threads_dict = [[] for _ in range(MAX_THREADS)]

class CustomThread(threading.Thread):
    def __init__(self, target=None, args=(), kwargs=None, priority=0):
        super().__init__(target=target, args=args, kwargs=kwargs)
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority
    
class SpeakRoute(BaseRoute):
    def __init__(self):
        super(SpeakRoute, self).__init__(prefix="/speak")

    def join_threads_by_priority(self, threads):
        # Create a priority queue to store the threads
        q = queue.PriorityQueue()

        # Add the threads to the priority queue
        for thread in threads:
            q.put((thread.priority, thread))

        # Join the threads in order of their priorities
        while not q.empty():
            priority, thread = q.get()

            print("Thread " + thread.name + " finished")
            thread.join()

    def take_thread_value(self, threads_dict):
        # Concatenate the audio files from each thread
        concatenated_values = []
        for value in threads_dict:
            print(f'Len: {len(value)}')
            concatenated_values.extend(value)

        with open('output.txt', 'w') as f:
            for value in concatenated_values:
                f.write(f'{value}\n')

        # Create the OutDataSpeech object for the final audio file
        out_data = OutDataSpeech(speech=concatenated_values)

        return out_data

    # improvement: use a thread pool to process the input text
    # partition the input text into 4 chunks and process them in parallel
    def partition_input(self, input_text):
        # Split the input text into sentences using the nltk library
        nltk.download('punkt')
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

    def wrapper_function(self, data: DataSpeech, generator, dct, threads_dict):
        # Dictionary containing generator and dct tuples for each gender
        inputs = self.partition_input(data.text)
        # print(input)
        threads = []
        prio = 0

        for index, input in enumerate(inputs):
            # Create a thread for each input
            thread = CustomThread(target=self.translate_func, args=(
                data, input, generator, dct, threads_dict, index),
                priority=prio
            )
            
            threads.append(thread)
            prio += 1

            # Add the thread to the dictionary

            # 1 - Thread <OutDataSpeech> <Thread1> ---> [A,B]
            # 2 - Thread <OutDataSpeech> <Thread2> ---> [A,B]
            # 3 - Thread <OutDataSpeech> <Thread3> ---> [A,B]
            # 4 - Thread <OutDataSpeech> <Thread4> ---> [A,B]
            # want to know thread's audio file

        # wait for all threads to finish
        # for thread in threads_dict.items():
        #     print(thread)
        #     thread.join()
        # Start all threads
        for thread in threads:
            thread.name = thread.name + " " + str(thread.priority)
            thread.start()

        # Wait for all threads to finish
        # for thread in threads:
        #     # inform finished threads
        #     print("Thread " + thread.name + " finished")
        #     thread.join()

        self.join_threads_by_priority(threads)

        # return the output of the threads
        res = self.take_thread_value(threads_dict)
        return res

    # need to create all 4 threads and wait for them to finish

    def translate_func(self, data: DataSpeech, input, generator, dct, threads_dict, index):
        input_text = input
        print(input_text)

        for i, input in enumerate(input_text):
            # rint(input)
            if data.gender:
                gender = data.gender
            else:
                gender = "both"

            # generate_wav_file should take a wav file as argument
            # process input_text into 4 chunks (multithreading)
            if gender == "male":
                y = infer(input, generator, dct)
            elif gender == "female":
                y = infer(input, generator_fm, dct_fm)
            else:
                y = infer(input, generator, dct)
                y_fm = infer(input, generator_fm, dct_fm)

            audio_data = self.make_audio(y)

            if gender == "both":
                audio_data_fm = self.make_audio(y_fm)
                if i == 0:
                    threads_dict[index] = [
                        OutDataSpeech(speech=audio_data, speech_fm=audio_data_fm)
                    ]
                else:
                    threads_dict[index].append(OutDataSpeech(
                        speech=audio_data, speech_fm=audio_data_fm)
                    )
            else:
                if i == 0:
                    threads_dict[index] = [OutDataSpeech(speech=audio_data)]
                else:
                    threads_dict[index].append(OutDataSpeech(speech=audio_data))

    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def translate(data: DataSpeech):
            # self.partition_input(data.text)
            return await self.wait(self.wrapper_function, data, generator, dct, threads_dict)
