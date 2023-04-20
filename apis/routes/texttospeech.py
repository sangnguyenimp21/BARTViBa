from GraphTranslation.apis.routes.base_route import BaseRoute

from objects.data import DataSpeech, OutDataSpeech
from TTS.main import generator, dct, generator_fm, dct_fm, hifigan, infer, output_sampling_rate
import io
from scipy.io.wavfile import write
import base64
import torch
import threading
import nltk
import queue
from pydub import AudioSegment
import datetime
import math

MAX_THREADS = 4

SPEECH_DATA = dict()

# SERVER_URL = "https://bahnar.dscilab.site:20007"
SERVER_URL = "http://localhost:8080"
USER = "/Users/khangnguyen/lab"

def current_datetime():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

class CustomThread(threading.Thread):
    def __init__(self, target=None, args=(), kwargs=None, priority=0):
        super().__init__(target=target, args=args, kwargs=kwargs)
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority

class SpeakRoute(BaseRoute):
    FILE_NUMBER = 0
    MP3_SIGNATURE = current_datetime()
    def __init__(self):
        super(SpeakRoute, self).__init__(prefix="/speak")

    def join_threads_by_priority(self, threads):
        # Create a priority queue to store the threads
        q = queue.PriorityQueue()

        # Add the threads to the priority queue
        for thread in threads:
            q.put((thread.priority, thread))

        urls = list()
        # Join the threads in order of their priorities
        while not q.empty():
            priority, thread = q.get()

            # print("Thread " + thread.name + " finished")
            thread.join()
            # return the output of the threads
            self.take_thread_value(SPEECH_DATA, priority)

    def take_thread_value(self, SPEECH_DATA, priority):
        print("take thread value's prio is ", priority)
        self.decode_audio(SPEECH_DATA[priority].speech, self.FILE_NUMBER)

    # improvement: use a thread pool to process the input text
    # partition the input text into 4 chunks and process them in parallel
    def partition_input(self, input_text):
        """
        :param input_text:
        :return: list of 4-element list
        """
        # Split the input text into sentences using the nltk library
        nltk.download('punkt')
        sentences = nltk.sent_tokenize(input_text)
        print("sentences: ", sentences)
        num_sentences = len(sentences)

        # Determine the chunk size based on the number of sentences
        num_jobs = math.ceil( num_sentences / MAX_THREADS )
        print("num_jobs: ", num_jobs)
        jobs = []
        for i in range(num_jobs):
            start = i*4
            end = (i+1) * 4
            jobs.append(tuple(sentences[start:end]))
        return jobs

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

    def decode_audio(self, audio_data, file_num):
        # Decode the base64-encoded string to bytes
        wav_bytes = base64.b64decode(audio_data.encode('utf-8'))

        wav_audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        mp3_audio = io.BytesIO()
        wav_audio.export(mp3_audio, format="mp3")

        mp3_bytes = mp3_audio.getvalue()

        with open(f"{USER}/BARTViBa/to-speech/{self.MP3_SIGNATURE}+{file_num}.mp3", "ab") as f:
            f.write(mp3_bytes)

    def translate_func(self, data: DataSpeech, input, generator, dct, SPEECH_DATA, index):
        input_text = input
        print("it: ", input_text)

        if data.gender:
            gender = data.gender
        else:
            gender = "both"

        # generate_wav_file should take a wav file as argument
        # process input_text into 4 chunks (multithreading)
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
            SPEECH_DATA[index] = OutDataSpeech(speech=audio_data, speech_fm=audio_data_fm)
            return

        SPEECH_DATA[index] = OutDataSpeech(speech=audio_data)


    # need to create all 4 threads and wait for them to finish
    def wrapper_function(self, data: DataSpeech, generator, dct, SPEECH_DATA):
        # Dictionary containing generator and dct tuples for each gender
        inputs = self.partition_input(data.text)
        print("wf: ", inputs)
        num_jobs = len(inputs)
        urls = list()
        for i in range(num_jobs):
            mp3_filename = f"{self.MP3_SIGNATURE}+{i}.mp3"
            mp3_url = f"{SERVER_URL}/to-speech/{mp3_filename}"
            urls.append(mp3_url)
        print(urls) # asynch this shiz
        threads = []

        for index, job in enumerate(inputs):
            """
             index: file number
             job: batch of 4 sentences
            """
            for prio, input in enumerate(job):
                print("prio ", prio, "input ", input)
                # Create a thread for each input
                thread = CustomThread(target=self.translate_func, args=(
                    data, input, generator, dct, SPEECH_DATA, prio), priority=prio)
                threads.append(thread)
            # Start all threads
            for thread in threads:
                thread.name = thread.name + " " + str(thread.priority)
                thread.start()

            self.FILE_NUMBER = index
            self.join_threads_by_priority(threads)
            threads = []


        self.MP3_SIGNATURE = current_datetime()
        return urls

    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def translate(data: DataSpeech):
            return await self.wait(self.wrapper_function, data, generator, dct, SPEECH_DATA)
