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

MAX_THREADS = 4

SPEECH_DATA = dict()




class CustomThread(threading.Thread):
    def __init__(self, target=None, args=(), kwargs=None, priority=0):
        super().__init__(target=target, args=args, kwargs=kwargs)
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority

class SpeakRoute(BaseRoute):
    MP3_SIGNATURE = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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

            print("Thread " + thread.name + " finished")
            thread.join()
            # return the output of the threads
            url = self.take_thread_value(SPEECH_DATA, priority)
            urls.append(url)
        return urls

    def take_thread_value(self, SPEECH_DATA, priority):

        # TEMPORARY:
        if not priority in SPEECH_DATA:
            SPEECH_DATA[priority] = []
        # TEMPORARY:

        for out_speech in SPEECH_DATA[priority]:
            self.decode_audio(out_speech.speech, priority)
            priority += 1
        # Construct and return the URL of the generated mp3 file
        mp3_filename = f"{self.MP3_SIGNATURE}+{priority}.mp3"
        mp3_url = f"http://localhost:8080/to-speech/{mp3_filename}"
        return mp3_url

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

    def decode_audio(self, audio_data, priority):
        # Decode the base64-encoded string to bytes
        wav_bytes = base64.b64decode(audio_data.encode('utf-8'))

        wav_audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        mp3_audio = io.BytesIO()
        wav_audio.export(mp3_audio, format="mp3")

        mp3_bytes = mp3_audio.getvalue()

        with open(f"/Users/khangnguyen/to-speech/{self.MP3_SIGNATURE}+{priority}.mp3", "wb") as f:
            f.write(mp3_bytes)

    def translate_func(self, data: DataSpeech, input, generator, dct, SPEECH_DATA, index):
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
                    SPEECH_DATA[index] = [
                        OutDataSpeech(speech=audio_data, speech_fm=audio_data_fm)
                    ]
                else:
                    SPEECH_DATA[index].append(
                        OutDataSpeech(speech=audio_data, speech_fm=audio_data_fm)
                    )
            else:
                if i == 0:
                    SPEECH_DATA[index] = [OutDataSpeech(speech=audio_data)]
                else:
                    SPEECH_DATA[index].append(OutDataSpeech(speech=audio_data))



    # need to create all 4 threads and wait for them to finish
    def wrapper_function(self, data: DataSpeech, generator, dct, SPEECH_DATA):
        # Dictionary containing generator and dct tuples for each gender
        inputs = self.partition_input(data.text)
        # print(input)
        threads = []
        prio = 0

        for index, input in enumerate(inputs):
            # Create a thread for each input
            thread = CustomThread(target=self.translate_func, args=(
                data, input, generator, dct, SPEECH_DATA, index),
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

        # Start all threads
        for thread in threads:
            thread.name = thread.name + " " + str(thread.priority)
            thread.start()

        urls = self.join_threads_by_priority(threads)
        return urls

    def create_routes(self):
        router = self.router

        @router.post("/vi_ba")
        async def translate(data: DataSpeech):
            return await self.wait(self.wrapper_function, data, generator, dct, SPEECH_DATA)
