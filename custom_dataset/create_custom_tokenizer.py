import os
from tqdm import tqdm

from transformers import AutoTokenizer, MBartModel, BartTokenizer, BartphoTokenizer


DATA_FOLDER = "data"
ba_files = [item for item in os.listdir(DATA_FOLDER) if ".ba" in item]
ba_files = [os.path.join(DATA_FOLDER, item) for item in ba_files]

ba_tokens = set()
l = None
for ba_file in ba_files:
    data = open(ba_file, "r", encoding="utf8").readlines()
    data = [item.replace("\n", "").strip() for item in data]
    for line in tqdm(data):
        if l is None:
            l = line
        words = line.split(" ")
        for word in words:
            ba_tokens.add(word)

syllable_tokenizer = BartphoTokenizer.from_pretrained("pretrained/bartpho_syllable")
print(syllable_tokenizer.get_vocab()["<unk>"])
# syllable_tokenizer.add_tokens(list(ba_tokens), special_tokens=True)
# syllable_tokenizer.save_pretrained("pretrained/bart_vi_ba")

print(syllable_tokenizer(l))
