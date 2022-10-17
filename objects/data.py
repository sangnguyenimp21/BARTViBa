from pydantic import BaseModel
from typing import Optional


class Data(BaseModel):
    text: str
    model: Optional[str]

    def __init__(self, text: str, model: str = None):
        super(Data, self).__init__(text=text, model=model)
        self.text = text
        self.model = model


class OutData(BaseModel):
    src: str
    tgt: str

    def __init__(self, src: str, tgt: str = None):
        super(OutData, self).__init__(src=src, tgt=tgt)
        self.src = src
        self.tgt = tgt
