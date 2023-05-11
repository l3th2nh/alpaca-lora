import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import sys
from typing import List
 
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
 
import fire
import torch
from datasets import load_dataset
import pandas as pd
 
import matplotlib.pyplot as plt
import matplotlib as mpl

from pylab import rcParams
 
import logging

 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


df = pd.read_csv("bitcoin-sentiment-tweets.csv")
df.head()


def sentiment_score_to_name(score: float):
    logging.debug('This is a debug message - '+score)
    score = float(score)
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    return "Neutral"
 
dataset_data = [
    {
        "instruction": "Detect the sentiment of the tweet.",
        "input": row_dict["Tweet"],
        "output": sentiment_score_to_name(row_dict["sent_score"])
    }
    for row_dict in df.to_dict(orient="records")
]

import json
with open("alpaca-bitcoin-sentiment-dataset.json", "w") as f:
   json.dump(dataset_data, f)

