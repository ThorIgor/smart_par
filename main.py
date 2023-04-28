from tqdm import tqdm
import logging

import numpy as np
import pandas as pd

import typer

import torch
from transformers import GPT2Tokenizer, T5ForConditionalGeneration 
#from sentence_transformers import SentenceTransformer


import spacy

from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def rev_sigmoid(x:float)->float:
  return (1 / (1 + np.exp(0.5*x)))

def activate_similarities(similarities:np.array, p_size=10)->np.array:
  """ Function returns list of weighted sums of activated sentence similarities
  Args:
      similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
      p_size (int): number of sentences are used to calculate weighted sum 
  Returns:
      list: list of weighted sums
  """
  # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
  x = np.linspace(-10,10,p_size)
  # Then we need to apply activation function to the created space
  y = np.vectorize(rev_sigmoid) 
  # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
  activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
  ### 1. Take each diagonal to the right of the main diagonal
  diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
  ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
  diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
  ### 3. Stack those diagonals into new matrix
  diagonals = np.stack(diagonals)
  ### 4. Apply activation weights to each row. Multiply similarities with our activation.
  diagonals = diagonals * activation_weights.reshape(-1,1)
  ### 5. Calculate the weighted sum of activated similarities
  activated_similarities = np.sum(diagonals, axis=0)
  return activated_similarities

def paragraphizise(input_path, output_path):

    tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/FRED-T5-1.7B',eos_token='</s>')

    model = T5ForConditionalGeneration.from_pretrained('ai-forever/FRED-T5-1.7B')
    model.to('cuda')

    #model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    docs = pd.read_csv(input_path)

    nlp = spacy.blank("ru")
    nlp.add_pipe('sentencizer')

    new_docs = []
    for doc in tqdm(docs['content']):
      par_df = []

      sents = [sent.text.replace("\n", "") for sent in nlp(doc).sents]
      toks = tokenizer(sents, return_tensors="np")
      embeddings = model.encode(**toks)
      #embeddings = model.encode(sents)
      sims = cosine_similarity(embeddings)

      act_sims = activate_similarities(sims, p_size = np.min([len(sents), 10]))
      loc_min_i = argrelextrema(act_sims, np.less, order = 1)[0]

      last_i = 0
      for i in loc_min_i:
        par_df.append(" ".join(sents[last_i:i]))
        last_i = i
      par_df.append(" ".join(sents[last_i:len(sents)]))

      new_docs.append("\n\n".join(par_df))

    new_docs = pd.DataFrame(new_docs, columns = ["Par doc"])
    new_docs["Raw doc"] = docs['content']
    new_docs.to_csv(output_path, index = False)

if __name__ == "__main__":
  typer.run(paragraphizise)