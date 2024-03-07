import streamlit as st
import torch
import torch.nn as nn
from model.model import LSTMConcatAttentionEmbed
from src.rnn_preprocessing import preprocess_single_string
import json

# Загрузка модели
model_concat_embed = LSTMConcatAttentionEmbed()
model_concat_embed.load_state_dict(torch.load('model/model_weights.pt'))

# Загрузка словарей
with open('model/vocab.json', 'r') as f:
    vocab_to_int = json.load(f)
    
with open('model/int_vocab.json', 'r') as f:
    int_to_vocab = json.load(f)


# Функция предсказания
def plot_and_predict(review: str, SEQ_LEN: int, model: nn.Module):

    inp = preprocess_single_string(review, SEQ_LEN, vocab_to_int)
    model.eval()
    with torch.inference_mode():
        pred, att_scores = model(inp.long().unsqueeze(0))
    pred = pred.sigmoid().item()
    #att_scores = att_scores[0].squeeze().detach().cpu().numpy()
    #plt.figure(figsize=(4, 8))
    st.write(f'Prediction {pred:.3f}')
    st.write(f'Negative' if pred < 0.75 else 'positive')
    #plt.barh(np.arange(len(inp)), att_scores)
    #plt.yticks(
        #ticks = np.arange(len(inp)),
        #labels = [int_to_vocab[x.item()] for x in inp]
        #);


st.title('Классификация отзыва')
st.header('введите текст')

title = st.text_input('Введите отзыв для его классификаций', 'Хорошее место')

plot_and_predict(
    review = title, 
    SEQ_LEN=10, 
    model = model_concat_embed
)