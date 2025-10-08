!pip install torch

import streamlit as st
import torch
import sentencepiece as spm
import torch.nn as nn
import re

# --- CONFIG (match training) ---
EXP_NAME = 'exp_emb256_hid512_lr5e-4_att'
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
ENC_N_LAYERS = 2
DEC_N_LAYERS = 4
DROPOUT = 0.3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = f'final_model_{EXP_NAME}.pth'
SP_UR_MODEL = 'sp_model_ur.model'
SP_EN_MODEL = 'sp_model_en.model'

# --- MODEL CLASSES (must match training script exactly) ---
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = hidden.view(self.n_layers, 2, hidden.size(1), hidden.size(2))
        cell = cell.view(self.n_layers, 2, cell.size(1), cell.size(2))
        combined_hidden = torch.cat((hidden[-1, 0, :, :], hidden[-1, 1, :, :]), dim=1)
        combined_cell = torch.cat((cell[-1, 0, :, :], cell[-1, 1, :, :]), dim=1)
        return outputs, combined_hidden, combined_cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim + (enc_hid_dim * 2), dec_hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_transform = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.cell_transform = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        if hidden.dim() == 2:
            transformed_hidden = torch.tanh(self.hidden_transform(hidden)).unsqueeze(0).repeat(self.n_layers, 1, 1)
            transformed_cell = torch.tanh(self.cell_transform(cell)).unsqueeze(0).repeat(self.n_layers, 1, 1)
        else:
            transformed_hidden, transformed_cell = hidden, cell
        a = self.attention(transformed_hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (transformed_hidden, transformed_cell))
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        enc_outputs, hidden, cell = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, enc_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

# --- TRANSLATION FUNCTION ---
def translate_sentence(sentence, model, sp_source, sp_target, device, max_len=50):
    model.eval()
    tokens = sp_source.encode(sentence, out_type=int)
    tokens = [sp_source.bos_id()] + tokens + [sp_source.eos_id()]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        enc_outputs, hidden, cell = model.encoder(src_tensor)
    trg_indexes = [sp_target.bos_id()]
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        output, hidden, cell = model.decoder(trg_tensor, hidden, cell, enc_outputs)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == sp_target.eos_id():
            break
    return sp_target.decode(trg_indexes[1:-1])

# --- MAIN APP ---
@st.cache_resource
def load_models():
    sp_ur = spm.SentencePieceProcessor()
    sp_ur.load(SP_UR_MODEL)
    sp_en = spm.SentencePieceProcessor()
    sp_en.load(SP_EN_MODEL)

    attn = Attention(HIDDEN_DIM, HIDDEN_DIM)
    enc = Encoder(sp_ur.get_piece_size(), EMBEDDING_DIM, HIDDEN_DIM, ENC_N_LAYERS, DROPOUT)
    dec = Decoder(sp_en.get_piece_size(), EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM, DEC_N_LAYERS, DROPOUT, attn)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, sp_ur, sp_en

def main():
    st.title("Urdu to Roman Urdu Transliterator")
    st.write("Enter Urdu text to get Roman Urdu transliteration.")
    model, sp_ur, sp_en = load_models()
    user_input = st.text_input("Urdu Text:")
    if st.button("Translate"):
        if user_input.strip():
            with st.spinner('Translating...'):
                try:
                    translation = translate_sentence(user_input.strip(), model, sp_ur, sp_en, DEVICE)
                    st.success(f"Roman Urdu: {translation}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter some Urdu text.")

if __name__ == "__main__":

    main()

