import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm

import neunet
import neunet.nn as nn
from datasets import load_dataset  # type: ignore
from neunet import Tensor
from neunet.optim import Adam

"""
Seq2Seq Transformer for language translation from English to German
"""

# [Model implementation]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.depth = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias = False)
        self.wk = nn.Linear(d_model, d_model, bias = False)
        self.wv = nn.Linear(d_model, d_model, bias = False)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]=None):
        batch_size = q.shape[0]
        q = self.wq(q).contiguous().reshape(batch_size, -1, self.n_heads, self.depth).transpose(0, 2, 1, 3)
        k = self.wk(k).contiguous().reshape(batch_size, -1, self.n_heads, self.depth).transpose(0, 2, 1, 3)
        v = self.wv(v).contiguous().reshape(batch_size, -1, self.n_heads, self.depth).transpose(0, 2, 1, 3)

        # q = self.wq(q).contiguous().reshape(batch_size, self.n_heads, -1, self.depth)
        # k = self.wk(k).contiguous().reshape(batch_size, self.n_heads, -1, self.depth)
        # v = self.wv(v).contiguous().reshape(batch_size, self.n_heads, -1, self.depth)  

        scores = neunet.matmul(q, k.transpose(0, 1, 3, 2)) / self.scale
        if mask is not None:
            # mask = mask.unsqueeze(1)#.repeat(1, self.n_heads, 1, 1)
            mask = mask[:, None, ...]
            # scores = scores.masked_fill(mask == 0, -1e9)
            scores = neunet.where(mask == 0, -1e9, scores)

        attn = self.dropout(nn.Softmax(axis = -1)(scores))

        x = neunet.matmul(attn, v)
        x = x.contiguous().transpose(0, 2, 1, 3).reshape(batch_size, -1, self.n_heads * self.depth)
        # x = x.contiguous().reshape(batch_size, -1, self.n_heads * self.depth)
        x = self.fc(x)

        return x, attn
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc_2(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps = 0.001)
        self.norm2 = nn.LayerNorm(d_model, eps = 0.001)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(_src)
        src = self.norm1(src)

        # Feed-forward network
        _src = self.ffn(src)
        src = src + self.dropout(_src)
        src = self.norm2(src)

        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps = 0.001)
        self.norm2 = nn.LayerNorm(d_model, eps = 0.001)
        self.norm3 = nn.LayerNorm(d_model, eps = 0.001)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask, src, src_mask):
        # Masked self-attention (for the target sequence)
        _tgt, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(_tgt)
        tgt = self.norm1(tgt)

        # Cross-attention (attending to the encoder output)
        _tgt, attn = self.cross_attn(tgt, src, src, src_mask)
        tgt = tgt + self.dropout(_tgt)
        tgt = self.norm2(tgt)

        # Feed-forward network
        _tgt = self.ffn(tgt)
        tgt = tgt + self.dropout(_tgt)
        tgt = self.norm3(tgt)

        return tgt, attn



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
            https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
        """
        super(PositionalEncoding, self).__init__()
        pe = neunet.zeros(max_len, d_model, requires_grad=False)
        position = neunet.arange(0, max_len, dtype=neunet.float32)[:, None, ...]
        div_term = neunet.exp(neunet.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = neunet.sin(position * div_term)
        pe[:, 1::2] = neunet.cos(position * div_term)
        # print(pe.shape, pe[None, ...].shape,  pe[None, ...].data.shape)
        self.pe = pe[None, ...]

    def forward(self, x):
        # print(x.shape, self.pe.shape, self.pe[:, :x.shape[1]].shape)
        x = x + self.pe[:, :x.shape[1]] # (batch_size, seq_len, d_model)
        return x


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_heads, d_ff, n_layers, dropout=0.1, max_len=5000):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(src_vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(d_model)

    def forward(self, src, mask=None):
        src = self.token_embedding(src)  * self.scale
        src = self.position_embedding(src)
        src = self.dropout(src)

        for layer in self.layers:
            src = layer(src, mask)
        return src

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, n_heads, d_ff, n_layers, dropout=0.1, max_len=5000):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(d_model)


    def forward(self, tgt, tgt_mask, src, src_mask):
        tgt = self.token_embedding(tgt) * self.scale
        tgt = self.position_embedding(tgt)
        tgt = self.dropout(tgt)
        for layer in self.layers:
            tgt, attn = layer(tgt, tgt_mask, src, src_mask)

        out = self.fc_out(tgt)
        return out, attn

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, pad_idx) -> None:
        self.encoder = encoder
        self.decoder = decoder

        self.pad_idx = pad_idx

    def get_pad_mask(self,  x: np.ndarray) -> np.ndarray:
        #x: (batch_size, seq_len)
        return (x != self.pad_idx).astype(int)[:, np.newaxis, :]

    def get_sub_mask(self, x: np.ndarray) -> np.ndarray:
        #x: (batch_size, seq_len)
        seq_len = x.shape[1]
        subsequent_mask = np.triu(np.ones((seq_len, seq_len)), k = 1).astype(int)
        subsequent_mask = np.logical_not(subsequent_mask)
        return subsequent_mask

    def forward(self, src: np.ndarray, tgt: np.ndarray) -> tuple[Tensor, Tensor]:
        #src: (batch_size, source_seq_len)
        #tgt: (batch_size, target_seq_len)

        # src_mask: (batch_size, 1, seq_len)
        # tgt_mask: (batch_size, seq_len, seq_len)
        src_mask = self.get_pad_mask(src)

        tgt_mask = self.get_pad_mask(tgt) & self.get_sub_mask(tgt)

        src, src_mask = neunet.tensor(src, dtype=neunet.int32, device=device), neunet.tensor(src_mask, dtype=neunet.int32, device=device)
        tgt, tgt_mask = neunet.tensor(tgt, dtype=neunet.int32, device=device), neunet.tensor(tgt_mask, dtype=neunet.int32, device=device)
        
        enc_src = self.encoder(src, src_mask)

        out, attention = self.decoder(tgt, tgt_mask, enc_src, src_mask)
        # out: (batch_size, target_seq_len, vocab_size)
        # attention: (batch_size, heads_num, target_seq_len, source_seq_len)
        return out, attention
    


# [Data preprocessing]

BATCH_SIZE = 32

PAD_TOKEN = '<pad>' # noqa: S105
SOS_TOKEN = '<sos>' # noqa: S105
EOS_TOKEN = '<eos>' # noqa: S105
# UNK_TOKEN = '<unk>'

DATASET_PATH = Path("./datasets/multi30k/")
SAVE_PATH = Path("./saved models/seq2seq/")

if not DATASET_PATH.exists():
    data = load_dataset("bentrevett/multi30k", cache_dir="datasets/multi30k")

    for split, split_dataset in data.items():
        with Path(f"./datasets/multi30k/{split}.en").open('w', encoding='utf-8') as f:
            for item in split_dataset:
                f.write(item['en'] + '\n')

        with Path(f"./datasets/multi30k/{split}.de").open('w', encoding='utf-8') as f:
            for item in split_dataset:
                f.write(item['de'] + '\n')

FILE_PATHS = [DATASET_PATH / "train.en", DATASET_PATH / "train.de", DATASET_PATH / "val.en", DATASET_PATH / "val.de", DATASET_PATH / "test.en", DATASET_PATH / "test.de"]


# [Train and load Tokenizer]
if not (SAVE_PATH / "vocab").exists():
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files=[str(path) for path in FILE_PATHS], vocab_size=15000, min_frequency=1, special_tokens=[
        PAD_TOKEN,
        SOS_TOKEN,
        EOS_TOKEN,
        # UNK_TOKEN
    ])

    (SAVE_PATH / "vocab").mkdir(parents=True)
    tokenizer.save_model(str(SAVE_PATH / "vocab"), "multi30k-tokenizer")

tokenizer = ByteLevelBPETokenizer(
    str(SAVE_PATH / "vocab/multi30k-tokenizer-vocab.json"),
    str(SAVE_PATH / "vocab/multi30k-tokenizer-merges.txt"),
)



PAD_INDEX = tokenizer.token_to_id(PAD_TOKEN)
SOS_INDEX = tokenizer.token_to_id(SOS_TOKEN)
EOS_INDEX = tokenizer.token_to_id(EOS_TOKEN)
# UNK_INDEX = tokenizer.token_to_id(UNK_TOKEN)


class DataPreprocessor():
    def __init__(self, tokenizer: ByteLevelBPETokenizer):
        self.tokenizer = tokenizer

        self.tokenizer._tokenizer.post_processor  = TemplateProcessing( # noqa SLF001
            single=f"{SOS_TOKEN} $A {EOS_TOKEN}",
            special_tokens=[
                (f"{SOS_TOKEN}", tokenizer.token_to_id(f"{SOS_TOKEN}")),
                (f"{EOS_TOKEN}", tokenizer.token_to_id(f"{EOS_TOKEN}")),
            ],
        )

        # self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_token = PAD_TOKEN)
        
    def tokenize(self, paths: list[str], batch_size: int, lines_limit: Optional[int] = None) -> list[np.ndarray]:
        examples = []

        for src_file in paths:
            print(f"Processing {src_file}")
            path_src_file = Path(src_file)
            lines = path_src_file.read_text(encoding="utf-8").splitlines()

            if lines_limit:
                lines = lines[:lines_limit]

            for i in range(0, len(lines), batch_size):
                examples.append(np.array([x.ids for x in self.tokenizer.encode_batch(lines[i:i+batch_size])]))


        return examples

    def __call__(self, paths: list[str], batch_size: int, lines_limit: Optional[int] = None) -> list[np.ndarray]:
        return self.tokenize(paths, batch_size, lines_limit)


data_post_processor = DataPreprocessor(tokenizer)

train_src = data_post_processor([str(DATASET_PATH / "train.en")], batch_size = BATCH_SIZE)
train_tgt = data_post_processor([str(DATASET_PATH / "train.de")], batch_size = BATCH_SIZE)

val_src = data_post_processor([str(DATASET_PATH / "val.en")], batch_size = BATCH_SIZE)
val_tgt = data_post_processor([str(DATASET_PATH / "val.de")], batch_size = BATCH_SIZE)

test_src = data_post_processor([str(DATASET_PATH / "test.en")], batch_size = BATCH_SIZE)
test_tgt = data_post_processor([str(DATASET_PATH / "test.de")], batch_size = BATCH_SIZE)


train_data = train_src, train_tgt
val_data = val_src, val_tgt
test_data = test_src, test_tgt



# [Model intialization]

encoder = Encoder(
    src_vocab_size = tokenizer.get_vocab_size(),
    d_model = 256, # 512
    n_heads = 8,
    d_ff = 512, # 2048
    n_layers = 3, # 6
    dropout = 0.1
)

decoder = Decoder(
    tgt_vocab_size = tokenizer.get_vocab_size(),
    d_model = 256, # 512
    n_heads = 8,
    d_ff = 512, # 2048
    n_layers = 3, # 6
    dropout = 0.1
)


model = Seq2SeqTransformer(
    encoder = encoder,
    decoder = decoder,
    pad_idx = PAD_INDEX
)

device = "cuda"
model = model.to(device)

optimizer = Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps = 1e-9)

loss_function = nn.CrossEntropyLoss(ignore_index = PAD_INDEX)



# [train, eval, predict methods definition]

def train_step(source: list[np.ndarray], target: list[np.ndarray], epoch: int, epochs: int) -> float:
    total_loss = 0.0
    model.train()

    tqdm_range = tqdm(enumerate(zip(source, target, strict=False)), total = len(source))
    for batch_num, (source_batch, target_batch) in tqdm_range:

        output, _ = model.forward(source_batch, target_batch[:,:-1])
        
        output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])

        loss = loss_function(output, neunet.tensor(target_batch[:, 1:].flatten(), device=device, dtype=neunet.int32))
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()
        total_loss += loss.detach().item()


        tqdm_range.set_description(
                f"training | loss: {loss.detach().item():.7f} | perplexity: {np.exp(loss.detach().item()):.7f} | epoch {epoch + 1}/{epochs}" #loss: {loss:.4f}
            )

        if batch_num == (len(source) - 1):
            epoch_loss = total_loss / len(source)

            tqdm_range.set_description(
                    f"training | avg loss: {epoch_loss:.7f} | avg perplexity: {np.exp(epoch_loss):.7f} | epoch {epoch + 1}/{epochs}"
            )

    return epoch_loss

def eval(source: list[np.ndarray], target: list[np.ndarray]) -> float:
    total_loss = 0.0
    model.eval()

    tqdm_range = tqdm(enumerate(zip(source, target, strict=False)), total = len(source))
    for batch_num, (source_batch, target_batch) in tqdm_range:
        
        output, _ = model.forward(source_batch, target_batch[:,:-1])
        
        output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])
        
        loss = loss_function(output, neunet.tensor(target_batch[:, 1:].flatten(), device=device, dtype=neunet.int32))
        total_loss += loss.detach().item()
        
        tqdm_range.set_description(
                f"testing  | loss: {loss.detach().item():.7f} | perplexity: {np.exp(loss.detach().item()):.7f}"
            )

        if batch_num == (len(source) - 1):
            epoch_loss = total_loss / len(source)

            tqdm_range.set_description(
                    f"testing  | avg loss: {epoch_loss:.7f} | avg perplexity: {np.exp(epoch_loss):.7f}"
            )

    return epoch_loss


def train(train_data: tuple[list[np.ndarray], list[np.ndarray]], val_data: tuple[list[np.ndarray], list[np.ndarray]], epochs: int, save_every_epochs: int, save_path: Optional[str] = None, validation_check: bool = False):
    best_val_loss = float('inf')
    
    train_loss_history = []
    val_loss_history = []

    train_source, train_target = train_data
    val_source, val_target = val_data

    for epoch in range(epochs):

        train_loss_history.append(train_step(train_source, train_target, epoch, epochs))
        val_loss_history.append(eval(val_source, val_target))


        if (save_path is not None) and ((epoch + 1) % save_every_epochs == 0):
            if not Path(save_path).exists():
                Path(save_path).mkdir(parents=True, exist_ok=True)
            if validation_check == False:

                neunet.save(model.state_dict(), f"{save_path}/seq2seq_{epoch + 1}.nt")
            else:
                if val_loss_history[-1] < best_val_loss:
                    best_val_loss = val_loss_history[-1]
                    
                    neunet.save(model.state_dict(), f"{save_path}/seq2seq_{epoch + 1}.nt")
                else:
                    print('Current validation loss is higher than previous. Not saved.')
                    break
            
    return train_loss_history, val_loss_history



def predict(sentence: str, max_length: int = 50) -> tuple[str, Tensor]:
    model.eval()

    tokens = tokenizer.encode(sentence)
    src_ids = tokens.ids
    # src_ids = [SOS_INDEX] + src_ids + [EOS_INDEX] # Special tokens already here
    
    src = np.asarray(src_ids).reshape(1, -1)
    src_mask =  model.get_pad_mask(src)

    src, src_mask = neunet.tensor(src, dtype=neunet.int32, device=device), neunet.tensor(src_mask, dtype=neunet.int32, device=device)

    enc_src = model.encoder.forward(src, src_mask)

    tgt_tokens = [SOS_INDEX]

    for _ in range(max_length):
        tgt = np.asarray(tgt_tokens).reshape(1, -1)
        tgt_mask = model.get_pad_mask(tgt) & model.get_sub_mask(tgt)

        tgt, tgt_mask = neunet.tensor(tgt, dtype=neunet.int32, device=device), neunet.tensor(tgt_mask, dtype=neunet.int32, device=device)

        outputs, attention = model.decoder.forward(tgt, tgt_mask, enc_src, src_mask)
        
        tgt_next_token = outputs.detach().cpu().numpy().argmax(axis=-1)[:, -1].item()
        tgt_tokens.append(tgt_next_token)

        if tgt_next_token == EOS_INDEX or len(tgt_tokens) >= max_length:
            break
    
    
    # Remove special tokens
    if SOS_INDEX in tgt_tokens:
        tgt_tokens.remove(SOS_INDEX)
    if EOS_INDEX in tgt_tokens:
        tgt_tokens.remove(EOS_INDEX)

    decoded_sentence = tokenizer.decode(tgt_tokens)

    return decoded_sentence, attention


# [Train the Model]

# model.load_state_dict(neunet.load("./saved models/seq2seq/seq2seq_8.nt"))

train_loss_history, val_loss_history = None, None
train_loss_history, val_loss_history = train(train_data, val_data, epochs=30, save_every_epochs=1, save_path = str(SAVE_PATH), validation_check=True)


# [Model inferecnce and Plot]

def plot_loss_history(train_loss_history, val_loss_history):
    plt.plot(train_loss_history)
    plt.plot(val_loss_history)
    plt.title('Loss history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
        
if train_loss_history is not None and val_loss_history is not None:
    plot_loss_history(train_loss_history, val_loss_history)



raw_test_data: list[dict[str, str]] = []

en_file = [l.strip() for l in Path(DATASET_PATH / "test.en").open('r', encoding='utf-8')]
de_file = [l.strip() for l in Path(DATASET_PATH / "test.de").open('r', encoding='utf-8')]

for i in range(len(en_file)):
    if en_file[i] == '' or de_file[i] == '':
        continue
    en_seq, de_seq = en_file[i], de_file[i]

    raw_test_data.append({'en': en_seq, 'de': de_seq})
    
sentences_num = 10

random_indices = np.random.randint(0, len(raw_test_data), sentences_num)
sentences_selection = [raw_test_data[i] for i in random_indices]

# [Translate sentences from validation set]
for i, example in enumerate(sentences_selection):
    print(f"\nExample №{i + 1}")
    print(f"Input sentence: {example['en']}")
    print(f"Decoded sentence: {predict(example['en'])[0]}")
    print(f"Target sentence: {example['de']}")



def plot_attention(sentence: str, translation: str, attention: Tensor, heads_num: int = 8, rows_num: int = 2, cols_num: int = 4):
    if rows_num * cols_num != heads_num:
        raise ValueError("heads_num must be equal to rows_num * cols_num")
    attention = attention.detach().cpu().numpy().squeeze()

    sentence = tokenizer.encode(sentence, add_special_tokens=False).tokens
    translation = tokenizer.encode(translation, add_special_tokens=False).tokens


    fig = plt.figure(figsize = (15, 25))
    
    for h in range(heads_num):
        
        ax = fig.add_subplot(rows_num, cols_num, h + 1)
        ax.set_xlabel(f'Head {h + 1}')
        
        ax.matshow(attention[h], cmap = 'inferno')

        ax.tick_params(labelsize = 7)

        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(translation)))

        ax.set_xticklabels(sentence, rotation=90)
        ax.set_yticklabels(translation)


    plt.show()

# [Plot Attention]
sentence = sentences_selection[0]['en']
print(f"\nInput sentence: {sentence}")
decoded_sentence, attention =  predict(sentence)
print(f"Decoded sentence: {decoded_sentence}")

plot_attention(sentence, decoded_sentence, attention)