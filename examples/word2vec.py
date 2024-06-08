import sys
from pathlib import Path

sys.path[0] = str(Path(sys.path[0]).parent)

from tqdm import tqdm

import neunet as nnet
import neunet.nn as nn
import neunet.optim as optim

# Example based on the https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)]
        + [test_sentence[i + j + 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i],
    )
    for i in range(len(test_sentence) - CONTEXT_SIZE)
]

# Print the first 3, just so you can see what they look like.
print(ngrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class CBOWModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOWModeler, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2 * context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).reshape((1, -1))
        out = self.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = self.softmax(out).log()
        return log_probs


loss_function = nn.NLLLoss()
model = CBOWModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
tqdm_range = tqdm(range(epochs))
for _ in tqdm_range:
    total_loss = 0
    for context, target in ngrams:
        context_idxs = nnet.tensor([word_to_ix[w] for w in context], dtype=nnet.int16)

        optimizer.zero_grad()

        log_probs = model(context_idxs)

        loss = loss_function(
            log_probs,
            nnet.tensor([word_to_ix[target]], dtype=nnet.int16, requires_grad=False),
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    tqdm_range.set_description(f"CBOW loss: {total_loss:.7f}")

print(f'Embedding of the word "beauty":\n{model.embeddings.weight[word_to_ix["beauty"]]}')


class SkipGramModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(SkipGramModeler, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, 2 * context_size * vocab_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).reshape((1, -1))
        out = self.relu(self.linear1(embeds))
        out = self.linear2(out).reshape(2 * self.context_size, -1)
        log_probs = self.softmax(out).log()
        return log_probs


loss_function = nn.NLLLoss()
model = SkipGramModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
tqdm_range = tqdm(range(epochs))
for _ in tqdm_range:
    total_loss = 0
    for context, target in ngrams:
        context_idx = nnet.tensor([word_to_ix[target]], dtype=nnet.int16)

        optimizer.zero_grad()

        log_probs = model(context_idx)

        loss = loss_function(
            log_probs, nnet.tensor([word_to_ix[w] for w in context], dtype=nnet.int16)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    tqdm_range.set_description(f"Skip-Gram loss: {total_loss:.7f}")

print(f'Embedding of the word "beauty":\n{model.embeddings.weight[word_to_ix["beauty"]]}')
