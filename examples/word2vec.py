import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

import neunet as nnet
import neunet.nn as nn
import neunet.optim as optim
from tqdm import tqdm


# Example taken from the https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

CONTEXT_SIZE = 2
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
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.
# Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]
# Print the first 3, just so you can see what they look like.
print(ngrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


# Skip-Gram model
class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)
        

    def forward(self, inputs):
        embeds = self.embeddings(inputs).reshape((1, -1))
        out = self.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = self.softmax(out).log()
        return log_probs


# losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 100
tqdm_range = tqdm(range(epochs))
for epoch in tqdm_range:
    total_loss = 0
    for context, target in ngrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = nnet.tensor([word_to_ix[w] for w in context], dtype=nnet.int16)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        optimizer.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function.
        loss = loss_function(log_probs, nnet.tensor([word_to_ix[target]], dtype=nnet.int16, requires_grad=False))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        # print(context_idxs.grad)
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.data
    # losses.append(total_loss)
    tqdm_range.set_description(f"Skip-Gram loss: {total_loss:.7f}")
# print(losses)  # The loss decreased every iteration over the training data!

# To get the embedding of a particular word, e.g. "beauty"
print(f'Embedding of the word "beauty":\n{model.embeddings.weight[word_to_ix["beauty"]]}')