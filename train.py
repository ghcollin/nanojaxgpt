from functools import partial

# We start by reading in the entire training corpus
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Two choices of encoding are implemented here. 

# This uses a pre-trained encoding. "gpt2" gives the smallest voacb size of the available options.
import tiktoken
enc = tiktoken.encoding_for_model("gpt2")
vocab_size = enc.n_vocab
def encode(text):
    return enc.encode(text)
def decode(tokens):
    return enc.decode(tokens)

# A much smaller vocab size can be obtained by using character-wise encoding.

# vocab = list(set(text)) # list of all unique characters in the training corpus
# vocab.sort()
# vocab_transpose = { c:vocab.index(c) for c in vocab } # Get index given a character
# vocab_size = len(vocab)
# def encode(text_):
#     # The encoding is defined by the index of a character in `voacb`.
#     return list(map(lambda c: vocab_transpose[c], text_))
# def decode(tokens):
#     return "".join(map(lambda t: vocab[t], tokens))

print("Vocab size = {}".format(vocab_size))

text_enc = encode(text) # An encoded verison of the training corpus

import numpy

data = numpy.array(text_enc)

# We now create a test/train split. 10% of data is reserved for testing.
n_train = int(0.9*len(data))
train_data = data[:n_train]
val_data = data[n_train:]

####################
# Defining the model
####################

# These are the various hyperparameters of the network.
block_size = 128 # How many tokens will be included in one training example
batch_size = 16
n_embed = 4*64 # The number of dimensions used when embedding tokens into a vector space
head_size = 4*64 # Controls the dimension of the query and key vector spaces, as well as the output of the Head
num_heads = 4 # Number of parallel heads to run
n_layers = 10 # How many combined attention/feed forward layers to use
linear_up_factor = 4 # Factor by which to increase the embedding dimenion inside the feed forward layers
dropout_rate = 0.2

numpy.random.seed(0)

def get_batch(split):
    # Returns a batch of examples selected randomly from the training/testing corpus.
    # Testing or training is selected with the `split` argument.
    # The batch is organised as a tuple, with the inputs first and the targets second.
    data_set = train_data if split == 'train' else val_data
    # Each example in the batch is defined by a random offset into the corpus.
    rand_idxs = numpy.random.randint(len(data_set) - block_size, size=(batch_size,))
    x = numpy.stack([ data_set[i:i+block_size] for i in rand_idxs ])
    # The targets are the next token along in the data, so y is the same as x but shifted across one.
    y = numpy.stack([ data_set[i+1:i+block_size+1] for i in rand_idxs ])
    return x, y

test_batch, test_targets = get_batch('train') # This is a dummy batch used later for defining array sizes.

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax

# Here we define a standard normal initializer to use, residual projection modules have a scaled
# initialisation, see https://github.com/karpathy/nanoGPT/blob/master/model.py under "init all weights"
dense_init = lambda proj: nn.initializers.normal(stddev=(0.02/jnp.sqrt(2*n_layers) if proj else 0.02))
# Then we create a function for building dense modules with this initialiser, default bias init is zero
dense_module = lambda f, proj=False: nn.Dense(f, kernel_init=dense_init(proj))
# We'll also define an embedding module with custom initialiser
embedding_module = lambda num_embeddings, features: nn.Embed(num_embeddings, features, embedding_init=nn.initializers.normal(stddev=0.02))

class Head(nn.Module):
    # The attention head module
    head_size: int # Dimension size for the query/key and output vector spaces
    training: bool # If currently training, for dropout

    @nn.compact
    def __call__(self, x):
        bck_len, emb_size = x.shape

        # The standard keys/queries/values matrices for attention.
        # These are formed by dense layers that transform each token from the embedding vector space to the head vector space.
        keys = jax.vmap(dense_module(self.head_size))(x)
        querys = jax.vmap(dense_module(self.head_size))(x)
        values = jax.vmap(dense_module(self.head_size))(x)

        # The weights define how much each token contributes to the output.
        # It is formed by the contraction in head vector space between the queries and keys.
        # As this contraction will tend to grow in size with the dimension of the head vector space, we also scale it.
        ln_weights = querys @ keys.T / jnp.sqrt(self.head_size)
        # The full weight matrix contains entries that allow earlier tokens to query later tokens, this violates
        # the autoregressive property of token prediction so we zero these offending entries out, forming a triangular matrix.
        ln_weights_masked = jnp.where(jnp.tril(jnp.full((bck_len, bck_len), True)), ln_weights, -jnp.inf)
        # The weights should be positive and normalised, we can use the softmax transform to do this.
        # Note that the softmax is performed individually for each row (axis=-1), such that the matrix product with the
        # values matrix forms a weighted average of the values
        weights = jax.nn.softmax(ln_weights_masked, axis=-1)
        weights = nn.Dropout(dropout_rate, deterministic=not self.training)(weights)
        return weights @ values

class MultiHead(nn.Module):
    # Organises multiple attention heads to opererate on input data in parallel
    training: bool

    @nn.compact
    def __call__(self, x):
        # Each head is identical in architecture, and differs only in parameters. Ideally then, we would use JAX primitives
        # to map the head architecture over the parameters.
        # multihead = nn.vmap(
        #     Head,
        #     variable_axes={'params': 0},
        #     split_rngs={'params': True},
        #     in_axes=0, out_axes=0
        # )(head_size=head_size//num_heads)
        # pre_proj = jnp.concatenate(multihead(jnp.tile(x, (num_heads, 1, 1))), axis=-1)
        # However, the above internally generates a generalised dot product operation, which is not yet implemented
        # in JAX on Metal. So instead, we do the mapping in Python (using a loop). This is functionally identical,
        # but comes at the cost of higher compile times.
        pre_proj = jnp.concatenate([ Head(head_size=head_size//num_heads, training=self.training)(x) for _ in range(num_heads) ], axis=-1)
        post_proj = jax.vmap(dense_module(n_embed, proj=True))(pre_proj)
        return post_proj

class FeedForward(nn.Module):
    # The feed forward layer used for computation after attention

    @nn.compact
    def __call__(self, x):
        # This layer is formed by a transformation to higher dimensional space, a relu, and then a projection back to
        # the embedding space.
        x2 = jax.vmap(dense_module(linear_up_factor*n_embed))(x)
        x3 = jax.nn.relu(x2)
        x4 = jax.vmap(dense_module(n_embed, proj=True))(x3)
        return x4

class Block(nn.Module):
    # One block is formed by an attention layer followed by a feed forward layer.
    # The block uses skip connections (resnet) and layer normalisation.
    training: bool

    @nn.compact
    def __call__(self, x):
        x = x + nn.Dropout(dropout_rate, deterministic=not self.training)(MultiHead(training=self.training)(nn.LayerNorm()(x)))
        x = x + nn.Dropout(dropout_rate, deterministic=not self.training)(FeedForward()(nn.LayerNorm()(x)))
        return x

class Model(nn.Module):
    # The whole model
    training: bool

    @nn.compact
    def __call__(self, block):
        # First, the tokens are embedded into a vector space.
        tok_emb = jax.vmap(embedding_module(vocab_size, n_embed))(block)
        # And an encoding of the token positions is also learned in this vector space.
        pos_emb = jax.vmap(embedding_module(block_size, n_embed))(jnp.arange(len(block)))
        x = tok_emb + pos_emb # The final embedding is a linear combination of both.
        # Blocks are then constructed to do the computation.
        x = nn.Sequential([ Block(training=self.training) for _ in range(n_layers) ])(x)
        # A final projection into a vector space that represents the unnormalised log-probability of tokens.
        logits = jax.vmap(dense_module(vocab_size))(nn.LayerNorm()(x))

        return logits

####################
# Training the model
####################

# We now need to set up some flax specific boiler-plate for managing the training.
import flax.training.train_state
class TrainState(flax.training.train_state.TrainState):
    rng_key: jax.Array

def create_train_state(module, rng_key, learning_rate):
    # This initialises a TrainState, containing a derived rng key, for the given model, using the AdamW optimiser.
    rng_key, init_key = jax.random.split(rng_key)
    params = module.init(init_key, test_batch[0])['params']
    del init_key
    tx = optax.adamw(learning_rate)
    return TrainState.create(
        apply_fn = module.apply, params=params, tx=tx, rng_key=rng_key
    )

@jax.jit
def train_step(state, batch, targets):
    # Fold-in has better random properties than split and carry.
    dropout_train_key = jax.random.fold_in(state.rng_key, state.step)
    # But we still need to split out more keys for each example in the batch.
    dropout_train_keys = jax.random.split(dropout_train_key, len(batch))
    # We now define the loss function over the whole batch. We do this as a function of params to make it easier to
    # apply the JAX gradient primitive.
    def loss_fn(params):
        # apply_one evaluates the logits for just one example
        apply_one = lambda x, key: state.apply_fn({'params': params}, x, rngs={'dropout': key})
        # Then we vmap it over the whole batch
        logits = jax.vmap(apply_one)(batch, dropout_train_keys)
        # The resulting logits array is 3D, with dims: batch, time (token sequence), logits over vocab size.
        # Thus, we need to double vmap to apply softmax_cross_entropy over just the logits dimension.
        loss = jax.vmap(jax.vmap(lambda logits_, target: optax.softmax_cross_entropy(logits_, jax.nn.one_hot(target, num_classes=vocab_size))))(logits, targets)
        return loss.mean()
    # We now compute the gradients and update the parameters accordingly.
    grad_fn = jax.value_and_grad(loss_fn)
    value, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return value, state

# We need one model for training, and a second for test/validation.
module = Model(training=True)
val_module = Model(training=False)
def val_apply(state, x):
    # This function returns the logits for a given input block, x
    return val_module.apply({'params': state.params}, x)
def val_loss(state, batch, targets):
    # Here we compute the loss for a given batch, for use in test/validation.
    apply_one = lambda x: val_apply(state, x)
    logits = jax.vmap(apply_one)(batch)
    loss = jax.vmap(jax.vmap(lambda logits_, target: optax.softmax_cross_entropy(logits_, jax.nn.one_hot(target, num_classes=vocab_size))))(logits, targets)
    return loss.mean()

# Now we initialise the training state.
rng_key = jax.random.key(0)
rng_key, init_rng = jax.random.split(rng_key)
state = create_train_state(module, init_rng, learning_rate=1e-3)
del init_rng

loss_avg = 0.0 # Initial value for the average training loss.
# Hyperparameters for diagnostics during training.
loss_avg_rate = 0.1 # Exponential averaging parameters for the training loss.
print_rate = 100 # How often to print diagnostics.
val_runs = 10 # How many batches to average the test/validation loss over per diagnostic.


# Now we define the training loop
for iter in range(10000):
    # We get the next batch
    batch, targets = get_batch('train')

    #for i in range(block_size):
    # Train on this batch
    loss, state = train_step(state, batch, targets)
    # Update the average training loss
    loss_avg = loss_avg_rate * loss + (1-loss_avg_rate) * loss_avg

    if ((iter+1) % print_rate) == 0:
        # If it is time to print diagnostics.
        vloss = 0.0 # Initial value for test/validation loss average.
        for _ in range(val_runs):
            # Accumulate losses for test/validation data.
            vbatch, vtargets = get_batch('val')
            vloss += val_loss(state, vbatch, vtargets)
        vloss = vloss/val_runs # Compute the average
        print("train = {}, val = {}".format(loss_avg, vloss))

print("final train = {}".format(loss_avg)) 

#######################################
# Generate a response based on a prompt
#######################################

generate_length = 1000 # how many tokens to generate past the prompt.

prompt = encode("\n")

output = numpy.array(prompt + [])

rng_key, gen_key = jax.random.split(rng_key)

gen_model = jax.jit(lambda x: val_apply(state, x))

# Now we loop to draw random tokens according to the distribution given by the model conditioned on the current output.
for _ in range(generate_length):
    # The first few iterations will induce recompilation as output[-block_size:] will be growing in size until
    # it reached block_size.
    logits = gen_model(output[-block_size:])[-1]
    gen_key, token_rng_key = jax.random.split(gen_key)
    token = jax.random.categorical(token_rng_key, logits) # draws a random token according to the distribution of logits
    output = numpy.append(output, token)

print(decode(output))
