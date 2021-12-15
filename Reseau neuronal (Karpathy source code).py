"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy) // Modèle basique de réseaux neuronal récurrent par Andrej Karpathy
BSD License
"""
import numpy as np

# Input de données
data = open('input.txt', 'r').read() # Doit être un fichier texte en .txt pour être compris par le programme
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# Paramètres du réseau
hidden_size = 100 # taille de la couche cachée de neuronnes
seq_length = 25 # Nombre d'étapes par laquelle le réseau va passer
learning_rate = 1e-1

# Paramètres du modèle
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input dans la couche cachée
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # de la couche cachée à une nouvelle couche cachée
Why = np.random.randn(vocab_size, hidden_size)*0.01 # de la couche cachée à l'output
bh = np.zeros((hidden_size, 1)) # biais pour la couche cachée de neurones
by = np.zeros((vocab_size, 1)) # biais pour l'output

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # Lecture du réseau de neurones du premier au dernier dans cet ordre (forward pass)
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encodage dans une représentation 1 sur k représentations totales
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # état caché du réseau
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilités du prochain caractère
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # Lecture du réseau de neurones du dernier au premier dans cet ordre (backward pass)
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # retropropagation dans y. le lien http://cs231n.github.io/neural-networks-case-study/#grad permet de mieux comprendre l'événement
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # retropropagation du gradient dans h (https://fr.wikipedia.org/wiki/R%C3%A9tropropagation_du_gradient)
    dhraw = (1 - hs[t] * hs[t]) * dh # retropropagation du gradient au travers d'une fonction tanh non linéaire
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # fonction clip afin de réduire le phénomène de gradient qui explose https://machinelearningmastery.com/exploding-gradients-in-neural-networks/
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # permet de faire bouger le pointeur
  n += 1 # compteur d'itérations 
