# -*- coding: utf-8 -*-
#
# task as translation

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch using", device)

import os
import numpy as np

np.warnings.filterwarnings('ignore')

CUSTOM_SEED = 42
np.random.seed(CUSTOM_SEED)

from tqdm import tqdm

# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import np_utils
from numpy import argmax
from collections import Counter

from models.base_model import BaseModel
from models.keyword_features import FeaturesReader, getTrainTestData, normaliseFeatures, getRootDir

MAX_LENGTH = 100

SOS_token = 0
EOS_token = 1
UNK_token = 3

EOS_marker = "#EOS"

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%d:%d:%d' % (h, m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s ( %s)' % (asMinutes(s), asMinutes(rs))


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 3: "UNK"}
        self.n_words = len(self.index2word)  # Count SOS / EOS / UNK

    def addSentence(self, sentence):
        assert isinstance(sentence, list)

        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


#

######################################################################
# The Seq2Seq Model
# =================
#
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.
#
# A `Sequence to Sequence network <http://arxiv.org/abs/1409.3215>`__, or
# seq2seq network, or `Encoder Decoder
# network <https://arxiv.org/pdf/1406.1078v3.pdf>`__, is a model
# consisting of two RNNs called the encoder and decoder. The encoder reads
# an input sequence and outputs a single vector, and the decoder reads
# that vector to produce an output sequence.
#
# .. figure:: /_static/img/seq-seq-images/seq2seq.png
#    :alt:
#
# Unlike sequence prediction with a single RNN, where every input
# corresponds to an output, the seq2seq model frees us from sequence
# length and order, which makes it ideal for translation between two
# languages.
#
# Consider the sentence "Je ne suis pas le chat noir" → "I am not the
# black cat". Most of the words in the input sentence have a direct
# translation in the output sentence, but are in slightly different
# orders, e.g. "chat noir" and "black cat". Because of the "ne/pas"
# construction there is also one more word in the input sentence. It would
# be difficult to produce a correct translation directly from the sequence
# of input words.
#
# With a seq2seq model the encoder creates a single vector which, in the
# ideal case, encodes the "meaning" of the input sequence into a single
# vector — a single point in some N dimensional space of sentences.
#


######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#


class ExtraFeaturesEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, extra_features):
        super(ExtraFeaturesEncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        # self.extra_features =

        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # self.embedding = torch.cat([self.embedding, self.extra_features])
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#


######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
#
# If only the context vector is passed betweeen the encoder and decoder,
# that single vector carries the burden of encoding the entire sentence.
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
#
# .. figure:: https://i.imgur.com/1152PYf.png
#    :alt:
#
# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.
#
# .. figure:: /_static/img/seq-seq-images/attention-decoder-network.png
#    :alt:
#
#

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################
# .. note:: There are other forms of attention that work around the length
#   limitation by using a relative position approach. Read about "local
#   attention" in `Effective Approaches to Attention-based Neural Machine
#   Translation <https://arxiv.org/abs/1508.04025>`__.
#
# Training
# ========
#
# Preparing Training Data
# -----------------------
#
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
#

def indexesFromSentence(lang, sentence):
    res = []
    for word in sentence:
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(UNK_token)

    return res


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def tensorsWithFeatures(context, input_lang, output_lang, dict_vectorizer):
    features = [dict_vectorizer.transform(t) for t in context["tokens"]]
    text_in = getTextTokens(context)
    text_out = getTokensToExtract(context)

    feat_len = len(features[0])
    features_tensors = [torch.tensor(feat, dtype=torch.long, device=device).view(-1, feat_len) for feat in features]

    indexes = tensorFromSentence(input_lang, text_in)
    input_list = [p for p in zip(indexes, features_tensors)]

    target_tensor = tensorFromSentence(output_lang, text_out)
    return (input_list, target_tensor)


def matrixFromContextFeatures(contexts, dict_vectorizer, MAX_CONTEXT_LEN, dtype=np.float32):
    all_features = [dict_vectorizer.transform(context) for context in contexts]
    # all_features = torch.(all_features, maxlen=MAX_CONTEXT_LEN, padding="post", dtype=dtype)
    #     matrix=np.zeros((len(all_features),all_features[0].shape[0],all_features[0].shape[1]))
    #     for index, vect in enumerate(all_features):
    #         matrix[index,:,:]=vect
    # print(dict_vectorizer.get_feature_names())
    matrix = np.stack(all_features, axis=0)
    print(matrix.shape)
    return matrix


######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
#

teacher_forcing_ratio = 0.0


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = min(input_tensor.size(0), max_length)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points, filename):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(filename, dpi=600)


######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append(EOS_marker)
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#


######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it makes it easier to run multiple experiments) we can actually
# initialize a network and start training.
#
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
#
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
#

# ======================================

def getTextTokens(context):
    tokens = [t["text"] for t in context["tokens"]]
    return tokens


def getContextsTextTokens(contexts):
    return [getTextTokens(context) for context in contexts]


def getTokensToExtract(context):
    return [t[0] for t in context["best_kws"]]


def getTargetTranslations(contexts):
    translations = []
    for context in contexts:
        tokens = [t[0] for t in context["best_kws"]]
        translations.append(tokens)
    return translations


def measurePR(truth, predictions):
    if len(truth) == 0:
        return 0, 0

    tp = fp = fn = 0
    for word in predictions:
        if word in truth:
            tp += 1
        else:
            fp += 1

    for word in truth:
        if word not in predictions:
            fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall, tp, (tp + fp), (tp + fn)


class TorchModel(BaseModel):
    def __init__(self, exp_dir, params={},
                 train_data_filename="feature_data.json.gz",
                 test_data_filename="feature_data_test.json.gz"):
        super(TorchModel, self).__init__(exp_dir, params, train_data_filename, test_data_filename)

        self.n_iters = params.get("n_iters", 50000)
        self.print_every = params.get("print_every", 100)
        self.plot_every = params.get("plot_every", 100)
        self.learning_rate = params.get("learning_rate", 0.01)
        self.hidden_size = params.get("hidden_size", 512)
        self.dropout_p = params.get("dropout_p", 0.1)

    def augmentSingleContext(self, context):
        pass

    def processFeatures(self):
        self.context_tokens = getContextsTextTokens(self.contexts)

        # for context in tqdm(self.contexts, desc="Adding context features"):
        #     self.augmentSingleContext(context)

        normaliseFeatures(self.contexts)

    def postProcessLoadedData(self):
        self.MAX_CONTEXT_LEN = max([len(x["tokens"]) for x in self.contexts]) + 2

        train_val_cutoff = int(.80 * len(self.contexts))
        self.training_contexts = self.contexts[:train_val_cutoff]
        self.validation_contexts = self.contexts[train_val_cutoff:]

        self.X_train = getContextsTextTokens(self.training_contexts)
        self.X_val = getContextsTextTokens(self.validation_contexts)

        # self.X_train, self.y_train = getTrainTestData(self.training_contexts)
        # self.X_val, self.y_val = getTrainTestData(self.validation_contexts)

        # self.X_train = matrixFromContextFeatures(self.X_train, self.dict_vectorizer, self.MAX_CONTEXT_LEN)
        # self.X_val = matrixFromContextFeatures(self.X_val, self.dict_vectorizer, self.MAX_CONTEXT_LEN)

        self.y_train = getTargetTranslations(self.training_contexts)
        self.y_val = getTargetTranslations(self.validation_contexts)

        self.lang = Lang("input")
        # self.output_lang = Lang("output")

        for words in self.X_train + self.X_val:
            for word in words:
                self.lang.addWord(word)

        for words in self.y_train + self.y_val:
            for word in words:
                self.lang.addWord(word)

        self.pairs = [p for p in zip(self.X_train, self.y_train)]
        # normaliseFeatures(self.contexts)

    def defineModel(self):
        print("Creating model...")
        hidden_size = 512
        self.encoder = EncoderRNN(self.lang.n_words,
                                  hidden_size).to(device)
        self.decoder = AttnDecoderRNN(hidden_size,
                                      self.lang.n_words,
                                      dropout_p=self.dropout_p,
                                      max_length=self.MAX_CONTEXT_LEN).to(device)

    def trainModel(self):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.learning_rate)

        # training_pairs = [tensorsFromPair(pair, self.lang, self.lang)
        #                   for pair in self.pairs]

        lens = []
        to_choose = []
        for index, pair in enumerate(self.pairs):
            cur_len = len(pair[1])
            lens.append((index, cur_len))
            for _ in range(cur_len):
                to_choose.append(index)

        lens = sorted(lens, key=lambda x: x[1], reverse=True)

        # First we fill the list with unique examples, starting with the longest extracted query first
        pairs_list = [self.pairs[p[0]] for p in lens[:self.n_iters]]
        remaining = max(0, self.n_iters - len(lens))

        # If we need more training data, we stochastically pick more training examples by length as above
        random.shuffle(to_choose)
        pairs_list.extend([self.pairs[random.choice(to_choose)] for _ in range(remaining)])

        print("Vectorizing data...")
        training_pairs = [tensorsFromPair(p, self.lang, self.lang) for p in pairs_list]
        # training_pairs = [tensorsWithFeatures(context, self.lang, self.lang, self.dict_vectorizer) for context in self.training_contexts]

        criterion = nn.NLLLoss()

        print("Training...")
        for iter in range(1, self.n_iters + 1):
            try:
                training_pair = training_pairs[iter - 1]
                input_tensor = training_pair[0]
                target_tensor = training_pair[1]

                loss = train(input_tensor,
                             target_tensor,
                             self.encoder,
                             self.decoder,
                             encoder_optimizer,
                             decoder_optimizer,
                             criterion,
                             max_length=self.MAX_CONTEXT_LEN)
                print_loss_total += loss
                plot_loss_total += loss

                if iter % self.print_every == 0:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (timeSince(start, iter / self.n_iters),
                                                 iter, iter / self.n_iters * 100, print_loss_avg))

                if iter % self.plot_every == 0:
                    plot_loss_avg = plot_loss_total / self.plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0
            except KeyboardInterrupt:
                print("Training interrupted")
                break

        showPlot(plot_losses, os.path.join(self.exp_dir, "pytorch_training.png"))

    def testModel(self):
        print("Testing...")
        self.reader = FeaturesReader(self.test_data_filename)
        self.testing_contexts = [c for c in self.reader]

        self.X_test = getContextsTextTokens(self.testing_contexts)
        self.y_test = getTargetTranslations(self.testing_contexts)

        all_recall = []
        all_precision = []
        all_tp = []
        all_p_sum = []
        all_r_sum = []

        for index, tokens in enumerate(self.X_test):
            truth = {t for t in self.y_test[index]}
            predictions, attentions = evaluate(self.encoder,
                                               self.decoder,
                                               tokens,
                                               self.lang, self.lang, max_length=self.MAX_CONTEXT_LEN)

            predictions = [p for p in predictions if p != EOS_marker]
            predictions = Counter(predictions)
            precision, recall, tp, p_sum, r_sum = measurePR(truth, predictions)

            all_recall.append(recall)
            all_precision.append(precision)
            all_tp.append(precision)
            all_p_sum.append(p_sum)
            all_r_sum.append(r_sum)

        numsamples = float(len(all_recall))

        tp = sum(all_tp)
        p_sum = sum(all_p_sum)
        r_sum = sum(all_r_sum)

        overall_recall = sum(all_recall) / numsamples
        overall_precision = sum(all_precision) / numsamples

        print("Precision %d/%d %0.2f Recall %d/%d %0.2f" % (tp, p_sum, overall_precision,
                                                            tp, r_sum, overall_recall))

    def plotPerformance(self):
        """ Plot model loss and accuracy through epochs. """

        pass

    def saveModel(self):
        model_name = self.__class__.__name__

    def evaluateRandomly(self, n=10):
        for i in range(n):
            pair = random.choice(self.pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = evaluate(self.encoder, self.decoder, pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')


def main(n_iters=250000, reset=False):
    params = {
        "n_iters": n_iters,
        "print_every": 100,
        "learning_rate": 0.01,
    }
    exp_dir = os.path.join(getRootDir("aac"), "experiments", "aac_generate_kw_trace")
    model = TorchModel(exp_dir, params=params)
    model.run()


if __name__ == '__main__':
    import plac

    plac.call(main)
