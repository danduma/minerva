# -*- coding: utf-8 -*-
#
# task as NER
# now with BiLSTM + CRF

from __future__ import unicode_literals, print_function, division

import os
import numpy as np

np.warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch using", device)

import torchtext.vocab as vocab

CUSTOM_SEED = 42
np.random.seed(CUSTOM_SEED)

from tqdm import tqdm

from collections import Counter

from models.base_model import BaseModel
from models.keyword_features import FeaturesReader, normaliseFeatures, measurePR, getRootDir

MAX_LENGTH = 100

START_TAG = "<START>"
STOP_TAG = "<STOP>"

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%02d:%02d:%02d' % (h, m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    if percent == 0:
        return "? (?)"
    es = s / percent
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

    def getIndex(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return 3


#####################################################################
# Helper functions to make the code more readable.


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, lang):
    idxs = [lang.getIndex(w) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def prepare_tags(context, tag_to_ix):
    return torch.tensor([tag_to_ix[t] for t in context["extract_mask"]], dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


#####################################################################
# Create model


class BiLSTM_CRF(nn.Module):

    def __init__(self, lang, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lang = lang
        self.vocab_size = lang.n_words
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
        self.loadWordVectors()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def loadWordVectors(self):
        local = "/Users/masterman/NLP/PhD/vectors/glove"
        if os.path.isdir(local):
            vector_dir = local
        else:
            vector_dir = "/tmp/tmp-1135029/glove"

        self.glove = vocab.GloVe(name='6B', dim=300, cache=vector_dir)
        print('Loaded {} words'.format(len(self.glove.itos)))

        for word, emb_index in self.lang.word2index.items():
            # if the word is in the loaded glove vectors
            if word.lower() in self.glove.stoi:
                # get the index into the glove vectors
                glove_index = self.glove.stoi[word.lower()]
                # finally, if net is our network, and emb is the embedding layer:
                self.word_embeds.weight.data[emb_index, :].set_(self.glove.vectors[glove_index])

        self.glove = None

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker


def showPlot(points, filename):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(filename, dpi=600)


# ======================================

def getTextTokens(context):
    tokens = [t["text"].lower() for t in context["tokens"]]
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


class TorchModel(BaseModel):
    def __init__(self, exp_dir, params={},
                 train_data_filename="feature_data.json.gz",
                 test_data_filename="feature_data_test.json.gz"):
        super(TorchModel, self).__init__(exp_dir, params, train_data_filename, test_data_filename)

        self.epochs = params.get("num_epochs", 10)
        self.optimizer_class = params.get("optimizer", "SGD")
        self.print_every = params.get("print_every", 100)
        self.plot_every = params.get("plot_every", 100)
        self.learning_rate = params.get("learning_rate", 0.01)
        self.hidden_size = params.get("hidden_size", 512)
        self.dropout_p = params.get("dropout_p", 0.1)
        self.tag_to_ix = {False: 0, True: 1, START_TAG: 2, STOP_TAG: 3}

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

        self.lang = Lang("input")

        for context in self.training_contexts:
            for token in context["tokens"]:
                self.lang.addWord(token["text"].lower())

    def defineModel(self):
        print("Creating model...")

        embedding_dim = 300
        hidden_dim = 200

        self.model = BiLSTM_CRF(self.lang, self.tag_to_ix, embedding_dim, hidden_dim)

    def trainModel(self):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        # Check predictions before training
        with torch.no_grad():
            precheck_sent = prepare_sequence([t["text"].lower() for t in self.training_contexts[0]["tokens"]],
                                             self.lang)
            precheck_tags = prepare_tags(self.training_contexts[0], self.tag_to_ix)
            print(self.model(precheck_sent))

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        # optimizer = getattr(optim, self.optimizer_class)
        # encoder_optimizer = optimizer(self.encoder.parameters(), lr=self.learning_rate)
        # decoder_optimizer = optimizer(self.decoder.parameters(), lr=self.learning_rate)

        # training_pairs = []
        # for context in tqdm(self.training_contexts, desc="Vectorizing data"):
        #     training_pairs.append(getTensorsWithFeatures(context,
        #                                                  self.lang,
        #                                                  self.lang,
        #                                                  self.dict_vectorizer))

        print("Training...")
        for epoch in range(1, self.epochs + 1):
            interrupted = False
            for iteration, context in enumerate(self.training_contexts):
                try:
                    # Step 1. Remember that Pytorch accumulates gradients.
                    # We need to clear them out before each instance
                    self.model.zero_grad()

                    # Step 2. Get our inputs ready for the network, that is,
                    # turn them into Tensors of word indices.
                    sentence = [t["text"].lower() for t in context["tokens"]]
                    sentence_in = prepare_sequence(sentence, self.lang)
                    targets = prepare_tags(context, self.tag_to_ix)
                    # targets = torch.tensor([self.tag_to_ix[t] for t in tags], dtype=torch.long)

                    # Step 3. Run our forward pass.
                    loss = self.model.neg_log_likelihood(sentence_in, targets)

                    # Step 4. Compute the loss, gradients, and update the parameters by
                    # calling optimizer.step()
                    loss.backward()
                    optimizer.step()

                    print_loss_total += loss
                    plot_loss_total += loss

                    if iteration % self.print_every == 0:
                        print_loss_avg = print_loss_total / self.print_every
                        print_loss_total = 0
                        print('Epoch %d: %s (%d %d%%) %.4f' % (epoch,
                                                               timeSince(start, iteration / float(self.epochs)),
                                                               iteration,
                                                               iteration / len(self.training_contexts),
                                                               print_loss_avg))

                    if iteration % self.plot_every == 0:
                        plot_loss_avg = plot_loss_total / self.plot_every
                        plot_losses.append(plot_loss_avg)
                        plot_loss_total = 0
                except KeyboardInterrupt:
                    print("Training interrupted")
                    interrupted = True
                    break

                ## Uncomment to run a single time
                # interrupted=True
                # break

            if interrupted:
                break

        showPlot(plot_losses, os.path.join(self.exp_dir, "pytorch_training.png"))

    def testModel(self):
        print("Testing...")
        self.reader = FeaturesReader(self.test_data_filename)
        self.testing_contexts = [c for c in self.reader]

        self.y_test = getTargetTranslations(self.testing_contexts)

        all_recall = []
        all_precision = []
        all_tp = []
        all_p_sum = []
        all_r_sum = []

        for index, context in enumerate(tqdm(self.testing_contexts)):
            truth = [self.tag_to_ix[t] for t in context["extract_mask"]]
            sentence_in = prepare_sequence([t["text"].lower() for t in context["tokens"]], self.lang)
            predicted = self.model(sentence_in)

            predicted_tokens = set()

            predicted = self.filterStopWordsInPredicted(context["tokens"], predicted)
            print("Extra stopwords removed", self.extra_stopwords_removed)

            for index, pred in enumerate(predicted):
                if pred:
                    predicted_tokens.add(context["tokens"][index]["text"].lower())

            predicted = Counter(predicted_tokens)
            precision, recall, tp, p_sum, r_sum = measurePR(truth, predicted)

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

    # def evaluateRandomly(self, n=10):
    #     for i in range(n):
    #         pair = random.choice(self.pairs)
    #         print('>', pair[0])
    #         print('=', pair[1])
    #         output_words, attentions = evaluate(self.encoder, self.decoder, pair[0])
    #         output_sentence = ' '.join(output_words)
    #         print('<', output_sentence)
    #         print('')


def main(num_epochs=1, reset=False):
    params = {
        "num_epochs": num_epochs,
        "hidden_size": 200,
        "print_every": 100,
        # "learning_rate": 0.003,
        "learning_rate": 0.01,
        # "optimizer": "Adam",
        "optimizer": "SGD",

    }
    exp_dir = os.path.join(getRootDir("aac"), "experiments", "aac_generate_kw_trace")
    model = TorchModel(exp_dir, params=params,
                       train_data_filename="feature_data_w_min2.json.gz",
                       test_data_filename="feature_data_test_w_min2.json.gz")
    model.run()


if __name__ == '__main__':
    import plac

    plac.call(main)
