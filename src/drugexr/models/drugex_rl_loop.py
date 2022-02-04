import getopt
import os
import sys
import time
from pathlib import Path
from shutil import copy2

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from drugexr.config.constants import MODEL_PATH, PROC_DATA_PATH, ROOT_PATH
from drugexr.data_structs import vocabulary
from drugexr.data_structs.environment import Environment
from drugexr.models.predictor import Predictor
from drugexr.scoring.ra_scorer import RetrosyntheticAccessibilityScorer
from drugexr.utils import tensor_ops, normalization

DEVICE = torch.device("cuda")


class PGLearner(object):
    """ Reinforcement learning framework with policy gradient. This class is the base structure for all
        policy gradient-based  deep reinforcement learning models.
    Arguments:
        agent (models.Generator): The agent which generates the desired molecules
        env (utils.Env): The environment which provides the reward and judge
                                 if the genrated molecule is valid and desired.
        prior: The auxiliary model which is defined differently in each methods.
    """

    def __init__(self, agent, env, prior=None, memory=None, mean_func='geometric'):
        self.replay = 1
        self.agent = agent
        self.prior = prior
        self.batch_size = 16  # * 4
        self.n_samples = 32  # * 8
        self.env = env
        self.epsilon = 1e-3
        self.penalty = 0
        self.scheme = 'PR'
        self.out = None
        self.memory = memory
        # mean_func: which function to use for averaging: 'arithmetic' or 'geometric'
        self.mean_func = mean_func

    def policy_gradient(self):
        pass

    def fit(self):
        best = 0
        last_save = 0
        log = open(self.out + '.log', 'w')
        for epoch in range(1000):
            print('\n----------\nEPOCH %d\n----------' % epoch)
            self.policy_gradient()
            seqs = self.agent.sample(self.n_samples)
            ix = tensor_ops.unique(seqs)
            smiles = [self.agent.voc.decode(s) for s in seqs[ix]]
            scores = self.env(smiles, is_smiles=True)

            desire = scores.DESIRE.sum() / self.n_samples
            score = scores[self.env.keys].values.mean()
            valid = scores.VALID.mean()

            if best <= score:
                torch.save(self.agent.state_dict(), self.out + '.pkg')
                best = score
                last_save = epoch

            print("Epoch: %d average: %.4f valid: %.4f unique: %.4f" %
                  (epoch, score, valid, desire), file=log)
            for i, smile in enumerate(smiles):
                score = "\t".join(['%0.3f' % s for s in scores.values[i]])
                print('%s\t%s' % (score, smile), file=log)
            if epoch - last_save > 100:
                break
        for param_group in self.agent.optim.param_groups:
            param_group['lr'] *= (1 - 0.01)
        log.close()


class Evolve(PGLearner):
    """ DrugEx algorithm (version 2.0)
    Reference: Liu, X., Ye, K., van Vlijmen, H.W.T. et al. DrugEx v2: De Novo Design of Drug Molecule by
               Pareto-based Multi-Objective Reinforcement Learning in Polypharmacology.
               J Cheminform (2021). https://doi.org/10.1186/s13321-019-0355-6
    Arguments:
        agent (models.Generator): The agent network which is constructed by deep learning model
                                   and generates the desired molecules.
        env (utils.Env): The environment which provides the reward and judge
                                 if the genrated molecule is valid and desired.
        prior (models.Generator): The pre-trained network which is constructed by deep learning model
                                   and ensure the agent to explore the approriate chemical space.
    """

    def __init__(self, agent, env, prior=None, crover=None, mean_func='geometric', memory=None):
        super(Evolve, self).__init__(agent, env, prior, mean_func=mean_func, memory=memory)
        self.crover = crover

    def policy_gradient(self, crover=None, memory=None, epsilon=None):
        seqs = []
        start = time.time()
        for _ in range(self.replay):
            seq = self.agent.evolve1(self.batch_size, epsilon=epsilon, crover=crover, mutate=self.prior)
            seqs.append(seq)
        t1 = time.time()
        seqs = torch.cat(seqs, dim=0)
        if memory is not None:
            mems = [memory, seqs]
            seqs = torch.cat(mems)
        smiles = np.array([self.agent.voc.decode(s) for s in seqs])
        # smiles = np.array(utils.canonicalize_list(smiles))
        ix = tensor_ops.unique(np.array([[s] for s in smiles]))
        smiles = smiles[ix]
        seqs = seqs[torch.LongTensor(ix).to(DEVICE)]

        scores = self.env.calc_reward(smiles, self.scheme)
        if memory is not None:
            scores[:len(memory), 0] = 1
            ix = scores[:, 0].argsort()[-self.batch_size * 4:]
            seqs, scores = seqs[ix, :], scores[ix, :]
        t2 = time.time()
        ds = TensorDataset(seqs, torch.Tensor(scores).to(DEVICE))
        loader = DataLoader(ds, batch_size=self.n_samples, shuffle=True)

        self.agent.PGLoss(loader)
        t3 = time.time()
        print(t1 - start, t2 - t1, t3 - t2)

    def fit(self):
        best = 0
        log = open(self.out + '.log', 'w')
        last_smiles = []
        last_scores = []
        interval = 250
        last_save = -1

        for epoch in range(10000):
            print('\n----------\nEPOCH %d\n----------' % epoch)
            if epoch < interval and self.memory is not None:
                self.policy_gradient(crover=None, memory=self.memory, epsilon=1e-1)
            else:
                self.policy_gradient(crover=self.crover, epsilon=self.epsilon)
            seqs = self.agent.sample(self.n_samples)
            smiles = [self.agent.voc.decode(s) for s in seqs]
            smiles = np.array(tensor_ops.canonicalize_smiles_list(smiles))
            ix = tensor_ops.unique(np.array([[s] for s in smiles]))
            smiles = smiles[ix]
            scores = self.env(smiles, is_smiles=True)

            desire = scores.DESIRE.sum() / self.n_samples
            if self.mean_func == 'arithmetic':
                score = scores[self.env.keys].values.sum() / self.n_samples / len(self.env.keys)
            else:
                score = scores[self.env.keys].values.prod(axis=1) ** (1.0 / len(self.env.keys))
                score = score.sum() / self.n_samples
            valid = scores.VALID.sum() / self.n_samples

            print("Epoch: %d average: %.4f valid: %.4f unique: %.4f" %
                  (epoch, score, valid, desire), file=log)
            if best < score:
                torch.save(self.agent.state_dict(), self.out + '.pkg')
                best = score
                last_smiles = smiles
                last_scores = scores
                last_save = epoch

            if epoch % interval == 0 and epoch != 0:
                for i, smile in enumerate(last_smiles):
                    score = "\t".join(['%.3f' % s for s in last_scores.values[i]])
                    print('%s\t%s' % (score, smile), file=log)
                self.agent.load_state_dict(torch.load(self.out + '.pkg'))
                self.crover.load_state_dict(torch.load(self.out + '.pkg'))
            if epoch - last_save > interval:
                break
        log.close()


class Generator(nn.Module):
    def __init__(self, voc: vocabulary.Vocabulary, embed_size=128, hidden_size=512, is_lstm=True, lr=1e-3):
        super(Generator, self).__init__()
        self.voc = voc
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = voc.size

        self.embed = nn.Embedding(voc.size, embed_size)
        self.is_lstm = is_lstm
        rnn_layer = nn.LSTM if is_lstm else nn.GRU
        self.rnn = rnn_layer(embed_size, hidden_size, num_layers=3, batch_first=True)
        self.linear = nn.Linear(hidden_size, voc.size)
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.to(DEVICE)

    def forward(self, input, h):
        output = self.embed(input.unsqueeze(-1))
        output, h_out = self.rnn(output, h)
        output = self.linear(output).squeeze(1)
        return output, h_out

    def init_h(self, batch_size, labels=None):
        h = torch.rand(3, batch_size, 512).to(DEVICE)
        if labels is not None:
            h[0, batch_size, 0] = labels
        if self.is_lstm:
            c = torch.rand(3, batch_size, self.hidden_size).to(DEVICE)
        return (h, c) if self.is_lstm else h

    def likelihood(self, target):
        batch_size, seq_len = target.size()
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(DEVICE)
        h = self.init_h(batch_size)
        scores = torch.zeros(batch_size, seq_len).to(DEVICE)
        for step in range(seq_len):
            logits, h = self(x, h)
            logits = logits.log_softmax(dim=-1)
            score = logits.gather(1, target[:, step:step+1]).squeeze()
            scores[:, step] = score
            x = target[:, step]
        return scores

    def PGLoss(self, loader):
        for seq, reward in loader:
            self.zero_grad()
            score = self.likelihood(seq)
            loss = score * reward
            loss = -loss.mean()
            loss.backward()
            self.optim.step()

    def sample(self, batch_size):
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(DEVICE)
        h = self.init_h(batch_size)
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(DEVICE)
        isEnd = torch.zeros(batch_size).bool().to(DEVICE)

        for step in range(self.voc.max_len):
            logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            x = torch.multinomial(proba, 1).view(-1)
            x[isEnd] = self.voc.tk2ix['EOS']
            sequences[:, step] = x

            end_token = (x == self.voc.tk2ix['EOS'])
            isEnd = torch.ge(isEnd + end_token, 1)
            if (isEnd == 1).all(): break
        return sequences

    def evolve(self, batch_size, epsilon=0.01, crover=None, mutate=None):
        # Start tokens
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(DEVICE)
        # Hidden states initialization for exploitation network
        h = self.init_h(batch_size)
        # Hidden states initialization for exploration network
        h1 = self.init_h(batch_size)
        h2 = self.init_h(batch_size)
        # Initialization of output matrix
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(DEVICE)
        # labels to judge and record which sample is ended
        is_end = torch.zeros(batch_size).bool().to(DEVICE)

        for step in range(self.voc.max_len):
            logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            if crover is not None:
                ratio = torch.rand(batch_size, 1).to(DEVICE)
                logit1, h1 = crover(x, h1)
                proba = proba * ratio + logit1.softmax(dim=-1) * (1 - ratio)
            if mutate is not None:
                logit2, h2 = mutate(x, h2)
                is_mutate = (torch.rand(batch_size) < epsilon).to(DEVICE)
                proba[is_mutate, :] = logit2.softmax(dim=-1)[is_mutate, :]
            # sampling based on output probability distribution
            x = torch.multinomial(proba, 1).view(-1)

            is_end |= x == self.voc.tk2ix['EOS']
            x[is_end] = self.voc.tk2ix['EOS']
            sequences[:, step] = x
            if is_end.all(): break
        return sequences

    def evolve1(self, batch_size, epsilon=0.01, crover=None, mutate=None):
        # Start tokens
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(DEVICE)
        # Hidden states initialization for exploitation network
        h = self.init_h(batch_size)
        # Hidden states initialization for exploration network
        h2 = self.init_h(batch_size)
        # Initialization of output matrix
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(DEVICE)
        # labels to judge and record which sample is ended
        is_end = torch.zeros(batch_size).bool().to(DEVICE)

        for step in range(self.voc.max_len):
            is_change = torch.rand(1) < 0.5
            if crover is not None and is_change:
                logit, h = crover(x, h)
            else:
                logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            if mutate is not None:
                logit2, h2 = mutate(x, h2)
                ratio = torch.rand(batch_size, 1).to(DEVICE) * epsilon
                proba = logit.softmax(dim=-1) * (1 - ratio) + logit2.softmax(dim=-1) * ratio
            # sampling based on output probability distribution
            x = torch.multinomial(proba, 1).view(-1)

            x[is_end] = self.voc.tk2ix['EOS']
            sequences[:, step] = x

            # Judging whether samples are end or not.
            end_token = (x == self.voc.tk2ix['EOS'])
            is_end = torch.ge(is_end + end_token, 1)
            #  If all of the samples generation being end, stop the sampling process
            if (is_end == 1).all(): break
        return sequences

    def fit(self, loader_train, out, loader_valid=None, epochs=100, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        log = open(out + '.log', 'w')
        best_error = np.inf
        for epoch in range(epochs):
            for i, batch in enumerate(loader_train):
                optimizer.zero_grad()
                loss_train = self.likelihood(batch.to(DEVICE))
                loss_train = -loss_train.mean()
                loss_train.backward()
                optimizer.step()
                if i % 10 == 0 or loader_valid is not None:
                    seqs = self.sample(len(batch * 2))
                    ix = tensor_ops.unique(seqs)
                    seqs = seqs[ix]
                    smiles, valids = self.voc.check_smiles(seqs)
                    error = 1 - sum(valids) / len(seqs)
                    info = "Epoch: %d step: %d error_rate: %.3f loss_train: %.3f" % (epoch, i, error, loss_train.item())
                    if loader_valid is not None:
                        loss_valid, size = 0, 0
                        for j, batch in enumerate(loader_valid):
                            size += batch.size(0)
                            loss_valid += -self.likelihood(batch.to(DEVICE)).sum().item()
                        loss_valid = loss_valid / size / self.voc.max_len
                        if loss_valid < best_error:
                            torch.save(self.state_dict(), out + '.pkg')
                            best_error = loss_valid
                        info += ' loss_valid: %.3f' % loss_valid
                    elif error < best_error:
                        torch.save(self.state_dict(), out + '.pkg')
                        best_error = error
                    print(info, file=log)
                    for i, smile in enumerate(smiles):
                        print('%d\t%s' % (valids[i], smile), file=log)
        log.close()


if __name__ == "__main__":
    DEV_RUN = False
    opts, args = getopt.getopt(sys.argv[1:], "a:e:b:g:c:s:z:")
    OPT = dict(opts)
    os.environ["CUDA_VISIBLE_DEVICES"] = OPT['-g'] if '-g' in OPT else "0"
    case = 'OBJ4'
    z = OPT['-z'] if '-z' in OPT else 'REG'
    alg = OPT['-a'] if '-a' in OPT else 'evolve'
    scheme = OPT['-s'] if '-s' in OPT else 'PR'
    
    if DEV_RUN:
        keys = ['RA_SCORER']
        RA_SCORER = RetrosyntheticAccessibilityScorer(use_xgb_model=False)
        objs = [RA_SCORER]
    
    else:
        # construct the environment with three predictors
        keys = ['A1', 'A2A', 'ERG', 'RA_SCORER']
        A1 = Predictor(path=MODEL_PATH / f"output/single/RF_{z}_CHEMBL226.pkg", type_=z)
        A2A = Predictor(path=MODEL_PATH / f"output/single/RF_{z}_CHEMBL251.pkg", type_=z)
        ERG = Predictor(path=MODEL_PATH / f"output/single/RF_{z}_CHEMBL240.pkg", type_=z)
        RA_SCORER = RetrosyntheticAccessibilityScorer(use_xgb_model=False)

        # Chose the desirability function
        objs = [A1, A2A, ERG, RA_SCORER]
        
    n_objs = len(objs)

    if scheme == 'WS':
        mod1 = normalization.ClippedScore(lower_x=3, upper_x=10)
        mod2 = normalization.ClippedScore(lower_x=10, upper_x=3)
        ths = [0.5] * n_objs
    else:
        mod1 = normalization.ClippedScore(lower_x=3, upper_x=6.5)
        mod2 = normalization.ClippedScore(lower_x=10, upper_x=6.5)
        ths = [0.99] * n_objs
        
    mods = [lambda x: x] if DEV_RUN else [mod2, mod1, mod2, lambda x: x]
    env = Environment(objs=objs, mods=mods, keys=keys, ths=ths)

    # root = 'output/%s_%s_%s_%s/'% (alg, case, scheme, time.strftime('%y%m%d_%H%M%S', time.localtime()))
    root_suffix = f"output/{alg}_{case}_{scheme}_{time.strftime('%y%m%d_%H%M%S', time.localtime())}/"
    root: Path = MODEL_PATH / root_suffix
    os.mkdir(root)
    copy2(ROOT_PATH / "src/drugexr/models/drugex_r.py", root)
    copy2(ROOT_PATH / "src/drugexr/training/train_drugex_r.py", root)

    pr_path = MODEL_PATH / "output/rnn/pretrained_lstm.ckpt"
    ft_path = MODEL_PATH / "output/rnn/fine_tuned_lstm_lr_1e-4.ckpt"

    voc = vocabulary.Vocabulary(vocabulary_path=PROC_DATA_PATH / "chembl_voc.txt")
    agent = Generator(voc)
    agent.load_state_dict(torch.load(ft_path)['state_dict'])

    prior = Generator(voc)
    prior.load_state_dict(torch.load(pr_path)['state_dict'])

    if DEV_RUN:
        crover = None
    else:
        crover = Generator(voc)
        crover.load_state_dict(torch.load(ft_path)['state_dict'])

    learner = Evolve(agent, env, prior, crover)

    learner.epsilon = learner.epsilon if '-e' not in OPT else float(OPT['-e'])
    learner.penalty = learner.penalty if '-b' not in OPT else float(OPT['-b'])
    learner.scheme = learner.scheme if scheme is None else scheme

    outfile = "%s_%s_%s_%s" % (alg, learner.scheme, z, case)
    outfile += "_%.0e" % learner.epsilon
    output_path = root / outfile
    learner.out = str(output_path)

    learner.fit()
