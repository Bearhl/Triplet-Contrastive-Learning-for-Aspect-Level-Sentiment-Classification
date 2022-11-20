import os
import six
import sys
import copy
import random
import logging
import argparse
import torch
import pickle
import torch.nn as nn
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW
from models.ian import IAN
from models.atae_lstm import ATAE_LSTM
from models.tcl import TCLClassifier
from models.tcl_bert import TCLBertClassifier
from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix, Tokenizer4BertGCN, ABSAGCNData
from prepare_vocab import VocabHelp
import torch.nn.functional as F

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Keyword_ConLoss(nn.Module):
    def __init__(self, opt, temperature=1):
        super(Keyword_ConLoss, self).__init__()
        self.opt = opt
        self.temperature = temperature

    def forward(self, anchor, postive, negative):
        seq_len = anchor.size(1)

        anchor_dot_pos = torch.mul(torch.bmm(anchor, postive.transpose(1, 2)), 1 / self.temperature)
        anchor_dot_neg = torch.mul(torch.bmm(anchor, negative.transpose(1, 2)), 1 / self.temperature)
        # for numerical stability
        logits_max_pos, _ = torch.max(anchor_dot_pos, dim=2, keepdim=True)
        logits_pos = anchor_dot_pos - logits_max_pos.detach()
        logits_max_neg, _ = torch.max(anchor_dot_neg, dim=2, keepdim=True)
        logits_neg = anchor_dot_neg - logits_max_neg.detach()

        mask = torch.eye(seq_len).to(self.opt.device)

        logits = logits_pos * mask
        logits = logits.sum(dim=2) - torch.log(
            torch.sum(torch.exp(logits_pos), dim=2) + torch.sum(torch.exp(logits_neg), dim=2))
        loss = - logits.mean()
        return loss


class Instructor:

    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            dep_tag_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep_tag.vocab')
            opt.deptag_size = len(dep_tag_vocab)
            tokenizer = Tokenizer4BertGCN(opt.max_length, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
            trainset = ABSAGCNData(opt.dataset_file['train'], tokenizer, opt=opt, dep_tag_vocab=dep_tag_vocab)
            print(trainset.max_len)
            testset = ABSAGCNData(opt.dataset_file['test'], tokenizer, opt=opt, dep_tag_vocab=dep_tag_vocab)
            print(testset.max_len)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_length=opt.max_length,
                data_file='{}/{}_tokenizer.dat'.format(opt.vocab_dir, opt.dataset))
            embedding_matrix = build_embedding_matrix(
                vocab=tokenizer.vocab,
                embed_dim=opt.embed_dim,
                data_file='{}/{}d_{}_embedding_matrix.dat'.format(opt.vocab_dir, str(opt.embed_dim), opt.dataset))

            logger.info("Loading vocab...")
            token_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_tok.vocab')  # token
            post_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_post.vocab')  # position
            pos_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pos.vocab')  # POS
            dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')  # deprel
            dep_tag_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep_tag.vocab')  # deprel
            pol_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pol.vocab')  # polarity
            logger.info(
                "token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(len(token_vocab),
                                                                                                      len(post_vocab),
                                                                                                      len(pos_vocab),
                                                                                                      len(dep_vocab),
                                                                                                      len(pol_vocab)))

            # opt.tok_size = len(token_vocab)
            opt.post_size = len(post_vocab)
            opt.pos_size = len(pos_vocab)
            opt.deptag_size = len(dep_tag_vocab)

            vocab_help = (post_vocab, pos_vocab, dep_vocab, dep_tag_vocab, pol_vocab)
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
            trainset = SentenceDataset(opt.dataset_file['train'], tokenizer, opt=opt, vocab_help=vocab_help)
            testset = SentenceDataset(opt.dataset_file['test'], tokenizer, opt=opt, vocab_help=vocab_help)

        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')

        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)  # xavier_uniform_
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_bert_optimizer(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]

        if self.opt.diff_lr:
            logger.info("layered learning rate on")
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.learning_rate
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)

        else:
            logger.info("bert learning rate on")
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.opt.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer

    def _calculate_contrast_loss(self, anchor, target, labels, mu=1.0):

        BS = len(labels)
        with torch.no_grad():
            labels = labels.reshape(-1, 1)
            mask = torch.eq(labels, labels.T)  # (bs, bs)
            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(BS).view(-1, 1).to(self.opt.device),
                0
            )
            mask = mask * logits_mask
        # compute logits
        anchor_dot_target = torch.mul(torch.matmul(anchor, target.T), 1 / self.opt.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        # in case that mask.sum(1) has no zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = - mean_log_prob_pos.mean()
        return loss

    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, penal, sem_feature, syn_feature, h2, h2_neg, h2_pos = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                con_loss = Keyword_ConLoss(self.opt)
                dep_loss = con_loss(h2, h2_pos, h2_neg)
                contrastive_loss1 = self._calculate_contrast_loss(sem_feature, syn_feature, targets.view(-1, 1))
                contrastive_loss2 = self._calculate_contrast_loss(syn_feature, sem_feature, targets.view(-1, 1))
                contrastive_loss = self.opt.beta * (contrastive_loss1 + contrastive_loss2) + \
                                   self.opt.gamma * dep_loss

                loss = criterion(outputs, targets) + penal + contrastive_loss

                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./state_dict'):
                                os.mkdir('./state_dict')
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name,
                                                                                                  self.opt.dataset,
                                                                                                  test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info(
                        'con_loss1: {:.4f}, con_loss2: {:.4f}, dep_loss: {:.4f}, loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(
                            contrastive_loss1.item(), contrastive_loss2.item(),
                            dep_loss.item(), loss.item(), train_acc, test_acc, f1))
        return max_test_acc, max_f1, model_path

    def _evaluate(self, show_results=False):

        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None

        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs, penal, _, _, _, _, _ = self.model(inputs)

                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()

        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1

        return test_acc, f1

    def _test(self):
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)

    def run(self):
        criterion = nn.CrossEntropyLoss()
        if 'bert' not in self.opt.model_name:
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        else:
            optimizer = self.get_bert_optimizer(self.model)
        max_test_acc_overall = 0
        max_f1_overall = 0
        if 'bert' not in self.opt.model_name:
            self._reset_params()
        max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        torch.save(self.best_model.state_dict(), model_path)
        logger.info('>> saved: {}'.format(model_path))
        logger.info('#' * 60)
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
        logger.info('max_f1_overall:{}'.format(max_f1_overall))
        self._test()


def main():
    model_classes = {
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'tclbert': TCLBertClassifier,
        'tcl': TCLClassifier
    }

    dataset_files = {
        'restaurant': {
            'train': './dataset/Restaurants_corenlp/train.json',
            'test': './dataset/Restaurants_corenlp/test.json',
        },
        'laptop': {
            'train': './dataset/Laptops_corenlp/train.json',
            'test': './dataset/Laptops_corenlp/test.json'
        },
        'twitter': {
            'train': './dataset/Tweets_corenlp/train.json',
            'test': './dataset/Tweets_corenlp/test.json',
        }
    }

    input_colses = {
        'atae_lstm': ['text', 'aspect'],
        'ian': ['text', 'aspect'],
        'tcl': ['text', 'aspect', 'pos', 'deprel', 'deptag', 'post', 'mask', 'length', 'pos_mask'],
        'tclbert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'src_mask', 'aspect_mask', 'deptag', 'pos_mask']
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }

    # Hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='tclbert', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='restaurant', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--deptag_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=50, help='GCN mem dim.')
    parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
    parser.add_argument('--polarities_dim', default=3, type=int, help='3')

    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
    parser.add_argument('--loop', default=True)

    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')

    parser.add_argument('--attention_heads', default=1, type=int, help='number of multi-attention heads')
    parser.add_argument('--max_length', default=100, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--vocab_dir', type=str, default='./dataset/Laptops_corenlp')
    parser.add_argument('--pad_id', default=0, type=int)
    parser.add_argument('--reshape', default=True, action='store_true', help='reshape relational tree')
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--alpha', default=0.25, type=float)  # orthogonal_loss
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--theta', default=1, type=float)
    parser.add_argument('--gamma', default=1, type=float)
    parser.add_argument('--temperature', default=0.1, type=float)

    # * bert
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--diff_lr', default=False, action='store_true')
    parser.add_argument('--bert_lr', default=2e-5, type=float)
    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    print("choice cuda:{}".format(opt.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(
        opt.device)

    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('./log'):
        os.makedirs('./log', mode=0o777)
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./log', log_file)))


    ins = Instructor(opt)

    ins.run()


if __name__ == '__main__':
    main()
