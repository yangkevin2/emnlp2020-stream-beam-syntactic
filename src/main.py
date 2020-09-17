"""
Command-line interface for Span-Based Constituency Parser.
"""

import random
import sys
import argparse
import time
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from phrase_tree import PhraseTree, FScore
from features import FeatureMapper
from parser import Parser


def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)


def train(
    feature_mapper,
    word_dims,
    tag_dims,
    lstm_units,
    hidden_units,
    epochs,
    batch_size,
    train_data_file,
    dev_data_file,
    model_save_file,
    droprate,
    unk_param,
    lr,
    lr_decay,
    alpha=1.0,
    beta=0.0,
    device='cuda',
):

    start_time = time.time()

    fm = feature_mapper
    word_count = fm.total_words()
    tag_count = fm.total_tags()

    network = Network(
        word_count=word_count,
        tag_count=tag_count,
        word_dims=word_dims,
        tag_dims=tag_dims,
        lstm_units=lstm_units,
        hidden_units=hidden_units,
        struct_out=2,
        label_out=fm.total_label_actions(),
        droprate=droprate,
        device=device,
    )
    for param in network.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.uniform_(param, -0.01, 0.01)
    if args.load_npz_params is not None:
        network.load_dynet(args.load_npz_params)
        print('loaded npz params')
    network = network.to(device)
    print('num parameters', sum([p.numel() for p in network.parameters() if p.requires_grad]))

    # optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    optimizer = torch.optim.Adadelta(network.parameters(), rho=0.99, eps=1e-7)
    crit = nn.CrossEntropyLoss(reduction='sum')

    print('Hidden units: {},  per-LSTM units: {}'.format(
        hidden_units,
        lstm_units,
    ))
    print('Embeddings: word={}  tag={}'.format(
        (word_count, word_dims),
        (tag_count, tag_dims),
    ))
    print('Dropout rate: {}'.format(droprate))
    print('Random UNKing parameter z = {}'.format(unk_param))
    print('Exploration: alpha={} beta={}'.format(alpha, beta))

    training_data = fm.gold_data_from_file(train_data_file)
    num_batches = -(-len(training_data) // batch_size) 
    print('Loaded {} training sentences ({} batches of size {})!'.format(
        len(training_data),
        num_batches,
        batch_size,
    ))
    parse_every = -(-num_batches // 4)

    dev_trees = PhraseTree.load_treefile(dev_data_file)
    print('Loaded {} validation trees!'.format(len(dev_trees)))

    best_acc = FScore()

    np.random.seed(0)
    for epoch in range(1, epochs + 1):
        print('........... epoch {} ...........'.format(epoch))

        total_cost = 0.0
        total_states = 0
        training_acc = FScore()

        np.random.shuffle(training_data)

        for b in range(num_batches):
            batch = training_data[(b * batch_size) : ((b + 1) * batch_size)]
            network.eval()
            with torch.no_grad():
                explore = [
                    Parser.exploration(
                        example,
                        fm,
                        network,
                        alpha=alpha,
                        beta=beta,
                    ) for example in batch
                ]
                for (_, acc) in explore:
                    training_acc += acc

            batch = [example for (example, _) in explore]

            errors = []
            network.train()

            for example in batch: # TODO batch this?

                ## random UNKing ##
                for (i, w) in enumerate(example['w']):
                    if w <= 2:
                        continue

                    freq = fm.word_freq_list[w]
                    drop_prob = unk_param / (unk_param + freq)
                    r = np.random.random()
                    if r < drop_prob:
                        example['w'][i] = 0

            batch_struct_lefts, batch_struct_rights, batch_struct_correct = [], [], []
            batch_label_lefts, batch_label_rights, batch_label_correct = [], [], []
            example_select = []
            for i, example in enumerate(batch): # TODO batch this?
                for (left, right), correct in list(example['struct_data'].items()):
                    batch_struct_lefts.append(left)
                    batch_struct_rights.append(right)
                    batch_struct_correct.append(correct)
                    example_select.append(i)
                
                for (left, right), correct in list(example['label_data'].items()):
                    batch_label_lefts.append(left)
                    batch_label_rights.append(right)
                    batch_label_correct.append(correct)

            lengths = torch.LongTensor([len(example['w']) for example in batch]).to(network.device)
            batch_w = pad_sequence([torch.from_numpy(example['w']) for example in batch]).long().to(network.device)
            batch_t = pad_sequence([torch.from_numpy(example['t']) for example in batch]).long().to(network.device)
            lengths, batch_w, batch_t = lengths[example_select], batch_w[:, example_select], batch_t[:, example_select] # multiple targets per source

            batch_struct_lefts, batch_struct_rights, batch_struct_correct, batch_label_lefts, batch_label_rights, batch_label_correct = \
                    torch.LongTensor(batch_struct_lefts).to(network.device), torch.LongTensor(batch_struct_rights).to(network.device), \
                    torch.LongTensor(batch_struct_correct).to(network.device), torch.LongTensor(batch_label_lefts).to(network.device), \
                    torch.LongTensor(batch_label_rights).to(network.device), torch.LongTensor(batch_label_correct).to(network.device)
            
            fwd, back = network.evaluate_recurrent(batch_w, batch_t, lengths)
            struct_scores = network.evaluate_struct(fwd, back, batch_struct_lefts, batch_struct_rights)
            loss = crit(struct_scores, batch_struct_correct)
            errors.append(loss)
            total_states += sum([len(example['struct_data']) for example in batch])
            label_scores = network.evaluate_label(fwd, back, batch_label_lefts, batch_label_rights)
            loss = crit(label_scores, batch_label_correct)
            errors.append(loss)
            total_states += sum([len(example['label_data']) for example in batch])

            batch_error = sum(errors)
            total_cost += batch_error.item()
            optimizer.zero_grad()
            batch_error.backward()
            torch.nn.utils.clip_grad_norm(network.parameters(), 5.0)
            optimizer.step()

            mean_cost = total_cost / total_states

            print(
                '\rBatch {}  Mean Cost {:.4f} [Train: {}]'.format(
                    b,
                    mean_cost,
                    training_acc,
                ),
                end='',
            )
            sys.stdout.flush()

            if ((b + 1) % parse_every) == 0 or b == (num_batches - 1):
                network.eval()
                with torch.no_grad():
                    dev_acc = Parser.evaluate_corpus(
                        dev_trees,
                        fm,
                        network,
                        batch_size,
                        k=1,
                        ap=0, 
                        mc=1
                    )
                    print('  [Val: {}]'.format(dev_acc))

                    if dev_acc.fscore() > best_acc.fscore():
                        best_acc = dev_acc 
                        save_checkpoint(network.state_dict(), model_save_file)
                        print('    [saved model: {}]'.format(model_save_file)) 

        for param in optimizer.param_groups:
            param['lr'] *= lr_decay
        current_time = time.time()
        runmins = (current_time - start_time)/60.
        print('  Elapsed time: {:.2f}m'.format(runmins))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Span-Based Constituency Parser')
    parser.add_argument(
        '--l2',
        help='L2 regularization parameter. (DEFAULT=0)',
        default=0,
    )
    parser.add_argument(
        '--seed',
        help='Seed for RNG. (DEFAULT=0 : generate)',
        default=0,
    )
    parser.add_argument(
        '--model',
        help='File to save or load model.',
    )
    parser.add_argument(
        '--train',
        default='../data/debug.clean',
        help='Training trees. PTB (parenthetical) format.',
    )
    parser.add_argument(
        '--test',
        help=(
            'Evaluation trees. PTB (parenthetical) format.'
            ' Omit for training.'
        ),
    )
    parser.add_argument(
        '--dev', 
        default='../data/debug.clean',
        help=(
            'Validation trees. PTB (parenthetical) format.'
            ' Required for training'
        ),
    )
    parser.add_argument(
        '--vocab',
        help='JSON file from which to load vocabulary.',
    )
    parser.add_argument(
        '--vocab-output',
        help='Destination to save vocabulary from training data.',
    )
    parser.add_argument(
        '--preds_dir',
        help='Dir to save preds.',
    )
    parser.add_argument(
        '--word-dims',
        type=int,
        default=50,
        help='Embedding dimesions for word forms. (DEFAULT=50)',
    )
    parser.add_argument(
        '--tag-dims',
        type=int,
        default=20,
        help='Embedding dimesions for POS tags. (DEFAULT=20)',
    )
    parser.add_argument(
        '--lstm-units',
        type=int,
        default=200,
        help='Number of LSTM units in each layer/direction. (DEFAULT=200)',
    )
    parser.add_argument(
        '--hidden-units',
        type=int,
        default=200,
        help='Number of hidden units for each FC ReLU layer. (DEFAULT=200)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs. (DEFAULT=10)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of sentences per training update. (DEFAULT=10)',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout probability. (DEFAULT=0.5)',
    )
    parser.add_argument(
        '--unk-param',
        type=float,
        default=0.8375,
        help='Parameter z for random UNKing. (DEFAULT=0.8375)',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Softmax distribution weighting for exploration. (DEFAULT=1.0)',
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0,
        help='Probability of using oracle action in exploration. (DEFAULT=0)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate',
    )
    parser.add_argument(
        '--lr-decay',
        type=float,
        default=1,
        help='learning rate decay per epoch',
    )
    parser.add_argument(
        '--device',
        help='cuda or cpu',
    )
    parser.add_argument(
        '--beam-size',
        type=int,
        default=1,
        help='Beam size for decoding. (DEFAULT=5)',
    )
    parser.add_argument(
        '--ap',
        type=float,
        default=0,
        help='ap threshold for beam search',
    )
    parser.add_argument(
        '--mc',
        type=int,
        default=1,
        help='max candidates threshold for beam search',
    )
    parser.add_argument(
        '--load_npz_params',
        type=str,
        default=None,
        help='path to load npz params from',
    )

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.vocab is not None:
        from features import FeatureMapper
        fm = FeatureMapper.load_json(args.vocab)
    elif args.train is not None:
        from features import FeatureMapper
        fm = FeatureMapper(args.train)    
        if args.vocab_output is not None:
            fm.save_json(args.vocab_output)
            print('Wrote vocabulary file {}'.format(args.vocab_output))
            sys.exit()
    else:
        print('Must specify either --vocab or --train-data.')
        print('    (Use -h or --help flag for full option list.)')
        sys.exit()

    if args.model is None:
        print('Must specify --model or (or --vocab-output) parameter.')
        print('    (Use -h or --help flag for full option list.)')
        sys.exit()

    if args.test is not None:
        from phrase_tree import PhraseTree
        from network import Network
        from parser import Parser

        test_trees = PhraseTree.load_treefile(args.test)
        print('Loaded test trees from {}'.format(args.test))
        network = Network(
            word_count=fm.total_words(),
            tag_count=fm.total_tags(),
            word_dims=args.word_dims,
            tag_dims=args.tag_dims,
            lstm_units=args.lstm_units,
            hidden_units=args.hidden_units,
            struct_out=2,
            label_out=fm.total_label_actions(),
            droprate=args.dropout,
            device=args.device,
        )
        checkpoint = torch.load(args.model, map_location=args.device)
        network.load_state_dict(checkpoint)
        print('Loaded model from: {}'.format(args.model))
        network.eval()
        network = network.to(args.device)
        for method in ['greedy', 'beam', 'variable_beam', 'variable_beam_stream_fifo', 'variable_beam_stream_main']:
            print(method)
            print(datetime.datetime.now())
            with torch.no_grad():
                accuracy = Parser.evaluate_corpus(test_trees, fm, network, args.batch_size, k=args.beam_size, ap=args.ap, mc=args.mc, method=method, write_dir=args.preds_dir)
            print(datetime.datetime.now())
            print('Accuracy: {}'.format(accuracy))
    elif args.train is not None:
        from network import Network

        print('L2 regularization: {}'.format(args.l2))

        train(
            feature_mapper=fm,
            word_dims=args.word_dims,
            tag_dims=args.tag_dims,
            lstm_units=args.lstm_units,
            hidden_units=args.hidden_units,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_data_file=args.train,
            dev_data_file=args.dev,
            model_save_file=args.model,
            droprate=args.dropout,
            unk_param=args.unk_param,
            lr=args.lr,
            lr_decay=args.lr_decay,
            alpha=args.alpha,
            beta=args.beta,
            device=args.device,
        )


