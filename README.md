# A Streaming Approach For Efficient Batched Beam Search

This repo contains an implementation of the syntactic parsing experiments for the EMNLP 2020 paper "A Streaming Approach For Efficient Batched Beam Search" by Kevin Yang, Violet Yao, John DeNero, and Dan Klein (https://arxiv.org/abs/2010.02164). For the code for machine translation and semantic parsing experiments see https://github.com/yangkevin2/emnlp2020-stream-beam-mt and https://github.com/yangkevin2/emnlp2020-stream-beam-semantic. 

## Setup

This repo is forked from https://github.com/jhcross/span-parser and reimplements in PyTorch. See their original repo for setup instructions; additionally, install PyTorch. 

## Example Command

Run the following command from the `src` directory to benchmark each method in the paper once, including the FIFO version discussed in the appendix, in series. Prints the oracle reranking performance for each method along with time and efficiency information.

`python main.py --vocab ../data/vocabulary.json --test ../data/23.auto.clean --model ../data/model --device cpu --batch-size 10 --beam-size 10 --ap 2.5 --mc 3 --preds_dir ../preds`

We provide the pretrained model in `data/model` since we found reproducing the original results in pytorch to be a bit finicky and highly initialization-dependent. Switch to `--device cuda` for gpu training.