import argparse

from phrase_tree import PhraseTree, FScore

parser = argparse.ArgumentParser()
parser.add_argument('--gold', type=str)
parser.add_argument('--pred', type=str)
args = parser.parse_args()

gold_trees = PhraseTree.load_treefile(args.gold)
pred_trees = PhraseTree.load_treefile(args.pred)

accuracy = FScore()
for gold, pred in zip(gold_trees, pred_trees):
    local_accuracy = pred.compare(gold)
    accuracy += local_accuracy

print(accuracy)