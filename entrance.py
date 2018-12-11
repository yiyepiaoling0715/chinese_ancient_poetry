import argparse
from train import train
from infer import generate_control
from char2vec import Char2Vec
from char_dict import CharDict
from poems import Poems
from data_utils import batch_train_data
from rank_words import RankedWords
if __name__=='__main__':
    arguementparser=argparse.ArgumentParser(description='chinese poem generation')
    arguementparser.add_argument('-t',action='store_true',dest='train',default=False)
    arguementparser.add_argument('-p',action='store_true',dest='pretrain',default=False)
    arguementparser.add_argument('-i',action='store_true',dest='infer',default=False)
    # arguementparser.add_argument('-p', dest = 'planner', default = False,action = 'store_true',
    #                              help = 'train planning model')
    args=arguementparser.parse_args()
    # print('args==>',args)
    if args.train:
        print('进入训练阶段')
        train(n_epochs=1000)
    elif args.pretrain:
        print('进入预训练阶段')
        CharDict()
        RankedWords()
        Char2Vec()
        Poems()
        batch_train_data(32)
    elif args.infer:
        print('进入测试阶段')
        generate_control()