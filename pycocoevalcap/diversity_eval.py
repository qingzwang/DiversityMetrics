__author__ = 'Qignzhong'

import os
import json
import pickle
import numpy as np

from collections import defaultdict
from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from spice.spice import Spice
import nltk


class SelfCider:
    def __init__(self, pathToData, candName, num=10, dfMode = "coco-val-df"):
        """
        Reference file: list of dict('image_id': image_id, 'caption': caption).
        Candidate file: list of dict('image_id': image_id, 'caption': caption).
        :params: refName : name of the file containing references
        :params: candName: name of the file containing cnadidates
        """
        self.eval = {}
        # self._refName = refName
        self._candName = candName
        self._pathToData = pathToData
        self._dfMode = dfMode
        self._num = num
        if self._dfMode != 'corpus':
            with open('./data/coco-train2014-df.p', 'r') as f:
                self._df_file = pickle.load(f)

    def evaluate(self):
        """
        Load the sentences from json files
        """
        def readJson():
            path_to_cand_file = os.path.join(self._pathToData, self._candName)
            cand_list = json.loads(open(path_to_cand_file, 'r').read())

            res = defaultdict(list)

            for id_cap in cand_list:
                res[id_cap['image_id']].extend(id_cap['captions'])

            return res

        print 'Loading Data...'
        res = readJson()
        # res = {
        #     '0': [
        #         'a zebra standing in the forest',
        #         'a zebra standing near a tree in a field',
        #         'a zebra standing on a lush dry grass field',
        #         'a zebra standing on all four legs near trees and bushes with hills in the far distance',
        #         'a zebra is standing in the grass near a tree'
        #     ]
        # }
        # self._num=5
        ratio = {}
        avg_diversity = 0
        for im_id in res.keys():
            print ('number of images: %d\n')%(len(ratio))
            cov = np.zeros([self._num, self._num])
            for i in range(self._num):
                for j in range(i, self._num):
                    new_gts = {}
                    new_res = {}
                    # new_res[im_id] = [{'caption': res[im_id][i]}]
                    # new_gts[im_id] = [{'caption': res[im_id][j]}]
                    new_res[im_id] = [res[im_id][i]]
                    new_gts[im_id] = [res[im_id][j]]
                    # new_res[im_id] = ['a group of people are playing football on a grass covered field']
                    # new_gts[im_id] = ['a group of people are playing football on a grass covered field',
                    #                   'a group of people are watching a football match']
                    # new_gts[im_id] = gt
                    # =================================================
                    # Set up scorers
                    # =================================================
                    # print 'tokenization...'
                    # tokenizer = PTBTokenizer()
                    # new_gts = tokenizer.tokenize(new_gts)
                    # new_res = tokenizer.tokenize(new_res)

                    # =================================================
                    # Set up scorers
                    # =================================================
                    print 'setting up scorers...'
                    scorers = [
                        # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                        # (Meteor(), "METEOR"),
                        # (Rouge(), "ROUGE_L"),
                        (Cider(self._dfMode, self._df_file), "CIDEr"),
                        # (Spice(), "SPICE")
                    ]

                    # =================================================
                    # Compute scores
                    # =================================================
                    for scorer, method in scorers:
                        print 'computing %s score...'%(scorer.method())
                        score, scores = scorer.compute_score(new_gts, new_res)

                    cov[i, j] = score
                    cov[j, i] = cov[i, j]
            # np.save('log_att_x0_c1_d0.npy', cov)
            u, s, v = np.linalg.svd(cov)
            s_sqrt = np.sqrt(s)
            r = max(s_sqrt) / s_sqrt.sum()
            print('ratio=%.5f\n')%(-np.log10(r) / np.log10(self._num))
            ratio[im_id] = -np.log10(r) / np.log10(self._num)
            avg_diversity += -np.log10(r) / np.log10(self._num)
            if len(ratio) == 5000:
                break
        print('Average diversity: %.5f')%(avg_diversity / len(ratio))
        self.eval = ratio

    def setEval(self, score, method):
        self.eval[method] = score


class LSA:
    def __init__(self, pathToData, candName, num=10):
        self.eval = {}
        # self._refName = refName
        self._candName = candName
        self._pathToData = pathToData
        self._num = num

    def evaluate(self):
        """
        Load the sentences from json files
        """
        def readJson():
            path_to_cand_file = os.path.join(self._pathToData, self._candName)
            cand_list = json.loads(open(path_to_cand_file, 'r').read())

            res = defaultdict(list)

            for id_cap in cand_list:
                res[id_cap['image_id']].extend(id_cap['captions'])

            return res

        def get_vocab(captions):
            vocab = []
            for caption in captions:
                tokens = caption.split(' ')
                for token in tokens:
                    if token not in vocab:
                        vocab.append(token)
            return vocab

        def term_document(captions):
            vocab = get_vocab(captions)
            term_doc = np.zeros([len(vocab), len(captions)])
            for doc_id in range(len(captions)):
                caption = captions[doc_id]
                tokens = caption.split(' ')
                for token in tokens:
                    token_id = vocab.index(token)
                    term_doc[token_id, doc_id] += 1
            return term_doc

        print 'Loading Data...'
        res = readJson()
        ratio = {}
        avg_diversity = 0
        for im_id in res.keys():
            print ('number of images: %d\n')%(len(ratio))
            captions = res[im_id][:self._num]
            term_doc = term_document(captions)
            u, s, v = np.linalg.svd(term_doc)
            r = max(s) / s.sum()
            print('ratio=%.5f\n') % (r)
            ratio[im_id] = -np.log10(r) / np.log10(self._num)
            avg_diversity += -np.log10(r) / np.log10(self._num)
            if len(ratio) == 5000:
                break
        print ('Average diversity: %.5f')%(avg_diversity / len(ratio))
        self.eval = ratio


class mBLEU:
    def __init__(self, pathToData, candName, num=10, dfMode = "coco-val-df"):
        """
        Reference file: list of dict('image_id': image_id, 'caption': caption).
        Candidate file: list of dict('image_id': image_id, 'caption': caption).
        :params: refName : name of the file containing references
        :params: candName: name of the file containing cnadidates
        """
        self.eval = {}
        # self._refName = refName
        self._candName = candName
        self._pathToData = pathToData
        self._dfMode = dfMode
        self._num = num
        if self._dfMode != 'corpus':
            with open('./data/coco-train2014-df.p', 'r') as f:
                self._df_file = pickle.load(f)

    def evaluate(self):
        """
        Load the sentences from json files
        """
        def readJson():
            path_to_cand_file = os.path.join(self._pathToData, self._candName)
            cand_list = json.loads(open(path_to_cand_file, 'r').read())

            res = defaultdict(list)

            for id_cap in cand_list:
                res[id_cap['image_id']].extend(id_cap['captions'])

            return res

        print 'Loading Data...'
        res = readJson()
        ratio = {}
        avg_diversity = 0
        for im_id in res.keys():
            print ('number of images: %d\n')%(len(ratio))
            final_score = []
            for i in range(self._num):
                new_gts = {}
                new_res = {}
                new_res[im_id] = [res[im_id][i]]
                new_gts[im_id] = [res[im_id][j] for j in range(self._num) if j != i]
                # =================================================
                # Set up scorers
                # =================================================
                print 'setting up scorers...'
                scorers = [
                    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                    # (Meteor(), "METEOR"),
                    # (Rouge(), "ROUGE_L"),
                    # (Cider(self._dfMode, self._df_file), "CIDEr"),
                    # (Spice(), "SPICE")
                ]

                # =================================================
                # Compute scores
                # =================================================
                for scorer, method in scorers:
                    print 'computing %s score...'%(scorer.method())
                    score, scores = scorer.compute_score(gts=new_gts, res=new_res)
                final_score.append(score)
            mbleus = np.array(final_score).sum(0) / self._num
            ratio[im_id] = list(mbleus)
            avg_diversity += sum(mbleus) / 4
            if len(ratio) == 5000:
                break
        print('Average diversity: %.5f')%(avg_diversity / len(ratio))
        self.eval = ratio

    def setEval(self, score, method):
        self.eval[method] = score


class Accurate:
    def __init__(self, pathToData, refName, candName, model=None, num=10, dfMode = "corpus"):
        """
        Reference file: list of dict('image_id': image_id, 'caption': caption).
        Candidate file: list of dict('image_id': image_id, 'caption': caption).
        :params: refName : name of the file containing references
        :params: candName: name of the file containing cnadidates
        """
        self.eval = {}
        self._refName = refName
        self._candName = candName
        self._pathToData = pathToData
        self._dfMode = dfMode
        self._num = num
        if self._dfMode != 'corpus':
            with open('./data/coco-train2014-df.p', 'r') as f:
                self._df_file = pickle.load(f)

    def evaluate(self):
        """
        Load the sentences from json files
        """
        def readJson(refName, candName):

            path_to_ref_file = os.path.join(refName)
            path_to_cand_file = os.path.join(self._pathToData, candName)

            ref_list = json.loads(open(path_to_ref_file, 'r').read())
            cand_list = json.loads(open(path_to_cand_file, 'r').read())
            ref_list = ref_list['annotations']

            res = defaultdict(list)

            for id_cap in cand_list:
                res[id_cap['image_id']].extend(id_cap['captions'])

            gts = defaultdict(list)

            # change of naming convention from ref to gts
            for l in ref_list:
                gts[l['image_id']].append({"caption": l['caption']})

            return gts, res

        print 'Loading Data...'
        gts, res = readJson(self._refName, self._candName)
        # =================================================
        # Set up scorers
        # =================================================
        accuracy = {}
        tokenizer = PTBTokenizer()
        token_gts = tokenizer.tokenize(gts)
        avg_cider = 0
        for image_id in res.keys():
            accuracy[image_id] = []
            new_gts = {image_id: token_gts[image_id]}
            # new_gts = {"0": [
            #     "zebras stand on grassy field with trees",
            #     "zebras stand on grassy field"
            #     # "zebras gazing"
            # ]
            # }
            for i in range(self._num):
                new_res = {image_id: [res[image_id][i]]}
                # new_res = {"0": ["grassy field with trees"]}
                scorers = [
                    # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                    # (Meteor(),"METEOR"),
                    # (Rouge(), "ROUGE_L"),
                    (Cider(self._dfMode, self._df_file), "CIDEr"),
                    # (Spice(), "SPICE")
                ]

                # =================================================
                # Compute scores
                # =================================================
                for scorer, method in scorers:
                    print 'computing %s score...'%(scorer.method())
                    score, scores = scorer.compute_score(new_gts, new_res)
                accuracy[image_id].append(score)
            avg_cider += np.mean(accuracy[image_id])
        self.eval = accuracy
        print ('Average Cider: %.5f')%(avg_cider / len(accuracy))

    def setEval(self, score, method):
        self.eval[method] = score

