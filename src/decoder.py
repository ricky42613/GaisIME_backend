import kenlm
import heapq
from cryptography.fernet import Fernet
import dill
from biasLM import biasLM
import json

class Path:

    def __init__(self,):
        self.s0 = None
        self.s1 = None
        self.hist = None
        self.accum = 0.0
        self.score = 0.0
        self.curIdx = 0
    

    def __iter__(self,):
        return iter((self.s0, self.s1, self.hist, self.accum, self.score, self.curIdx))
    

    def __repr__(self,):
        return repr((self.s0, self.s1, self.hist, self.accum, self.score, self.curIdx))
    

    def __lt__(self, other):
        return self.score < other.score


class Decoder:
    
    def __init__(self,baseDir='data/', lexiPath = "lexicon.txt", baseLmPath = "biglm.bin", \
                        biasLmPath="bias.lm", salt=None):
        self.salt = salt
        self.baseDir = baseDir
        self.maxWordLen = 1
        self.maxTermLen = 4
        self.maxEmojiLen = 10
        self.lexicon = self.load_lexicon(lexiPath)
        self.lm = self.load_base_language_model(baseLmPath)
        self.extlm = self.load_bias_language_model(biasLmPath)
        with open(self.baseDir + 'marks.json', "r", encoding='utf-8') as mf:
            self.markList = json.load(mf)

    
    def load_lexicon(self, lexiPath):
        """
        Return a dictionary where its keys are readings and values
        are lists of words having the identical readings
        Each line of the file lexiPath has a word and (one of) its reading(s)
        The words having more than self.maxWordLen characters will be ignored
        """
        
        # retDict = dict()
        # cfilter = {}
        # fp = open(lexiPath, "r",encoding='utf-8')
        # for line in fp:
        #     word, reading = line.strip().split(maxsplit=1)
        #     if len(word) != self.maxWordLen:
        #         continue
        #     cfilter[word] = True
        #     if reading in retDict:
        #         retDict[reading].add(word)
        #     else:
        #         retDict[reading] = {word}
        # fp.close()

        # for key in retDict:
        #     retDict[key] = list(retDict[key])
        # print("There are %d different chinese words in lexicon" % len(cfilter))
        with open(lexiPath,'r',encoding='utf-8') as f:
            retDict = json.load(f)
        return retDict
    

    def load_base_language_model(self, baseLmPath):
        """
        Load base language model from arpa or its binary format
        Currently using KenLM as our language model kernel
        """
        retLM = kenlm.LanguageModel(baseLmPath)
        return retLM


    def load_bias_language_model(self, biasLmPath):
        """
        Load bias language model if it exists; otherwise,
        create an new instance
        """
        try:
            key = Fernet(self.salt)
            with open(biasLmPath, "rb") as fp:
                saltedBin = fp.read()

            return dill.loads(key.decrypt(saltedBin))

        except:
            print('bias lm not existed')
            return biasLM(baseDir=self.baseDir)


    def build_decoding_graph(self, bpmSeq):
        """
        Return a adjancency list, where keys are the current
        indices and values are the lists of all the possible
        words 
        """
        retGraph = list()
        bpmSeq = bpmSeq.split()
        seqLen = len(bpmSeq)
        for i in range(seqLen):
            words = []
            for j in range(i, i + self.maxWordLen):
                if j >= seqLen:
                    break
                curKey = " ".join(bpmSeq[i : j+1])
                if curKey in self.lexicon:
                    words += [w+curKey for w in self.lexicon[curKey]]
            if not words:
                # print("Error: Invalid bopomo sequence @index %d: %s" %(i, bpmSeq[i]))
                return []
            retGraph.append(words)

        return retGraph


    # def beam_search_decoder(self, decGraph, beamWidth = 16):
    #     """
    #     Decode with beam search algorithm (heuristic search)
    #     The greater beamWidth is, the greater probability to
    #     find global optimal solution
    #     """
        
    #     minHeap = list()
    #     slen = len(decGraph)
    #     for word in decGraph[0]:
    #         path = Path()
    #         path.s0 = kenlm.State()
    #         path.s1 = kenlm.State()
    #         path.hist = [word]
    #         self.lm.BeginSentenceWrite(path.s0)
    #         baseScore = -self.lm.BaseScore(path.s0, word, path.s1)
    #         path.s0, path.s1 = path.s1, path.s0
    #         path.curIdx += len(word)
    #         path.accum = baseScore
    #         path.score = path.accum #/ path.curIdx
    #         heapq.heappush(minHeap, path)
        
    #     finished = False
    #     while not finished:
    #         finished = True
    #         newHeap = list()
    #         for i in range(beamWidth):
    #             if not minHeap:
    #                 break
    #             curPath = heapq.heappop(minHeap)
                
    #             if curPath.curIdx == slen:
    #                 heapq.heappush(newHeap, curPath)
    #                 continue
                
    #             for word in decGraph[curPath.curIdx]:
    #                 finished = False
    #                 path = Path()
    #                 path.s0 = curPath.s0.__copy__()
    #                 path.s1 = curPath.s1.__copy__()
    #                 path.hist = curPath.hist.copy()
    #                 path.accum = curPath.accum
    #                 path.curIdx = curPath.curIdx
    #                 path.hist.append(word)
    #                 baseScore = -self.lm.BaseScore(path.s0, word, path.s1)
    #                 path.s0, path.s1 = path.s1, path.s0
    #                 path.curIdx += len(word)
    #                 path.accum += baseScore
    #                 path.score = path.accum #/ path.curIdx
    #                 heapq.heappush(newHeap, path)

    #         minHeap = newHeap

    #     ret = list()
    #     while minHeap:
    #         curPath = heapq.heappop(minHeap)
    #         ret.append((curPath.score, " ".join(curPath.hist)))

    #     return ret


    def interpolate_beam_search_decoder(self, decGraph, beamWidth = 16, 
                                            alpha = .4, beta = .6):
        """
        Decode with interpolatedbeam search algorithm
        The greater beamWidth is, the greater probability to
        find global optimal solution
        """
        minHeap = list()
        slen = len(decGraph)
        for word in decGraph[0]:
            path = Path()
            path.s0 = kenlm.State()
            path.s1 = kenlm.State()
            path.hist = [word]
            self.lm.BeginSentenceWrite(path.s0)
            baseScore = -self.lm.BaseScore(path.s0, word, path.s1)
            # extScore = self.extlm.get_word_prob(path.hist, baseScore)
            scoreBias, match_order = self.extlm.get_word_prob(path.hist, baseScore)
            path.s0, path.s1 = path.s1, path.s0
            path.curIdx += 1 # len(word)
            # path.accum = baseScore if extScore == 0 else min(alpha * baseScore + beta * extScore, baseScore) 
            path.accum += scoreBias * baseScore
            path.score = path.accum #/ path.curIdx
            heapq.heappush(minHeap, path)
        
        finished = False
        while not finished:
            finished = True
            newHeap = list()
            for i in range(beamWidth):
                if not minHeap:
                    break
                curPath = heapq.heappop(minHeap)
                
                if curPath.curIdx == slen:
                    curPath.accum += -1*self.lm.BaseScore(curPath.s0, "</s>", curPath.s1)
                    curPath.score = curPath.accum 

                    heapq.heappush(newHeap, curPath)
                    # heapq.heappush(newHeap, curPath)
                    continue
                
                for word in decGraph[curPath.curIdx]:
                    finished = False
                    path = Path()
                    path.s0 = curPath.s0.__copy__()
                    path.s1 = curPath.s1.__copy__()
                    path.hist = curPath.hist.copy()
                    path.accum = curPath.accum
                    path.curIdx = curPath.curIdx
                    path.hist.append(word)
                    baseScore = -self.lm.BaseScore(path.s0, word, path.s1)
                    # extScore = self.extlm.get_word_prob(path.hist, baseScore)
                    scoreBias, match_order = self.extlm.get_word_prob(path.hist)
                    path.s0, path.s1 = path.s1, path.s0
                    path.curIdx += 1 # len(word)
                    # path.accum += baseScore if extScore == 0 else min(alpha * baseScore + beta * extScore, baseScore)
                    path.accum += scoreBias * baseScore
                    path.score = path.accum #/ path.curIdx
                    heapq.heappush(newHeap, path)
                    # 18.97491091489792
                    # 18.822440707683562

            minHeap = newHeap

        ret = list()
        while minHeap:
            curPath = heapq.heappop(minHeap)
            ret.append((curPath.score, " ".join([w[0] for w in curPath.hist])))
        return ret


    def decode(self, bpmSeq):
        # bgTime = timer()
        decGraph = self.build_decoding_graph(bpmSeq)
        if len(decGraph) == 0:
            return []
        # results = self.beam_search_decoder(decGraph, beamWidth = 16)
        results = self.interpolate_beam_search_decoder(decGraph, beamWidth = 64)
        # edTime = timer()
        return results

    def is_mark(self, target):
        """
        判斷target是否為符號
        """
        if target in self.markList:
            return True
        else:
            return False

    def list_mark(self, mark):
        """
        透過現有符號列出所有候選符號
        """
        return self.markList[mark]

    def list_candidate(self,sound):
        if sound in self.lexicon:
            return self.lexicon[sound]
        else:
            return []
        
