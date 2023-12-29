from jieba import Tokenizer
from pygtrie import StringTrie
from boltons.fileutils import atomic_save
from cryptography.fernet import Fernet
import dill


class biasLM:

    def __init__(self,baseDir='data/', maxOrder=7, m=-2.5, b=7):

        self.maxOrder = maxOrder
        self.ngramTrie = StringTrie()
        self.jiebaTokenizer = Tokenizer()

        self.jiebaTokenizer.set_dictionary(baseDir+"merge.4jieba.default.dict")
        self.jiebaTokenizer.load_userdict(baseDir+"merge.4jieba.extra.dict")

        self.set_bias_params(m, b)
        self.pruneCnter = 0
        self.pruneTHold = 1000
        self.pruneRatio = 0.005

        
    def set_bias_params(self, m, b):
        """
        These two params describe a length-linear func
        which is a decreasing function of N (m<0), the 
        maximum matched n-gram order.      
        """
        self.m = m
        self.b = b
    

    def adapt_sentence(self, sent, chewing, errIdxs):
        """
        This function should be called only if a sentence
        is complete and is modified by user
        """
        # sToks = [word for word in self.jiebaTokenizer.lcut(sent.strip(), HMM=False) if not word.isspace()]
        sToks = [w+chew for w, chew in zip(list(sent), chewing.split())]
        sToks = ['<s>'] + sToks + ['</s>']
        sToks.reverse()
        n = len(sToks)
        errIdxs = [n-i-2 for i in errIdxs]

        for order in range(self.maxOrder):
            
            borderIdx = max((len(sToks)-order), 0)

            for startIdx in range(borderIdx):
                # Iterate through all possible n-grams
                curKey = "/".join(sToks[startIdx : startIdx+order+1])
                containErr = False
                for i in errIdxs:
                    if i >= startIdx and i < startIdx+order+1:
                        containErr = True
                        break
                if not containErr:
                    continue 
                # Update adaption count if n-gram exists
                if self.ngramTrie.has_key(curKey):

                    temp = self.ngramTrie[curKey]
                    if temp['matchOrder'] > 1:
                        # temp['adpCnt'] += 1
                        temp['adpCnt'] = 1 if temp['adpCnt'] == 0 else temp['adpCnt'] * 2
                        self.ngramTrie[curKey] = temp

                # Insert n-gram if n-gram doesn't exist
                else:
                    self.ngramTrie[curKey] = {

                        'matchOrder' : order+1,
                        'adpCnt' : 0,
                        'useCnt' : 0
                    }
    

    def log_sentence(self, sent, chewing):
        """
        This should be called only if the sentence is complete and
        without user modification
        """
        sToks = [word for word in self.jiebaTokenizer.lcut(sent.strip(), HMM=False) if not word.isspace()]
        chewingList = chewing.split()
        for i in range(len(sToks)):
            term = sToks[i]
            spell = ''.join(chewingList[:len(term)])
            chewingList = chewingList[len(term):]
            sToks[i] = term+spell
        sToks = ['<s>'] + sToks + ['</s>']
        sToks.reverse()

        flagInTrie = False

        for order in range(self.maxOrder):
            # Iterate throught all possible n-grams
            borderIdx = max((len(sToks)-order), 0)

            for startIdx in range(borderIdx):

                curKey = "/".join(sToks[startIdx : startIdx+order+1])
                
                if self.ngramTrie.has_key(curKey):
                    
                    temp = self.ngramTrie[curKey]
                    if temp['matchOrder'] > 1:
                        temp['useCnt'] += 1
                        self.ngramTrie[curKey] = temp
                        flagInTrie = True
                
        # Do nothing if first-pass decoding result
        # is correct already
        if not flagInTrie:
            return

        self.pruneCnter += 1

        if self.pruneCnter == self.pruneTHold:
            # Prune those n-gram with useCnt less than 
            # (pruneTHold * pruneRatio) 
            self.prune_least_use()
        
    
    def prune_least_use(self):

        cutTHold = int(self.pruneTHold * self.pruneRatio)

        for curKey in self.ngramTrie.iterkeys():

            # If the key has been popped, 
            # continue to avoid key error
            if not self.ngramTrie.has_key(curKey):
                continue
            
            temp = self.ngramTrie[curKey]
            if temp['matchOrder'] > 1 and temp['useCnt'] < cutTHold:
                self.ngramTrie.pop(curKey, default=None)
            
            elif temp['matchOrder'] > 1:
                # Reset the countings
                temp['useCnt'] = 0
                self.ngramTrie[curKey] = temp
            
        self.pruneCnter = 0

    
    def get_word_prob(self, wseq, useSentBorder=True):

        wseq = wseq.copy()
        if useSentBorder:
            wseq = ['<s>'] + wseq
        wseq.reverse()
        matchOrder = 0
        matchKey = ''
        adpCnt = 0

        for idx in range(min(len(wseq), self.maxOrder)):
            curKey = "/".join(wseq[ : idx+1])
            if self.ngramTrie.has_key(curKey):
                matchKey = curKey
                temp = self.ngramTrie[curKey]
                matchOrder = temp['matchOrder']
                adpCnt = temp['adpCnt']
            else:
                break
        w_sum = adpCnt * 1 + 2 * (matchOrder) if matchOrder > 1 else 0
        return 2**(-0.5*w_sum), matchOrder
        # return (self.m + adpCnt*0.06) * matchOrder + self.b if matchOrder > 0 else 0
    

    def disk_backup(self, salt, writePath="fst/bias.lm"):
        
        f = Fernet(salt)
        with atomic_save(writePath) as fp:
            fp.write(f.encrypt(dill.dumps(self)))


