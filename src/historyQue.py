from datetime import datetime

class DecodeHistory:
    def __init__(self, maxsize=10, tolerate_second=2):
        self.historyQue = []
        self.size = 0
        self.max_size = maxsize
        self.tolerate_second = tolerate_second
    
    def push(self, textSeq, chewingSeq):
        #check if you are exceeding the limit
        if self.size == self.max_size:
            self.historyQue.pop(-1)
            self.size =  self.size - 1
        self.historyQue= [{'text': textSeq, 'chewingSeq': chewingSeq, 'time': datetime.now()}] + self.historyQue
        self.size = self.size + 1
    
    def get_valid_history(self):
        current = datetime.now()
        ret = []
        for i in range(0, self.size):
            history = self.historyQue[i]
            delta = current - history['time']
            second_diff = delta.total_seconds()
            if second_diff < self.tolerate_second:
                ret.append(self.historyQue[i])
        return ret
        
            