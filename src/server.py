# -*- coding: utf-8 -*-
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import os
import json
from decoder import Decoder
import sentry_sdk
from historyQue import DecodeHistory
import re
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
# from flask import Flask, request
sentry_sdk.init(
    "https://b22ff47578e940d8a9de5cad7fa41717@o823009.ingest.sentry.io/5867223",
    traces_sample_rate=1.0
)

dirPath = sys.argv[1]
lexiPath = dirPath + 'lexicon.json'
baseLM = dirPath + 'finetune.trie'
biasFile = os.path.expanduser('~') + "/Library/Input Methods/.noDeleteQQ.bin"
# single_word_file = dir_path+"words.json"
salt = 'psSAFaNayPTpZ5bc9O0GsLWc_TCgI0r9_cktMZVg53k='
decoder = Decoder(baseDir=dirPath,lexiPath=lexiPath,biasLmPath=biasFile,baseLmPath=baseLM, salt=salt)
errLog = os.path.expanduser('~') + "/Library/Input Methods/.errSent.log"
with open(dirPath + 'emoji.json', 'r', encoding="utf-8") as f:
    emoTab = json.load(f)
    checkEmoTab = {}
    for s in emoTab:
        for e in emoTab[s]:
            checkEmoTab[e] = 1
app = FastAPI()
historyController = DecodeHistory()
# app = Flask(__name__)

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5' or len(re.findall('\W', _char)) > 0:
            return False
    return True

def sent_segmentation(sent, spell, errCorrects):
    sents = []
    spellList = spell.split(" ")
    wordList = list(sent)
    if len(spellList) != len(wordList):
        return []
    tmp = []

    for i in range(0, len(wordList)):
        if is_all_chinese(wordList[i]):
            if str(i+1) in errCorrects:
                tmp.append((wordList[i], spellList[i], 1))
            else:
                tmp.append((wordList[i], spellList[i], 0))
            if i == len(wordList) - 1:
                sents.append(tmp)
                tmp = []
        else:
            if len(tmp) > 0:
                sents.append(tmp)
                tmp = []
    return sents


def manual_error_correct(oriSent, errCorrections, composeRst):
    for idx in errCorrections:
        if idx != None:
            composeRst[int(idx) - 1] = oriSent[int(idx) - 1]
    return composeRst


@app.get('/',response_class=PlainTextResponse)
def main():
    return "敬請期待Q_Q"

@app.get('/test',response_class=PlainTextResponse)
def handle_decode(chewing_seq:str):
    decode_rst = decoder.decode(bpmSeq=chewing_seq)
    if len(decode_rst) == 0:
        return "error"
    compose_str = "".join(decode_rst[0][1].replace(' ',''))
    return compose_str


@app.get('/decode',response_class=PlainTextResponse)
def handle_decode(chewing_seq:str,insertidx:str,sent:str,correct_idx:str):
    chewingSeq = chewing_seq
    insertIdx = int(insertidx)
    oriSent = sent
    # correct_idx = request.args.get('correct_idx')
    if oriSent == "none":
        oriSent = ""
    oriSentList = list(oriSent)
    oriSentList.insert(insertIdx, "")
    errCorrections = []
    if correct_idx.lower() != "none":
        errCorrections = correct_idx.split("|")
 
    history = historyController.get_valid_history()
    if len(history) > 0:
        chewingSeq = f'{history[0]["chewingSeq"]} $ {chewingSeq}'
    decode_rst = decoder.decode(bpmSeq=chewingSeq)
    if len(decode_rst) == 0:
        return "error"
    compose_str = "".join(decode_rst[0][1].replace(' ',''))
    compose_str = compose_str.split('$')[-1]
    compose_rst = list(compose_str)
    compose_rst = manual_error_correct(
        oriSentList, errCorrections, compose_rst)
    return "".join(compose_rst)

@app.get('/enter',response_class=PlainTextResponse)
def handle_enter(sent:str,chewing_seq:str,correct_idx:str):
    # sent = request.args.get('sent')
    chewingSeq = chewing_seq
    # correct_idx = request.args.get('correct_idx')
    errCorrections = []
    if correct_idx != "none":
        errCorrections = correct_idx.split("|")
    enter_sents = sent_segmentation(sent, chewingSeq, errCorrections)
    if len(enter_sents) == 0:
        print("empty sentence")
        return "error"
    for s in enter_sents:
        partSentence = ""
        partChewing = ""
        isFix = 0
        errIdxs = []
        for i, item in enumerate(s):
            partChewing += item[1] + " "
            partSentence += item[0]
            if item[2] == 1:
                errIdxs.append(i)
                isFix = 1
        if len(partSentence) == 0:
            continue
        if isFix:
            decoder.extlm.adapt_sentence(partSentence, partChewing.strip(), errIdxs)
            with open(errLog,'a',encoding='utf-8') as f:
                f.write('{}\n{}\n'.format(partChewing,partSentence))
        else:
            decoder.extlm.log_sentence(partSentence, partChewing.strip())
        historyController.push(partSentence, partChewing.strip())
    return "ok"

@app.get('/cands',response_class=PlainTextResponse)
def handle_cands(sent:str,chewing_seq:str,position:int,correct_idx:str, is_emo:str):
    # sent = request.args.get('sent')
    global emoTab
    chewingSeq = chewing_seq
    # position = int(request.args.get('position'))
    # correct_idx = request.args.get('correct_idx')
    errCorrections = []
    if correct_idx != "none":
        errCorrections = correct_idx.split("|")
    # wordList = list(sent)
    chewingList = chewingSeq.split(" ")
    if decoder.is_mark(chewingList[position]):
        marks = decoder.list_mark(chewingList[position])
        return "\n".join(marks) +  "\nemoji:\n"
    elif str(position+1) in errCorrections:
        cands = decoder.list_candidate(chewingList[position])
        emo_list = []
        for i in range(0, decoder.maxEmojiLen):
            if position+1-i >= 0:
                if sent[position-i:position+1] in emoTab:
                    emo_list += [emo + ":" + sent[position-i:position+1] for emo in emoTab[sent[position-i:position+1]]]
        emo_list = list(set(emo_list))
        return "\n".join(cands) +  "\nemoji:\n" + "\n".join(emo_list)
    elif is_emo == "1":
        emo_list = []  
        cand_list =  decoder.list_candidate(chewingSeq)
        for term in cand_list:
            if term in emoTab:
                emo_list += [emo + ":" + term for emo in emoTab[term]]
        emo_list = list(set(emo_list))
        return "\n".join(cand_list) + "\nemoji:\n" + "\n".join(emo_list)

    else:
        cand_list = []
        emo_list = []
        for i in range(0,decoder.maxTermLen):
            if position-i >= 0:
                cand_list = decoder.list_candidate(" ".join(chewingList[position-i:position+1])) + cand_list
        for i in range(0, decoder.maxEmojiLen):
            if position+1-i >= 0:
                if sent[position-i:position+1] in emoTab:
                    emo_list += [emo + ":" + sent[position-i:position+1] for emo in emoTab[sent[position-i:position+1]]]
        emo_list = list(set(emo_list))
        return "\n".join(cand_list) + "\nemoji:\n" + "\n".join(emo_list)

@app.get('/bkup',response_class=PlainTextResponse)
def handle_bkup():
    decoder.extlm.disk_backup(salt=salt, writePath=biasFile)
    return "ok"

if __name__ == '__main__':
#     # try:
    # app.run(host="127.0.0.1", debug=False,port=9218)
    os.umask(0)
    if not os.path.isfile(biasFile):
        f = os.open(biasFile, os.O_CREAT | os.O_WRONLY, 0o777)
        os.close(f)
    if not os.path.isfile(errLog):
        f = os.open(errLog, os.O_CREAT | os.O_WRONLY, 0o777)
        os.close(f)
    uvicorn.run(app, host="127.0.0.1", port=9228,workers=1)
#     # except:
#     #     print('error occur')
#     #     pass
