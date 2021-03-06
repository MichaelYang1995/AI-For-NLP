# -*- coding: utf-8 -*-

# auxiliary function

#@author:QingyuanYANG
import os
import jieba
import re
import math
import time
import gensim
import numpy as np
from numpy import linalg as la
#import logging
from collections import defaultdict
from functools import wraps
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

Word2vec_model_path = r'D:\Assignment\Project_01\Word2vec_model.w2v'
related_word_path = r'D:\Assignment\Project_01\related_word.txt'
LTP_DATA_DIR = r'D:\ltp\ltp_data_v3.4.0'
#LTP_DATA_DIR = os.path.join(web_root, 'ltp')  # ltp模型目录的路径

pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

import sys

from pyltp import Postagger
from pyltp import Parser
from pyltp import NamedEntityRecognizer
from pyltp import Segmentor
import jieba.posseg as pseg
import functools
sys.setrecursionlimit(100000)

def split(sentence: str) ->'List(str)':
    # merge two quotations
    sentence = sentence.replace('”“', '，')
    N = len(sentence)

    # clean "“”"
    stack = []
    tmp = list(sentence)
    i = 0
    while i < N:
        c = tmp[i]
        if c == '“':
            stack.append(i)
        elif c == '”':
            l = stack.pop(0)
            if not ((i + 1 < N and tmp[i+1] in ['。', '！', '？']) or tmp[i-1] in ['。', '！', '？', '，']):
                tmp[l] = '"'
                tmp[i] = '"'
        i += 1
    sentence = ''.join(tmp)

    res = []
    stack = []
    l = 0
    for i, char in enumerate(sentence):
        if char == '“':
            stack.append(i)
            continue
        if char == "。" or char == "？" or char == "！":
            if not stack:
                res.append(sentence[l:i])
                l = i+1
        if char == "”":
            stack.pop()
            # if sentence[i-1] == '。': # direct quotation
            #    res.append(sentence[l:i+1])
            if i + 1 < N and sentence[i+1] == '。': # direct quotation
                res.append(sentence[l:i+1])
                l = i + 1
    if l < i+1:
        res.append(sentence[l:i+1])
    return res

def get_related_word(file_path):
    file = open(file_path, "r", encoding="utf-8")
    content_line = file.readline()
    result = []
    i = 0
    while content_line:
        temp = ''
        content_line = content_line.strip("\n")
        if len(content_line) > 0:
            result.append(content_line)
        content_line = file.readline()

    return result
# LOG_FORMAT = "%(asctime)s - %(funcName)s - %(message)s"
# logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)
# def fn_timer(function):
#     @wraps(function)
#     def function_timer(*args, **kwargs):
#         t0 = time.time()
#         result = function(*args, **kwargs)
#         t1 = time.time()
#         print("Total time running %s: %s seconds" %
#               (function.__name__, str(t1 - t0))
#               )
#         return result
#
#     return function_timer

# main Model
class Model_Pyltp:
    def __init__(self):
        self.name_says = defaultdict(list) #定义成全局变量有可能从sentence_process()中写入，也可能从single_sentence()写入
        self.model = gensim.models.word2vec.Word2Vec.load(Word2vec_model_path).wv
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(pos_model_path)  # 加载模型
        self.say_sim = get_related_word(related_word_path)
        self.valid_sentence = []

    ######################################### SIF embedding ########################################
    @functools.lru_cache()
    # @fn_timer
    def get_count(self, word):
        word_count = 0 #定义默认值
        vector = np.zeros(1) #定义默认值
        keys = self.model.vocab.keys()
        # 获取词频及词向量
        total_words_count = sum([v.count for k,v in self.model.vocab.items()]) #单词总数
        if word in keys:
            word_count = self.model.vocab[word].count
            vector = self.model[word] # 单词词语数量
        word_frequency = word_count/total_words_count # 词频
        return word_frequency,vector

    #获取句子向量
    #TODO: 计算P(w)的过程可以优化
    def sentence_embedding(self, sentence):
        # 按照论文算法Vs=1/|s|*∑a/(a+p(w))*Vw
        sentences = self.process_content(sentence).replace(' ','')
        a = 1e-3 #0.001
        words = list(self.pyltp_cut(sentences))
        sentence_length = len(words) #句子长度
        sum_vector = sum([a/(a+float(self.get_count(w)[0]))*self.get_count(w)[1] for w in words])
        sentence_vector = sum_vector/sentence_length
        return sentence_vector

    # 欧式距离
    def euclidSimilar(self, inA, inB):
        return 1.0/(1.0+la.norm(inA-inB))

    # 皮尔逊相关系数
    def pearsonSimilar(self, inA, inB):
        if len(inA) != len(inB):
            return 0.0
        if len(inA) < 3:
            return 1.0
        return 0.5+0.5*np.corrcoef(inA, inB, rowvar=0)[0][1]

    #余弦相似度
    def cosSimilar(self,inA,inB):
        inA = np.mat(inA)
        inB = np.mat(inB)
        num = float(inA*inB.T)
        denom = la.norm(inA)*la.norm(inB)
        return 0.5+0.5*(num/denom)

    def compare_sentence(self, inA, inB):
        inC = self.sentence_embedding(inA)
        inD = self.sentence_embedding(inB)
        return self.pearsonSimilar(inC, inD) #皮尔逊
        # print(self.euclidSimilar(inC,inD))
        # print(self.pearsonSimilar(inC,inD))
        # print(self.cosSimilar(inC,inD))
        # print('------------------------')
    ######################################### SIF embedding ########################################

    #################################### NER & Dependency Prasing ##################################
    #句子依存分析
    def parsing(self, sentence):
        words = list(self.pyltp_cut(sentence))  # pyltp分词
        # words=list(jieba.cut(sentence)) #结巴分词
        postags = list(self.postagger.postag(words))  # 词性标注
        # tmp=[str(k+1)+'-'+v for k,v in enumerate(words)]
        # print('\t'.join(tmp))
        parser = Parser() # 初始化实例
        parser.load(par_model_path)  # 加载模型
        arcs = parser.parse(words, postags)  # 句法分析
        parser.release()  # 释放模型
        return arcs

    #命名实体
    @functools.lru_cache()
    def get_name_entity(self, strs):
        sentence = ''.join(strs)
        recognizer = NamedEntityRecognizer()  # 初始化实例
        recognizer.load(ner_model_path)  # 加载模型
        # words = list(jieba.cut(sentence))  # 结巴分词
        words = list(self.pyltp_cut(sentence))#pyltp分词更合理
        postags = list(self.postagger.postag(words))  # 词性标注
        netags = recognizer.recognize(words, postags)  # 命名实体识别
        # tmp=[str(k+1)+'-'+v for k,v in enumerate(netags)]
        # print('\t'.join(tmp))
        recognizer.release()  # 释放模型
        return netags

    #################################### NER & Dependency Prasing ##################################

    ##################################### Input article Processing #################################
    def process_content(self,content):
        # print(type(content))
        # content=''.join(content)
        content=re.sub('[+——() ? 【】“”！，：。？、~@#￥%……&*（）《 》]+', '',content)
        content=' '.join(jieba.cut(content))
        return content

    #pyltp中文分词
    #@fn_timer
    @functools.lru_cache()
    def pyltp_cut(self,sentence):
        segmentor = Segmentor()  # 初始化实例
        segmentor.load(cws_model_path)  # 加载模型
        words = segmentor.segment(sentence)  # 分
        segmentor.release()  # 释放模型
        return words
    ##################################### Input article Processing #################################

    ######################################### Parsing content ######################################
    # 判断是否为代词结构句子“他认为...，他表示....”
    # @fn_timer
    def judge_pronoun(self, sentence):
        subsentence = re.search('(.+)“|”(.+)', sentence)
        if subsentence:
            sentence = subsentence.group(1)
        cuts = list(self.pyltp_cut(sentence))  # 确定分词
        wp = self.parsing(sentence)  # 依存分析
        postags = list(self.postagger.postag(cuts))
        for k, v in enumerate(wp):
            if v.relation == 'SBV' and postags[k] == 'r':  # 确定第一个主谓句
                return True
        return False

    # 输入主语第一个词语、谓语、词语数组、词性数组，查找完整主语
    def get_name(self, name, predic, words, property, ne):
        index = words.index(name)
        cut_property = property[index + 1:] #截取到name后第一个词语
        pre=words[:index]#前半部分
        pos=words[index+1:]#后半部分
        #向前拼接主语的定语
        while pre:
            w = pre.pop(-1)
            w_index = words.index(w)

            if property[w_index] == 'ADV': continue
            if property[w_index] in ['WP', 'ATT', 'SVB'] and (w not in ['，','。','、','）','（']):
                name = w + name
            else:
                pre = False
        while pos:
            w = pos.pop(0)
            p = cut_property.pop(0)
            if p in ['WP', 'LAD', 'COO', 'RAD'] and w != predic and (w not in ['，', '。', '、', '）', '（']):
                name = name + w # 向后拼接
            else: #中断拼接直接返回
                return name
        return name

    # 获取谓语之后的言论
    def get_says(self, sentence, property, heads, pos):
        # word = sentence.pop(0) #谓语
        if '：' in sentence:
            return ''.join(sentence[sentence.index('：')+1:])
        while pos < len(sentence):
            w = sentence[pos]
            p = property[pos]
            h = heads[pos]
            # 谓语尚未结束
            if p in ['DBL', 'CMP', 'RAD']:
                pos += 1
                continue
            # 定语
            if p == 'ATT' and property[h-1] != 'SBV':
                pos = h
                continue
            # 宾语
            if p == 'VOB':
                pos += 1
                continue
            # if p in ['ATT', 'VOB', 'DBL', 'CMP']:  # 遇到此性质代表谓语未结束，continue
            #    continue
            else:
                if w == '，':
                    return ''.join(sentence[pos+1:])
                else:
                    return ''.join(sentence[pos:])

    # 输入一个句子，若为包含‘说’或近似词则提取人物、言论，否则返回空
    # just_name:仅进行返回名字操作 ws:整句分析不进行多个“说判断”
    @functools.lru_cache()
    def single_sentence(self, sentence, just_name=False, ws=False):
        sentence = '，'.join([x for x in sentence.split('，') if x])
        cuts = list(self.pyltp_cut(sentence))  # pyltp分词更合理
        # mixed = list(set(self.pyltp_cut(sentence)) & set(self.say_sim))
        # mixed.sort(key=cuts.index)

        # if not mixed: return False

        # 判断是否有‘说’相关词：
        mixed = [word for word in cuts if word in self.say_sim]
        if not mixed: return False

        ne = self.get_name_entity(tuple(sentence))  # 命名实体
        wp = self.parsing(sentence)  # 依存分析
        wp_relation = [w.relation for w in wp]
        postags = list(self.postagger.postag(cuts))
        name = ''

        stack = []
        for k, v in enumerate(wp):
            # save the most recent Noun
            if postags[k] in ['nh', 'ni', 'ns']:
                stack.append(cuts[k])

            if v.relation == 'SBV' and (cuts[v.head - 1] in mixed):  # 确定第一个主谓句
                name = self.get_name(cuts[k], cuts[v.head - 1], cuts, wp_relation, ne)

                if just_name == True: return name  # 仅返回名字
                says = self.get_says(cuts, wp_relation, [i.head for i in wp], v.head)
                if not says:
                    quotations = re.findall(r'“(.+?)”', sentence)
                    if quotations: says = quotations[-1]
                return name, says
            # 若找到‘：’后面必定为言论。
            if cuts[k] == '：':
                name = stack.pop()
                says = ''.join(cuts[k + 1:])
                return name, says
        return False
    ######################################### Parsing content ######################################

    ###################################### Get_person_and_point ####################################
    # 输入单个段落句子数组
    def valid_sentences(self, sentences, res):
        expect = 0.76

        tmp = ""  # 储存前一个言论
        while sentences:
            curr = sentences.pop(0)
            if curr[0] == '“':  # 当前句子或为 “言论在发言人前的直接引用”。
                print(curr)
                people = re.search('”(.+)“|”(.+)', curr)  # 提取发言人所在句段
                if people:
                    people = [i for i in people.groups() if i][0]
                elif res:
                    res[-1][1] += '。' + curr
                    continue
                else:
                    continue

                saying = curr.replace(people, '')  # 剩余部分被假设为“言论”
                if res and self.judge_pronoun(people):
                    res[-1][1] += '。' + saying
                else:
                    comb = self.single_sentence(people)
                    if comb:
                        saying += comb[1] if comb[1] else ''
                        res.append([comb[0], saying])
                continue

            # 尝试提取新闻 发言人，言论内容
            combi = self.single_sentence(curr)

            # 无发言人： 当前句子属于上一个发言人的言论 或 不属于言论
            if not combi:
                if res and tmp and self.compare_sentence(tmp, curr) > expect:  # 基于句子相似度判断
                    print('{} - {} : {}'.format(tmp, curr, self.compare_sentence(tmp, curr)))
                    res[-1][1] += '。' + curr
                    tmp = curr
                continue

            # 有发言人： 提取 发言人 和 言论。
            name, saying = combi
            if res and self.judge_pronoun(curr) and saying:
                res[-1][1] += '。' + saying
            elif saying:
                res.append([name, saying])
            tmp = saying
        return res

    #解析处理语句并返回给接口
    def sentence_process(self,sentence):
        # 文章 -->清除空行
        # 文章 -->句号分割：如果句号分割A.B, 若B存在‘说’，对B独立解析，否则判断A | B是否相似，确定A是否抛弃B句。
        # 句子 -->确定主谓宾: 依存分析、命名实体识别 -->首先要找到宾语，然后确定宾语是否与说近似，若存在多个与‘说’近似，确定第一个为陈述。在说前找命名实体，说后面到本句结尾为宾语
        # 命名实体 -->通过命名实体识别，若S - NE, NE = S - NE。若B - NE / I - NE / E - NE，NE = B - NE + I - NE + E - NE

        self.name_says = defaultdict(list)
        sentence = sentence.replace('\r\n','\n')
        sections = sentence.split('\n') #首先切割成段落
        sections = [s for s in sections if s.strip()]
        valids=''
        # TODO:
        res = []
        for sec in sections: #段落
            # sec = sec.replace('。”', '”。')  #当做纠正语法错误...
            # sentence_list = sec.split('。')  # 段落拆分成句子
            sentence_list = split(sec)
            sentence_list = [s.strip() for s in sentence_list if s.strip()]
            self.cut_sententce_for_name = [s for s in sentence_list if s]
            res += self.valid_sentences(sentence_list, [])

        if res:
            self.name_says = defaultdict()
            for name, saying in res:
                if name and saying:
                    self.name_says[name] = self.name_says.get(name, '') + saying + ' | '
        return self.name_says

    ###################################### Get_person_and_point ####################################

    # #获取人物及人物观点中的命名实体
    # def get_name_saywords(self,content):
    #     name_says=self.sentence_process(content)
    #     result=[]
    #     says_list=[]
    #     if name_says:
    #         for name,says in name_says.items():
    #             print(name)
    #             print(says)
    #             says_str = ''.join([''.join(s) for s in says])
    #             # name,says=name_says[0],name_says[1]
    #             name_entity=self.get_name_entity(tuple(says_str))
    #             name_entity=' '.join(name_entity)
    #             result.append((name,name_entity))
    #     else:
    #         return None

    ######################################### Tfidf #########################################
    def document_frequency(self,word,document):
        if sum(1 for n in document if word in n)==0:
            print(word)
            print(type(document))
            print(len(document))
            print(document[0])
        return sum(1 for n in document if word in n)

    def idf(self,word,content,document):
        """Gets the inversed document frequency"""
        return math.log10(len(content) / self.document_frequency(word,document))

    def tf(self,word, document):
        """
        Gets the term frequemcy of a @word in a @document.
        """
        words = document.split()

        return sum(1 for w in words if w == word)
    ######################################### Tfidf #########################################

if __name__ == '__main__':
    sentence="据巴西《环球报》7日报道，巴西总统博索纳罗当天签署行政法令，放宽枪支进口限制，并增加民众可购买弹药的数量。\r\n《环球报》称，该法令最初的目的是放松对收藏家与猎人的限制，但现在扩大到其他条款。新法令将普通公民购买枪支的弹药数量上限提高至每年5000发，此前这一上限是每年50发。博索纳罗在法令签署仪式上称，“我们打破了垄断”“你们以前不能进口，但现在这些都结束了”。另据法新社报道，当天在首都巴西利亚的一次集会上，博索纳罗还表示，“我一直说，公共安全从家里开始的。”\r\n这不是巴西第一次放宽枪支限制。今年1月，博索纳罗上台后第15天就签署了放宽公民持枪的法令。根据该法令，希望拥有枪支的公民须向联邦警察提交申请，通过审核者可以在其住宅内装备最多4把枪支，枪支登记有效期由5年延长到10年。《环球报》称，博索纳罗在1月的电视讲话中称，要让“好人”更容易持有枪支。“人民希望购买武器和弹药，现在我们不能对人民想要的东西说不”。\r\n2004年，巴西政府曾颁布禁枪法令，但由于多数民众反对，禁令被次年的全民公投否决。博索纳罗在参加总统竞选时就表示，要进一步放开枪支持有和携带条件。他认为，放宽枪支管制，目的是为了“威慑猖狂的犯罪行为”。资料显示，2017年，巴西发生约6.4万起谋杀案，几乎每10万居民中就有31人被杀。是全球除战争地区外最危险的国家之一。\r\n不过，“以枪制暴”的政策引发不少争议。巴西《圣保罗页报》称，根据巴西民调机构Datafolha此前发布的一项调查，61%的受访者认为应该禁止持有枪支。巴西应用经济研究所研究员塞奎拉称，枪支供应增加1%，将使谋杀率提高2%。1月底，巴西民众集体向圣保罗联邦法院提出诉讼，质疑博索纳罗签署的放宽枪支管制法令。\r\n巴西新闻网站“Exame”称，博索纳罗7日签署的法案同样受到不少批评。公共安全专家萨博称，新的法令扩大了少数人的特权，不利于保护整个社会。（向南）\r\n"
    self = Model_Pyltp()

# # sentence="《环球报》称，该法令最初的目的是放松对收藏家与猎人的限制，但现在扩大到其他条款。新法令将普通公民购买枪支的弹药数量上限提高至每年5000发，此前这一上限是每年50发。博索纳罗在法令签署仪式上称，“我们打破了垄断”“你们以前不能进口，但现在这些都结束了”。另据法新社报道，当天在首都巴西利亚的一次集会上，博索纳罗还表示，“我一直说，公共安全从家里开始的。”"
# # sentence="据美国总统特朗普《环球报》7日报道，巴西总统博索纳罗当天签署行政法令，放宽枪支进口限制，并增加民众可购买弹药的数量"
# # sentence='A股早在2013年6月就已纳入新兴市场指数的候选列表中，但此后几年，都因为配额分配、资本流动限制、资本利得税等所谓原因而遭否决。\n在2016年第三次闯关失败后，中国投资者和相关监管部门似乎对“A股入摩”已心灰意冷，甚至连证监会分管国际合作的副主席方星海都在今年一月份的时候表示，“中国与MSCI在股指期货上的观点存在分歧，中国并不急于加入MSCI全球指数”。'
# obj=Model()
# b=obj.sentence_process(sentence)
# # obj.jieba_compare_pyltp(sentence)
# # obj.sentence_split(sentence)
# # b=obj.single_sentence(sentence)
# print(b)

# inA='博索纳罗在法令签署仪式上称，“我们打破了垄断”“你们以前不能进口，但现在这些都结束了”'
# inB='他认为，放宽枪支管制，目的是为了“威慑猖狂的犯罪行为”'
# inC='另据法新社报道，当天在首都巴西利亚的一次集会上，博索纳罗还表示，“我一直说，公共安全从家里开始的。”'
# # inD='除了获得更好的成绩,没有其他的爱好了'
# # inE='马同学坦言他也没有谈过一场恋爱'
# # inF='并且早上起来就打游戏'
# instant=Model()
# ins=instant.compare_sentence(inA,inB)
# print(ins)
# c=instant.compare_sentence(inA,inC)
# print(c)
# instant.compare_sentence(inA,inD)
# instant.compare_sentence(inA,inE)
# instant.compare_sentence(inA,inF)


