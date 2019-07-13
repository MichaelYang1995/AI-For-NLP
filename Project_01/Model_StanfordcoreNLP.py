# -*- coding: utf-8 -*-

#auxiliary function

#@author:QingyuanYANG
import re
import gensim
import functools
import numpy as np
import pandas as pd
import os.path, jieba
from collections import Counter
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from stanfordcorenlp import StanfordCoreNLP

Word2vec_model_path = r'D:\Assignment\Project_01\Word2vec_model.w2v'
related_word_path = r'D:\Assignment\Project_01\related_word.txt'

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

class Model_StanfordcoreNLP:
    def __init__(self):
        self.model = gensim.models.word2vec.Word2Vec.load(Word2vec_model_path).wv
        self.related_word = get_related_word(related_word_path)
        self.nlp = StanfordCoreNLP(r'D:\stanford_nlp', lang='zh')

    ######################################### SIF embedding ########################################
    def get_probability(self, word):
        keys = self.model.vocab.keys()
        total_words_count = sum([v.count for k, v in self.model.vocab.items()])
        esp = 1 / total_words_count

        if word in keys:
            word_count = self.model.vocab[word].count
            return word_count / total_words_count
        else:
            return esp

    def get_SIF(self, word, a=0.01):
        return (a + self.get_probability(word))

    def SIF_sentence_embdding(self, sentence, a=0.01):
        result = 0
        for word in sentence:
            if word not in self.model:
                continue
            temp = self.model[word] * self.get_SIF(word)
            result += temp
        return result

    def PCA_SIF_sentence_embdding(self, news, embedding_size=35, a=0.01):
        X = [self.SIF_sentence_embdding(sentence, self.model) for sentence in news]

        pca = PCA(n_components=min(embedding_size, len(X)))

        pca.fit(np.array(X))

        u = pca.components_[0]  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT

        # pad the vector?  (occurs if we have less sentences than embeddings_size)
        if len(u) < embedding_size:
            for i in range(embedding_size - len(u)):
                u = np.append(u, 0)  # add needed extension for multiplication below
        # resulting sentence vectors, vs = vs -u x uT x vs
        Y = []
        for Vs in X:
            sub = np.multiply(u, Vs)
            Y.append(np.subtract(Vs, sub))
        return Y

    ######################################### SIF embedding ########################################

    #################################### NER & Dependency Prasing ##################################
    def NLP_NER(self, sentence):
        return self.nlp.ner(sentence)

    def NLP_POS(self, sentence):
        return self.nlp.pos_tag(sentence)

    def NLP_DPR(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def SUBJ(self, string):
        return re.search('subj', string)
    #################################### NER & Dependency Prasing ##################################

    ##################################### Input article Processing #################################
    def token(self, string):
        return ''.join(re.findall(r'[\d|\w]+', string))

    def cut(self, string):
        return ' '.join(jieba.cut(string))

    def cut_sentence(self, para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")

    def get_sentence_content_with_punctuation(self, news):
        news = re.sub(r'\n', '', news)
        if news != '' and news != ' ':
            result = self.cut_sentence(news)
        return result

    def get_sentence_word_content(self, news_sentence_with_punctuation):
        result = []

        for sentence in news_sentence_with_punctuation:
            temp_word = jieba.lcut(self.token(sentence))
            result.append(temp_word)
        return result
    ##################################### Input article Processing #################################

    ######################################### Parsing content ######################################
    def memo(func):
        cache = {}

        def _wrap(*args):  ## ? *args, **kwargs
            if args in cache:
                result = cache[args]
            else:
                result = func(*args)
                cache[args] = result
            return result

        return _wrap

    @memo
    def judge_pronoun(self, sentence):
        tmp_sen = re.search('(.+)“|”(.+)', sentence)
        if tmp_sen:
            sentence = subsentence.group(1)
        POS = self.NLP_POS(sentence)
        D_pr = self.NLP_DPR(sentence)
        for relation in D_pr:
            i = relation[-1] - 1
            if self.SUBJ(relation[0]) and (POS[i][-1] == 'PN' or POS[i][-1] == 'PRP'):
                return True
        return False

    def get_name(self, subj_index, word_cut, D_pr):
        name = word_cut[subj_index]
        for relation in D_pr:
            if relation[0] == 'appos': continue
            if relation[1] - 1 == subj_index:
                name = word_cut[relation[2] - 1] + name
                #print(relation)
                #print(word_cut[relation[2] - 1])
        return name

    def get_saying(self, predicate_index, word_cut):
        if ':' in word_cut:
            return ''.join(word_cut[word_cut.index(':') + 1:])

        return ''.join(word_cut[predicate_index + 1:])

    def parse_sentence(self, sentence):
        name = ''
        saying = ''
        NER = self.NLP_NER(sentence)
        word_cut = [w for w, _ in NER]

        # 寻找句中第一个与‘说’相关的词汇，若无则直接停止解析，返回False。
        say_related_word = [word for word in word_cut if word in self.related_word]
        if not say_related_word: return False

        D_pr = self.NLP_DPR(sentence)

        for i, relation in enumerate(D_pr):

            # 首先进行句式解析，判断依赖‘说’的主语是否为'PERSON/ ORGANIZATION/ LOCATION'。
            d = relation[1] - 1  # 谓语索引
            k = relation[2] - 1  # 主语索引

            if (word_cut[d] in say_related_word) and self.SUBJ(relation[0]):  # 找出第一个主谓结构
                if (NER[k][-1] == 'PERSON' or NER[k][-1] == 'ORGANIZATION' or NER[k][-1] == 'LOACTION'):
                    name = self.get_name(k, word_cut, D_pr)
                    saying = self.get_saying(d, word_cut)

                    if not saying:
                        quotations = re.findall(r'“(.+?)”', sentence)
                        if quotations:
                            saying = quotations[-1]
                    return [name, saying]

            # 若句子中有与‘说’相关词汇，且存在‘：’，则直取其后言论。
            if word_cut[i][0] == ':':
                for j in range(j):
                    if NER[j][-1] == 'PERSON' or NER[j][-1] == 'ORGANIZATION' or NER[j][-1] == 'LOACTION':
                        name += NER[j][-1]
                saying = ''.join(word_cut[i + 1:])
                return [name, saying]
        return False
    ######################################### Parsing content ######################################

    ####################################### Get_person_and_point ###################################
    def compare_sentence(self, sen1_index, sen2_index, Y):
        return cosine(Y[sen1_index], Y[sen2_index])

    def get_person_and_point(self, input_text):
        res = []
        V_point = defaultdict(list)
        flag = -1

        news_sentence_with_punctuation = self.get_sentence_content_with_punctuation(input_text)
        #news_sentence = get_sentence_content(news_sentence_with_punctuation)

        # news_word_with_punctuation = get_sentence_word_content(news_sentence_with_punctuation)
        news_word = self.get_sentence_word_content(news_sentence_with_punctuation)
        # print(news_sentence_with_punctuation)
        # print(news_sentence)
        # print(news_word_with_punctuation)
        # print(news_word)
        Y = self.PCA_SIF_sentence_embdding(news_word)
        tmp_sentence_index = ""  # 储存前一个言论的索引

        for j, sentence in enumerate(news_sentence_with_punctuation):
            expect = 0.75  # sentence embdding比较因素
            print(sentence)

            tmp_res = []
            # 特殊情况处理：当句子中第一个字符出现“
            if sentence[0] == '“':
                the_subsen_of_people_in = re.search('”(.+)“|”(.+)', sentence) # 提取发言人所在句段
                if the_subsen_of_people_in:
                    the_subsen_of_people_in = [sen for sen in the_subsen_of_people_in.groups() if sen][0]

                    saying = sentence.replace(the_subsen_of_people_in, '')  # 剩余部分即为言论
                    if res and self.judge_pronoun(the_subsen_of_people_in):
                        res[-1][1] += saying
                    else:
                        tmp_res = self.parse_sentence(the_subsen_of_people_in)
                        if tmp_res:
                            saying += tmp_res[1] if tmp_res[1] else ''
                            res.append([tmp_res[0], saying])
                    continue
                elif res:
                    res[-1][1] += sentence
                    continue
                else:
                    continue

            # 一般情况处理
            # 提取：发言人，言论内容
            tmp_res = self.parse_sentence(sentence)

            # 一般情况处理（1）：不存在发言人
            if not tmp_res:
                if res and tmp_sentence_index and self.compare_sentence(tmp_sentence_index, j, Y) > expect:
                    res[-1][1] += sentence
                    tmp_sentence_index = j
                else:
                    tmp_sentence_index = ""
                continue

            # 一般情况处理（2）：存在发言人
            if tmp_res[-1]:
                res.append(tmp_res)
            tmp_sentence_index = j
        return res

    def get_person_and_point_dict(self, input_text):
        res = defaultdict(list)

        tmp = self.get_person_and_point(input_text)

        for person, saying in tmp:
            if saying[0] in ['，', '。', '、', '）', '!', '?']:
                saying = saying[1:]
            res[person].append(saying)
        return res
    ####################################### Get_person_and_point ###################################

if __name__ == '__main__':
    test1 = "据巴西《环球报》7日报道，巴西总统博索纳罗当天签署行政法令，放宽枪支进口限制，并增加民众可购买弹药的数量。\r\n《环球报》称，该法令最初的目的是放松对收藏家与猎人的限制，但现在扩大到其他条款。新法令将普通公民购买枪支的弹药数量上限提高至每年5000发，此前这一上限是每年50发。博索纳罗在法令签署仪式上称，“我们打破了垄断”“你们以前不能进口，但现在这些都结束了”。另据法新社报道，当天在首都巴西利亚的一次集会上，博索纳罗还表示，“我一直说，公共安全从家里开始的。”\r\n这不是巴西第一次放宽枪支限制。今年1月，博索纳罗上台后第15天就签署了放宽公民持枪的法令。根据该法令，希望拥有枪支的公民须向联邦警察提交申请，通过审核者可以在其住宅内装备最多4把枪支，枪支登记有效期由5年延长到10年。《环球报》称，博索纳罗在1月的电视讲话中称，要让“好人”更容易持有枪支。“人民希望购买武器和弹药，现在我们不能对人民想要的东西说不”。\r\n2004年，巴西政府曾颁布禁枪法令，但由于多数民众反对，禁令被次年的全民公投否决。博索纳罗在参加总统竞选时就表示，要进一步放开枪支持有和携带条件。他认为，放宽枪支管制，目的是为了“威慑猖狂的犯罪行为”。资料显示，2017年，巴西发生约6.4万起谋杀案，几乎每10万居民中就有31人被杀。是全球除战争地区外最危险的国家之一。\r\n不过，“以枪制暴”的政策引发不少争议。巴西《圣保罗页报》称，根据巴西民调机构Datafolha此前发布的一项调查，61%的受访者认为应该禁止持有枪支。巴西应用经济研究所研究员塞奎拉称，枪支供应增加1%，将使谋杀率提高2%。1月底，巴西民众集体向圣保罗联邦法院提出诉讼，质疑博索纳罗签署的放宽枪支管制法令。\r\n巴西新闻网站“Exame”称，博索纳罗7日签署的法案同样受到不少批评。公共安全专家萨博称，新的法令扩大了少数人的特权，不利于保护整个社会。（向南）\r\n"
    test2 = "（原标题：44岁女子跑深圳约会网友被拒，暴雨中裸身奔走……', '）@深圳交警微博称：昨日清晨交警发现有一女子赤裸上身，行走在南坪快速上，期间还起了轻生年头，一辅警发现后赶紧为其披上黄衣，并一路劝说她。', '那么事发时到底都发生了些什么呢？', '南都记者带您一起还原现场南都记者在龙岗大队坂田中队见到了辅警刘青（发现女生的辅警），一位外表高大帅气，说话略带些腼腆的90后青年。', '刘青介绍，6月16日早上7时36分，他正在环城南路附近值勤，接到中队关于一位女子裸身进入机动车可能有危险的警情，随后骑着小铁骑开始沿路寻找，大概花了十多分钟在南坪大道坂田出口往龙岗方向的逆行辅道上发现该女子。', '女子身上一丝不挂地逆车流而行，时走时停，时坐时躺，险象环生。', '刘青停好小铁骑，和另外一名巡防员追了上去，发现女子的情绪很低落，话不多，刘青尝试和女子交流，劝说女子离开，可女子并不愿意接受，继续缓慢地往南坪快速路的主干道上走去。', '此时路边上已经聚集了很市民围观，为了不刺激女子的情绪，刘青和巡防员一边盯着女子一边驱赶着围观的群众。', '现场还原从警方提供的一份视频了解到，16日早上7时25分，女子出现在坂雪岗大道与环城南路的监控视频中，此时女子还穿着白色的内裤，正沿着坂雪岗大道往南坪快速的方向缓慢地走着。', '当时正值上班高峰期，十字路口的车流已经排起了长队。', '当女子出现时，路上的市民纷纷驻足观望，不少车辆也放慢了速度，但女子并不为市民观望停下脚步，依然缓慢走着。', '当女子行进到十字路口中间时，一辆大货车挡住了镜头，但是当女子再次出现镜头时，可以发现女子已经没穿内裤了，全身裸露继续朝着南坪快速方向走去。', '记者发现，视频中女子周围并没有人尾随或者上前劝止的市民。', '一大清早路上看到这样的情况恐怕大家都没办法淡定面对这一情况刘青表示，“一开始根本不敢看她，心里挺别扭，感觉很尴尬”，但当刘青跟随女子上了南坪快速路主干道时，女子作出了让人意想不到的举动，她突然靠近护栏要从上面跳下去，刘青赶忙冲上去拉住了女子的手，将其控制住并远离护栏。', '碍于女子没有穿衣服，刘青递上衣服，女子没接受还把衣服扔到排水沟里，继续往前走，没办法刘青只能紧紧拉着她的一只手跟在后面。', '刘青一路上耐心地开导安慰她，但只听到她不断地重复着一句话“要是你也遭遇我的事，你也会这样的”，期间她还不时试图挣脱刘青的手要冲向护栏往下跳。', '就这样，我被牵着走了大概十多分钟，天突然下起了大暴雨，雨大的连眼睛都睁不开”刘青继续说着，瞬间他们就被雨透了，但女子依然不愿意接受刘青的帮助，就继续冒着大雨往前走。', '大概走了有四十分钟吧，女子突然停下来说“我想回家了”，然后女子也接受了刘青递过来的小黄衣，就出现了深圳微博上的照片，女子披着小黄衣，刘青小心翼翼地在旁边走着的场景。', '从南平快速下来后，刘青和巡防员将女子带到了附近的坂田派出所。', '那姑娘到底是遭遇了什么样的事情才会说“要是你也遭遇我的事，你也会这样”据警方透露，该女子姓陈，系湖北人，今年44岁，据家属反映其有精神病史。', '三天前，陈某从老家来深圳约会网友，但约会受挫导致情绪异常，女子遂产生轻生念头。', '目前陈某已经被送往深圳某精神病医院进行治疗大大君只希望姑娘能早点康复其实真爱的到来并不存在年龄的限制你们说呢？', '因善良的原因一众网友纷纷为交警暖男点ZAN@弓常yan桦：就想问这个小哥哥有女票吗@原谅我这一辈子浪荡不羁爱萨摩耶：有什么过不去的要轻生嘛？', ' 想想自己的家人。', '同时也感谢交警蜀黍@火心聆听心灵：点赞交警@中華云盾：警察……', '警察就是群众最需时申出援手@Tomchlee：蜀黍帅！', '@SJ-李赫海i：这个交警很暖有木有！', '男子迷奸网友拍418个视频 女方从20岁到50岁不等去年6月7号上午，淮安市涟水县公安局刑警大队突然接到了一个奇怪的报警电话，一名女子言语不清，声称自己遭到了侵害。', '女子、被侵害、言语不清，几个关键词令接到电话的民警瞬间紧张起来。"
    model = Model_StanforcoreNLP()
    V_point1 = model.get_person_and_point_dict(test1)
    V_point2 = model.get_person_and_point_dict(test2)
    print(V_point1)
    print(V_point2)