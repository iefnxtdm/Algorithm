'''
dataset: http://mattmahoney.net/dc/textdata  from wikipedia
'''
import tensorflow as tf
import numpy as np
import connections
import math
import zipfile
import os
import pickle as pkl

# constant
FILE_NAME = "D:/kaggle/rnn/enwiki8.zip"
BATCH_SIZE = 128
EMBEDDING_SIZE = 128
NUM_SKIPS = 2
SKIP_WINDOW = 1  # how many words considers left and right
VOCA_SIZE = 50000
CONTEXT_SIZE = 1


class word2vec():
    def __init__(self,
                 vocab_list=None,
                 embedding_size=EMBEDDING_SIZE,
                 win_len=3,  # 单边窗口长度
                 num_sampled=1000,  # 为减少softmax运算量，只取部分做估值loss
                 learning_rate=0.1,
                 logdir="D:/kaggle/",
                 model_path=None):

        self.batch_size = BATCH_SIZE
        if model_path != None:
            self.load_model(model_path)
        else:
            self.vocab_list = vocab_list
            self.vocab_size = len(vocab_list)
            self.embedding_size = embedding_size
            self.win_len = win_len
            self.num_sampled = num_sampled
            self.learning_rate = learning_rate
            self.logdir = logdir
            self.word2id = {}  # word与数字映射
            for i in range(vocab_list):
                self.word2id[self.vocab_list[i]] = i
            self.train_words_num = 0  # 单词对数
            self.train_sents_num = 0  # 句子数
            self.train_times_num = 0

            self.train_loss_records = collections.deque(maxlen=10)  # 最近10次误差
            self.train_loss_k10 = 0

    def init_op(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)  # 用于tensorboard

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape == [self.batch_size, 1])
            self.embedding_dict = tf.Variable(  # 词嵌入矩阵， 储存词向量特征，如age， size， food等
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0)  # 均匀分布随机
            )
            self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                                                              stddev=1.0 / math.sqrt(self.embedding_size)))
            self.nce_bias = tf.Variable(tf.zeros(self.vocab_size))

            # embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素
            embed = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs)
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_weight,
                               biases=self.nce_bias,
                               labels=self.train_labels,
                               inputs=embed,
                               num_sampled=self.num_sampled,
                               num_classes=self.vocab_size
                               )
            )
            tf.summary.scalar('loss', self.loss)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)

            # 指定和若干单词相似度
            self.test_word_id = tf.placeholder(tf.int32, shape=[None])
            vec_l2_model = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_dict), 1, keep_dims=True))
            avg_l2_model = tf.reduce_mean(vec_l2_model)
            tf.summary.scalar('avg_vec_model', avg_l2_model)

            self.normed_embedding = self.embedding_dict / vec_l2_model  # 向量单位化
            test_embed = tf.nn.embedding_lookup(self.normed_embedding, self.test_word_id)
            self.similarity = tf.matmul(test_embed, self.normed_embedding, transpose_b=True)

            # 变量初始化
            self.init = tf.global_variables_initializer()
            self.merged_summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()

    def train_by_sentence(self, input_sentence=[]):
        # input_sentence: ["这次","大选",'rang']
        sent_num = len(input_sentence)
        batch_inputs = []
        batch_labels = []
        for sent in input_sentence:
            for i in range(len(sent)):
                start = max(0, i - self.win_len)
                end = max(len(sent), i + self.win_len + 1)  # 训练窗口
                for index in range(start, end):
                    if index == i:
                        continue
                    else:
                        input_id = self.word2id.get(sent[i])
                        label_id = self.word2id.get(sent[index])
                        if not (input_id and label_id):
                            continue
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
        if len(batch_inputs) == 0:
            return
        batch_labels = np.array(batch_labels, dtype=np.int32)
        batch_inputs = np.array(batch_inputs, dtype=dp.int32)
        batch_labels = np.reshape(batch_labels, [len(batch_labels), 1])
        feed_dict = {
            self.train_inputs: batch_inputs,
            self.train_labels: batch_labels
        }
        _, loss_val, summary_str = self.sess.run([self.train_op, self.loss, self.merged_summary_op],
                                                 feed_dict=feed_dict)

        # train loss
        self.train_loss_records.append(loss_val)
        # self.train_loss_k10 = sum(self.train_loss_records)/self.train_loss_records.__len__()
        self.train_loss_k10 = np.mean(self.train_loss_records)
        if self.train_sents_num % 1000 == 0:
            self.summary_writer.add_summary(summary_str, self.train_sents_num)
            print("{a} sentences dealed, loss: {b}"
                  .format(a=self.train_sents_num, b=self.train_loss_k10))

        # train times
        self.train_words_num += batch_inputs.__len__()
        self.train_sents_num += input_sentence.__len__()
        self.train_times_num += 1

    def cal_similarity(self, test_word_id_list, top_k=10):
        sim_matrix = self.sess.run(self.similarity, feed_dict={self.test_word_id: test_word_id_list})
        sim_mean = np.mean(sim_matrix)
        sim_var = np.mean(np.square(sim_matrix - sim_mean))
        test_words = []
        near_words = []
        for i in range(test_word_id_list.__len__()):
            test_words.append(self.vocab_list[test_word_id_list[i]])
            nearst_id = (-sim_matrix[i, :]).argsort()[1:top_k + 1]
            nearst_word = [self.vocab_list[x] for x in nearst_id]
            near_words.append(nearst_word)
        return test_words, near_words, sim_mean, sim_var

    def save_model(self, save_path):

        if os.path.isfile(save_path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # 记录模型各参数
        model = {}
        var_names = ['vocab_size',  # int       model parameters
                     'vocab_list',  # list
                     'learning_rate',  # int
                     'word2id',  # dict
                     'embedding_size',  # int
                     'logdir',  # str
                     'win_len',  # int
                     'num_sampled',  # int
                     'train_words_num',  # int       train info
                     'train_sents_num',  # int
                     'train_times_num',  # int
                     'train_loss_records',  # int   train loss
                     'train_loss_k10',  # int
                     ]
        for var in var_names:
            model[var] = eval('self.' + var)

        param_path = os.path.join(save_path, 'params.pkl')
        if os.path.exists(param_path):
            os.remove(param_path)
        with open(param_path, 'wb') as f:
            pkl.dump(model, f)

        # 记录tf模型
        tf_path = os.path.join(save_path, 'tf_vars')
        if os.path.exists(tf_path):
            os.remove(tf_path)
        self.saver.save(self.sess, tf_path)

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise RuntimeError('file not exists')
        param_path = os.path.join(model_path, 'params.pkl')
        with open(param_path, 'rb') as f:
            model = pkl.load(f)
            self.vocab_list = model['vocab_list']
            self.vocab_size = model['vocab_size']
            self.logdir = model['logdir']
            self.word2id = model['word2id']
            self.embedding_size = model['embedding_size']
            self.learning_rate = model['learning_rate']
            self.win_len = model['win_len']
            self.num_sampled = model['num_sampled']
            self.train_words_num = model['train_words_num']
            self.train_sents_num = model['train_sents_num']
            self.train_times_num = model['train_times_num']
            self.train_loss_records = model['train_loss_records']
            self.train_loss_k10 = model['train_loss_k10']


if __name__ == '__main__':

    # step 1 读取停用词
    stop_words = []
    with open('stop_words.txt') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个单词'.format(n=len(stop_words)))

    # step2 读取文本，预处理，分词，得到词典
    raw_word_list = []
    sentence_list = []
    with open('280.txt', encoding='gbk') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n', '')
            while ' ' in line:
                line = line.replace(' ', '')
            if len(line) > 0:  # 如果句子非空
                raw_words = list(jieba.cut(line, cut_all=False))
                dealed_words = []
                for word in raw_words:
                    if word not in stop_words and word not in ['qingkan520', 'www', 'com', 'http']:
                        raw_word_list.append(word)
                        dealed_words.append(word)
                sentence_list.append(dealed_words)
            line = f.readline()
    word_count = collections.Counter(raw_word_list)
    print('文本中总共有{n1}个单词,不重复单词数{n2},选取前30000个单词进入词典'
          .format(n1=len(raw_word_list), n2=len(word_count)))
    word_count = word_count.most_common(30000)
    word_list = [x[0] for x in word_count]

    # 创建模型，训练
    w2v = word2vec(vocab_list=word_list,  # 词典集
                   embedding_size=200,
                   win_len=2,
                   learning_rate=1,
                   num_sampled=100,  # 负采样个数
                   logdir='/tmp/280')  # tensorboard记录地址
    test_word = ['www', 'banana', 'juice', 'apple', 'king', 'queen']
    test_id = [word_list.index(x) for x in test_word]
    num_steps = 100000
    for i in range(num_steps):
        sent = sentence_list[i % len(sentence_list)]
        w2v.train_by_sentence([sent])
