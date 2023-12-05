# 单个句子预处理
def preprocessing_single_sentence(sentence):
    import jieba
    # 构建停顿词列表
    with open('corpus/stopwords_chinese/scu_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [word.strip('\n') for word in f.readlines()]
    questions_wordlist = jieba.lcut(sentence)
    questions_wordlist = [w for w in questions_wordlist if w not in stopwords]
    return questions_wordlist

# 同义词列表构建
def generate_synonym_dict():
    with open('corpus/dict_synonym.txt', 'r', encoding='utf-8') as f:
        words_lists = [s[9::].strip('\n') for s in f.readlines()]
        for s, i in zip(words_lists, range(len(words_lists))):
            tmp = s.split(' ')
            words_lists[i] = tmp
        f.close()
    return words_lists

# 将问答的json文件整理成Q-A的dict
def QA2dict(file_path):
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        dialog_json = json.load(f)
        questions = [d['question'] for d in dialog_json.values()]
        tmp = dialog_json.values()
        tmp_value = [e['evidences'].values() for e in tmp]
        answers = []
        for t in list(tmp_value):
            t = list(t)
            answer = t[0]['answer'][0] + '\n' + t[0]['evidence']
            answers.append(answer)
        f.close()
    qa_dict = dict(zip(questions, answers))
    return qa_dict

# 语言预处理
def preprocessing(sentences):
    import jieba
    # 构建停顿词列表
    with open('corpus/stopwords_chinese/scu_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [word.strip('\n') for word in f.readlines()]
    questions_wordlist = []
    for q in sentences:
        q_wordlist = jieba.lcut(q)
        q_wordlist = [word for word in q_wordlist if word not in stopwords]
        questions_wordlist.append(q_wordlist)
    return questions_wordlist


# tf-idf处理成向量

def tf_idf(questions):
    # 统计所有词，建立向量维度，并计算出在所有term中每个词的frequency
    word_frequency_in_all_doc = {}
    wordlist = []
    word_amount_in_all_doc = {}
    for q in questions:
        for w in q:
            if w not in wordlist:
                wordlist.append(w)
                word_amount_in_all_doc[w] = 0
            word_amount_in_all_doc[w] += 1
    for word in wordlist:
        word_frequency_in_all_doc[word] = float(word_amount_in_all_doc[word])/len(questions)

    # 计算各个词在各个term中的frequency
    word_frequency_in_term = []
    for q in questions:
        q_set = set(q)
        q_dict = {}
        for w in iter(q_set):
            amount = 0
            for i in q:
                if w == i:
                    amount += 1
            q_dict[w] = float(amount)/len(q)
        word_frequency_in_term.append(q_dict)
    # 将问题以tf-idf向量化
    import numpy as np
    questions_vector = []
    for q, i in zip(questions, range(len(questions))):
        q_vector = []
        for w in wordlist:
            if w in q:
                q_vector.append(word_frequency_in_term[i][w]*np.log(1.0/word_frequency_in_all_doc[w]))
            else:
                q_vector.append(0)
        questions_vector.append(q_vector)
    return questions_vector, wordlist, word_frequency_in_all_doc


# 将提问句子训练为vector，并对应一个回答
def questions2vector(QA_dict):
    questions = QA_dict.keys()
    # 进行preprocessing
    questions_wordlist = preprocessing(questions)
    # 进行tf-idf向量化
    questions_vec, wordlist, word_frequency_in_all_doc = tf_idf(questions_wordlist)
    vector_dict = dict(zip(questions, questions_vec))
    return vector_dict, wordlist, word_frequency_in_all_doc


if __name__ == '__main__':
    qa_dict = QA2dict('me_test.ann.json')
    vector, wordlist, idf = questions2vector(qa_dict)
    with open("vector.txt", 'w', encoding='utf-8') as f:
        for q in vector.keys():
            f.writelines(q+'\n')
            #f.writelines(','.join(vector[q]))
