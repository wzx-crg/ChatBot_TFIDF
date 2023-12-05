import setence2vec as s2v
import numpy as np

def question2vec(question, wordlist, tdf_dict):
    q_vec = []
    for word in wordlist:
        if word not in question:
            q_vec.append(0)
        else:
            amount = 0
            for w in question:
                if word == w:
                    amount += 1
            tf = float(amount)/len(question)
            q_vec.append(tf*np.log(1/tdf_dict[word]))
    return q_vec


if __name__ == '__main__':
    qa_dict = s2v.QA2dict('me_test.ann.json')   # 构建大的QA字典
    vector_dict, wordlist, tdf_dict = s2v.questions2vector(qa_dict)
    while True:
        best_match_question = None
        best_sim = 0
        question = input('>>')  # 输入问题
        q_w_list = s2v.preprocessing_single_sentence(question)
        question_vec = question2vec(q_w_list, wordlist, tdf_dict)  # 预处理
        for q, v in zip(vector_dict.keys(), vector_dict.values()):
            inner_product = 0
            length = []
            temp = 0
            for i in range(len(wordlist)):
                inner_product += v[i]*question_vec[i]
            for i in question_vec:
                temp += i*i
            length.append(temp)
            temp = 0
            for i in v:
                temp += i*i
            length.append(temp)
            cos = inner_product/np.sqrt(length[0])/np.sqrt(length[1])
            if cos > best_sim:
                best_sim = cos
                best_match_question = q
        if best_match_question == None:
            print("I don't know, sorry")
        else:
            print(qa_dict[best_match_question])
            print('相似度', best_sim)