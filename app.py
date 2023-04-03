from flask import Flask, render_template, request
from gensim import corpora, models
import pyLDAvis.gensim

import re
import nltk
#from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


app = Flask(__name__)


# def preprocess(text):
#     # 특수문자 제거
#     text = re.sub(r'[^\w\s]', '', text)
#
#     # 대소문자 통일
#     text = text.lower()
#
#     # 불용어(stopwords) 제거
#     stop_words = set(stopwords.words('english'))
#     tokens = nltk.word_tokenize(text)
#     tokens = [token for token in tokens if not token in stop_words]
#
#     # 어간추출(stemming)
#     stemmer = PorterStemmer()
#     tokens = [stemmer.stem(token) for token in tokens]
#
#     return tokens


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # 입력된 문장 전처리 코드
#         #sentence = preprocess(request.form['sentence'])
#         sentence = request.form['sentence']
#
#         # 저장된 모델 로드
#         dictionary = corpora.Dictionary.load('dictionary.dict')
#         lda_model = models.LdaModel.load('lda_model.model')
#
#         # 문장을 토큰화하여 bow 형태로 변환
#         bow_vector = dictionary.doc2bow(sentence.split())
#
#         # 문장의 주제 확인
#         topics = lda_model[bow_vector]
#
#         # 시각화
#         vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
#         html = pyLDAvis.prepared_data_to_html(vis)
#
#         return render_template('index.html', topics=topics, html=html)
#     else:
#         return render_template('index.html')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results', methods=['POST', 'GET'])
def results():
    # HTML form에서 전송한 데이터 받아오기
    documents = request.form.get('documents')
    num_topics = int(request.form.get('num_topics'))

    # 문서 토큰화 및 불용어 제거
    stopwords = ['the', 'and', 'of', 'to', 'in', 'a']
    texts = [[word for word in document.lower().split() if word not in stopwords] for document in documents.split('\n')]

    # 문서를 단어-빈도 행렬로 변환
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # LDA 모델링 수행
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

    # 각 토픽의 상위 단어 출력
    topic_words = []
    for i in range(num_topics):
        topic_words.append([word for word, prob in lda.show_topic(i)])

    return render_template('results.html', documents=documents, num_topics=num_topics, topic_words=topic_words)


if __name__ == '__main__':
    app.run(debug=True)
