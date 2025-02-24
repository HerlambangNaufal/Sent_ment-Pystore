
import streamlit as st
from google_play_scraper import Sort, reviews
import pandas as pd
import numpy as np
import re
import string
import nltk
import seaborn as sns
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, classification_report
from collections import Counter

def main():

    # make sidebar
    with st.sidebar:
        st.header('Sentiment Analisis Aplikasi Google Play Store')
        st.image(Image.open('sentiment-analysis.webp'))
        st.caption('© HerlambangNaufal 2024')

    # make tabs for Menu
    tab1,tab2,tab3,tab4 = st.tabs(["Scraping","Pre-Processing & Labeling","Visualisi Data","Model Evaluasi"])

    # Scraping Data
    with tab1:
        with st.form(key='my-form'):
            url = st.text_input('Enter Link Apps')
            counts = st.number_input('amount of data', min_value=50 ,step=1)
            submit = st.form_submit_button('Submit')
    
        if "submits" not in st.session_state:
            st.session_state.submits = False
    
        def callback():
            st.session_state.submits = False
    
        if submit or st.session_state.submits:
            st.session_state.submits = True
            try:
                # Initialize empty list for results and continuation token
                result = []
                continuation_token = None
    
                # Loop until desired count is reached or there are no more reviews
                while len(result) < counts:
                    reviews_batch, continuation_token = reviews(
                        url,
                        lang='id',
                        country='id',
                        sort=Sort.NEWEST,
                        count=min(counts - len(result), 100),  # Limit to 100 per batch for efficiency
                        continuation_token=continuation_token
                    )
                    result.extend(reviews_batch)
    
                    # Stop if there are no more reviews
                    if continuation_token is None:
                        break
    
                # Convert to DataFrame and filter by year (2021-2024)
                df = pd.DataFrame(np.array(result), columns=['review'])
                df = df.join(pd.DataFrame(df.pop('review').tolist()))
                df = df[['userName', 'score', 'at', 'content']]
                df['at'] = pd.to_datetime(df['at'])
                df = df[df['at'].dt.year.isin([2021, 2022, 2023, 2024, 2025])]
                # Simpan rating asli sebelum proses sentimen lexicon
                df['score_original'] = df['score']  # Simpan rating asli dari Google Play Store
    
                # Display filtered data and provide download option
                st.dataframe(df)
                st.download_button(label='Download CSV', data=df.to_csv(index=False, encoding='utf8'), file_name='Labeled_'+url+'.csv', on_click=callback)
    
            except Exception as e:
                st.write(f'Error: {e}')


    # Pre-Processing & Labeling
    with tab2:
        try:
            data_file = st.file_uploader("Upload CSV file",type=["csv"])            
            if data_file is not None :
                df = pd.read_csv(data_file)
                st.dataframe(df)

                proses = st.button('Start process')

                if "prosess" not in st.session_state:
                    st.session_state.prosess = False

                def callback():
                    st.session_state.prosess = False

                if proses or st.session_state.prosess:
                    st.session_state.prosess = True

                    # Cleaning Text
                    def cleansing(text):
                        #removing number
                        text = re.sub(r"\d+","", text)
                        # remove non ASCII (emoticon, chinese word, .etc)
                        text = text.encode('ascii', 'replace').decode('ascii')
                        # remove mention, link, hashtag
                        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
                        #Alphabeth only, exclude number and special character
                        text = re.sub(r'[^a-zA-Z]', ' ', text)
                        text = re.sub(r'\b[a-zA-Z]\b', ' ', text)
                        # replace word repetition with a single occutance ('oooooooo' to 'o')
                        text = re.sub(r'(.)\1+', r'\1\1', text)
                        # replace punctations repetitions with a single occurance ('!!!!!!!' to '!')
                        text = re.sub(r'[\?\.\!]+(?=[\?.\!])', '', text)
                        #remove multiple whitespace into single whitespace
                        text = re.sub('\s+',' ',text)
                        #remove punctuation
                        text = text.translate(text.maketrans("","",string.punctuation))
                        # Remove double word
                        text = text.strip()
                        text = ' '.join(dict.fromkeys(text.split()))
                        return text

                    # Case folding text
                    def casefolding(text):
                        text = text.lower()
                        return text

                    # Tokenize text
                    def tokenize(text):
                        text = word_tokenize(text)
                        return text

                    # Normalisasi text
                    normalizad_word = pd.read_excel("colloquial-indonesian-lexicon.xlsx")
                    normalizad_word_dict = {}
                    for index, row in normalizad_word.iterrows():
                        if row[0] not in normalizad_word_dict:
                            normalizad_word_dict[row[0]] = row[1]

                    def normalized_term(text):
                        return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in text]

                    # Filltering | stopwords removal
                    def stopword(text):
                        listStopwords = set(stopwords.words('indonesian'))
                        filtered = []
                        for txt in text:
                            if txt not in listStopwords:
                                filtered.append(txt)
                        text = filtered 
                        return text

                    # Remove punctuation
                    def remove_punct(text):
                        text = " ".join([char for char in text if char not in string.punctuation])
                        return text

                    # Deploy Function
                    st.write("===========================================================")
                    st.write("Start Pre-processing")

                    st.caption("| cleaning...")
                    df['cleansing'] = df['content'].apply(cleansing)

                    st.caption("| case folding...")
                    df['case_folding'] = df['cleansing'].apply(casefolding)
                    
                    st.caption("| tokenizing...")
                    df['text_tokenize'] = df['case_folding'].apply(tokenize)

                    st.caption("| normalization...")
                    df['tweet_normalized'] = df['text_tokenize'].apply(normalized_term)

                    st.caption("| removal stopwords...")
                    df['text_stopword'] = df['tweet_normalized'].apply(stopword)

                    # Remove Puct 
                    df['text_clean'] = df['text_stopword'].apply(lambda x: remove_punct(x))

                    # Remove NaN file
                    df['text_clean'].replace('', np.nan, inplace=True)
                    df.dropna(subset=['text_clean'],inplace=True)

                    # Reset index number
                    df = df.reset_index(drop=True)
                    st.write("Finish Pre-processing")
                    st.write("===========================================================")
                
                    # Determine sentiment polarity of doc using indonesia sentiment lexicon
                    st.write("Count Polarity and Labeling...")
                    st.caption("using indonesia sentiment lexicon")
                    lexicon = dict()
                    import csv
                    with open('InSet_Lexicon.csv', 'r') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',')
                        for row in reader:
                            lexicon[row[0]] = int(row[1])

                    # Function to determine sentiment polarity of tweets        
                    def sentiment_analysis_lexicon_indonesia(text):
                        #for word in text:
                        score = 0
                        for word in text:
                            if (word in lexicon):
                                score = score + lexicon[word]

                        polarity=''
                        if (score > 0):
                            polarity = 'positive'
                        elif (score == 0):
                            polarity = 'neutral'
                        else:
                            polarity = 'negative'
                        return score, polarity

                    results = df['text_stopword'].apply(sentiment_analysis_lexicon_indonesia)
                    results = list(zip(*results))
                    df['score'] = results[0]
                    df['sentiment'] = results[1]
                    st.text(df['sentiment'].value_counts())

                    st.dataframe(df)
                    st.download_button(label='Download CSV', data = df.to_csv(index=False, encoding='utf8'), file_name='Labeled_'+url+'.csv',on_click=callback)

        except:
            st.write('Select The Correct File')

    with tab3:
        try:
            data_file = st.file_uploader("Upload CSV file dengan label sentimen", type=["csv"])
            if data_file is not None:
                df = pd.read_csv(data_file)
                st.write("Data Preview:")
                st.dataframe(df.head())
                
                # Pastikan kolom yang dibutuhkan ada
                if 'text_clean' in df.columns and 'sentiment' in df.columns:
                    
                    # Mapping label sentimen ke angka
                    sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
                    df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)
                    
                    # TF-IDF Vectorization
                    vectorizer = TfidfVectorizer(max_features=5000)
                    X = vectorizer.fit_transform(df['text_clean'])
                    y = df['sentiment_label']
                    
                    # Split Data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Train SVM Model
                    model = SVC(kernel='linear', probability=True)
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    
                    # Classification Report
                    st.subheader("Classification Report")
                    st.text(classification_report(y_test, y_pred))
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    labels = ['Negative', 'Neutral', 'Positive']
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix")
                    st.pyplot(fig)
                    
                    # Distribusi Prediksi Sentimen
                    pred_counts = pd.Series(y_pred).map({-1: 'Negative', 0: 'Neutral', 1: 'Positive'}).value_counts()
                    fig, ax = plt.subplots(figsize=(6, 4))
                    pred_counts.plot(kind='bar', color=['red', 'gray', 'blue'])
                    plt.title("Distribusi Sentimen Prediksi")
                    plt.xlabel("Sentiment")
                    plt.ylabel("Jumlah Ulasan")
                    st.pyplot(fig)
                    
                    st.success("Model SVM berhasil dilatih dan divisualisasikan!")
                else:
                    st.error("Kolom 'text_clean' dan 'sentiment' tidak ditemukan dalam dataset.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    
            
    with tab4:
        try:
            import seaborn as sns
            data_file = st.file_uploader("Upload labeled CSV file",type=["csv"])
            if data_file is not None :
                df = pd.read_csv(data_file)
                st.dataframe(df)

                proseseval = st.button('Start processs')

                if "evalmodel" not in st.session_state:
                    st.session_state.evalmodel = False

                def callback():
                    st.session_state.evalmodel = False

                if proseseval or st.session_state.evalmodel:
                    st.session_state.evalmodel = True

                    st.write("\n Counting SVM Accuracy...")

                    def score_sentiment(score):
                        if score == 'positive':
                            return "positive"
                        elif score == 'negative':
                            return "negative"
                        else:
                            return "neutral"

                    biner = df['sentiment'].apply(score_sentiment)

                    X_train, X_test, Y_train, Y_test = train_test_split(df['text_clean'], biner, test_size=0.2, stratify=biner, random_state=42)
                    # Jumlah data latih
                    jumlah_data_latih_positive = sum(Y_train == "positive")
                    jumlah_data_latih_negative = sum(Y_train == "negative")
                    jumlah_data_latih_neutral = sum(Y_train == "neutral")

                    st.write("Data Latih:")
                    st.write(f"Jumlah data latih dengan sentimen positive: {jumlah_data_latih_positive}")
                    st.write(f"Jumlah data latih dengan sentimen negative: {jumlah_data_latih_negative}")
                    st.write(f"Jumlah data latih dengan sentimen neutral: {jumlah_data_latih_neutral}")

                    st.write("====================================================================")

                    from sklearn.model_selection import cross_val_score, StratifiedKFold
                    st.subheader("Evaluasi Model dengan Cross-Validation")
    
                    # Konversi text_clean ke fitur numerik menggunakan TF-IDF
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform(df['text_clean'])
                    y = df['sentiment']
    
                    # Definisikan model SVM
                    clfsvm = svm.SVC(kernel="linear")
    
                    # Gunakan 5-fold Cross-Validation
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    accuracy = cross_val_score(clfsvm, X, y, cv=skf, scoring='accuracy')
                    precision = cross_val_score(clfsvm, X, y, cv=skf, scoring='precision_macro')
                    recall = cross_val_score(clfsvm, X, y, cv=skf, scoring='recall_macro')
                    f1 = cross_val_score(clfsvm, X, y, cv=skf, scoring='f1_macro')
    
                    # Tampilkan hasil evaluasi
                    st.write(f"Rata-rata Akurasi Model: {accuracy.mean():.2f}")
                    st.write(f"Rata-rata Precision Model: {precision.mean():.2f}")
                    st.write(f"Rata-rata Recall Model: {recall.mean():.2f}")
                    st.write(f"Rata-rata F1-Score Model: {f1.mean():.2f}")

                    
                    # Jumlah data uji
                    st.write(f"Jumlah data uji: {len(X_test)}")

                    vectorizer = TfidfVectorizer()
                    X_train = vectorizer.fit_transform(X_train)
                    X_test = vectorizer.transform(X_test)

                    clfsvm = svm.SVC(kernel="linear")
                    clfsvm.fit(X_train,Y_train)
                    predict = clfsvm.predict(X_test)

                    st.write("SVM Accuracy score  -> ", accuracy_score(predict, Y_test)*100)
                    st.write("SVM Recall Score    -> ", recall_score(predict, Y_test, average='macro')*100)
                    st.write("SVM Precision score -> ", precision_score(predict, Y_test, average='macro')*100)
                    st.write("SVM f1 score        -> ", f1_score(predict, Y_test, average='macro')*100)
                    st.write("===========================================================")
                    cm = confusion_matrix(predict, Y_test)
                    # Buat heatmap dari confusion matrix
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                                xticklabels=["negative", "neutral", "positive"],
                                yticklabels=["negative", "neutral", "positive"])
                    plt.xlabel("Predicted Labels")
                    plt.ylabel("True Labels")
                    plt.title("Confusion Matrix")
                    st.pyplot()
                    st.write("===========================================================")
                    st.text('classification report : \n'+ classification_report(predict, Y_test, zero_division=0))
                    st.write("===========================================================")

        except:
            st.write(f'Terjadi kesalahan: {e}')

if __name__ == '__main__':
    main()
