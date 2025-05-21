
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
        st.caption('© HerlambangNaufal 2025')

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
            data_file = st.file_uploader("Upload CSV file", type=["csv"])            
            if data_file is not None:
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
                        text = re.sub(r"\d+", "", text)
                        text = text.encode('ascii', 'replace').decode('ascii')
                        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
                        text = re.sub(r'[^a-zA-Z]', ' ', text)
                        text = re.sub(r'\b[a-zA-Z]\b', ' ', text)
                        text = re.sub(r'(.)\1+', r'\1\1', text)
                        text = re.sub(r'[\?\.\!]+(?=[\?.\!])', '', text)
                        text = re.sub('\s+', ' ', text)
                        text = text.translate(text.maketrans("", "", string.punctuation))
                        text = text.strip()
                        text = ' '.join(dict.fromkeys(text.split()))
                        return text
    
                    # Case folding text
                    def casefolding(text):
                        return text.lower()
    
                    # Tokenize text
                    def tokenize(text):
                        return word_tokenize(text)
    
                    # Normalisasi text
                    normalizad_word = pd.read_excel("colloquial-indonesian-lexicon.xlsx")
                    normalizad_word_dict = {row[0]: row[1] for _, row in normalizad_word.iterrows()}
    
                    def normalized_term(text):
                        return [normalizad_word_dict.get(term, term) for term in text]
    
                    # Filtering | stopwords removal
                    def stopword(text):
                        listStopwords = set(stopwords.words('indonesian'))
                        listStopwords.update(["nya", "ya"])
                        return [txt for txt in text if txt not in listStopwords]
    
                    # Remove punctuation
                    def remove_punct(text):
                        return " ".join([char for char in text if char not in string.punctuation])
    
                    # Preprocessing
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
    
                    df['text_clean'] = df['text_stopword'].apply(remove_punct)
                    df['text_clean'].replace('', np.nan, inplace=True)
                    df.dropna(subset=['text_clean'], inplace=True)
                    df = df.reset_index(drop=True)
    
                    st.write("Finish Pre-processing")
                    st.write("===========================================================")
                
                    # Load Lexicon
                    st.write("Count Polarity and Labeling...")
                    st.caption("Using Indonesia Sentiment Lexicon")
                    lexicon = dict()
                    import csv
                    with open('InSet_Lexicon.csv', 'r') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',')
                        for row in reader:
                            lexicon[row[0]] = int(row[1])
    
                    # Fungsi sentiment analysis (mengembalikan total_score, polarity, dan word_scores)
                    def sentiment_analysis_lexicon_indonesia(text):
                        word_scores = []
                        total_score = 0
                        for word in text:
                            score = lexicon.get(word, 0)
                            word_scores.append(score)
                            total_score += score
    
                        polarity = 'neutral'
                        if total_score > 0:
                            polarity = 'positive'
                        elif total_score < 0:
                            polarity = 'negative'
                        
                        return total_score, polarity, word_scores
    
                    # Terapkan analisis sentimen dan tambahkan kolom baru di bagian belakang
                    results = df['text_stopword'].apply(sentiment_analysis_lexicon_indonesia)
                    results = list(zip(*results))
                    df['score'] = results[0]
                    df['sentiment'] = results[1]
                    df['word_scores'] = results[2]
    
                    st.text(df['sentiment'].value_counts())
                    st.dataframe(df)
                    st.download_button(
                        label='Download CSV',
                        data=df.to_csv(index=False, encoding='utf8'),
                        file_name='Labeled_Output.csv',
                        on_click=callback
                    )
    
        except Exception as e:
            st.write('Select The Correct File')
            st.write(e)
    with tab3:
        try:
            data_file = st.file_uploader("Upload Labeled CSV file",type=["csv"])
            if data_file is not None:
                df = pd.read_csv(data_file)
                st.dataframe(df)

                positif = len(df[df['sentiment'] == "positive"])
                negatif = len(df[df['sentiment'] == "negative"])
                netral = len(df[df['sentiment'] == "neutral"])

                docPositive = df[df['sentiment']=='positive'].reset_index(drop=True)
                docNegative = df[df['sentiment']=='negative'].reset_index(drop=True)
                docNeutral = df[df['sentiment']=='neutral'].reset_index(drop=True)

                option = st.radio('ingin lihat data apa? ',['Positive','Negative','Neutral'])
                if option == 'Positive':
                    st.write("========================================================================================")
                    st.write('Document Positive Sentiment')
                    st.caption(f"Positive = {positif}, {docPositive.shape[0]/df.shape[0]*100} % ")
                    st.dataframe(docPositive)
                elif option == 'Negative':
                    st.write("========================================================================================")
                    st.write('Document Negative Sentiment')
                    st.caption(f"Negative = {negatif}, {docNegative.shape[0]/df.shape[0]*100} % ")
                    st.dataframe(docNegative)
                else:
                    st.write("========================================================================================")
                    st.write('Document Neutral Sentiment')
                    st.caption(f"Neutral = {netral}, {docNeutral.shape[0]/df.shape[0]*100} % ")
                    st.dataframe(docNeutral)

                st.write("========================================================================================")
                try:
                    text = " ".join(df['text_clean'])
                    wordcloud = WordCloud(width = 600, height = 400, background_color = 'black', min_font_size = 10).generate(text)
                    fig, ax = plt.subplots(figsize = (8, 6))
                    ax.set_title('WordCloud of Comment Data', fontsize = 18)
                    ax.grid(False)
                    ax.imshow((wordcloud))
                    fig.tight_layout(pad=0)
                    ax.axis('off')
                    st.pyplot(fig)
                except:
                    st.write(' ')

                try:
                    st.write('WordCloud Positive')
                    train_s0 = df[df["sentiment"] == 'positive']
                    text = " ".join((word for word in train_s0["text_clean"]))
                    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=700, height=400,colormap='Blues', mode='RGBA').generate(text)
                    fig, ax = plt.subplots(1,figsize=(13, 13))
                    ax.set_title('WordCloud Positive', fontsize = 18)
                    ax.imshow(wordcloud, interpolation = 'bilinear')
                    plt.axis('off')
                    st.pyplot(fig)
                except:
                    st.write('tidak ada sentiment positif pada data')

                try:
                    st.write('WordCloud Negative')
                    train_s0 = df[df["sentiment"] == 'negative']
                    text = " ".join((word for word in train_s0["text_clean"]))
                    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=700, height=400,colormap='Reds', mode='RGBA').generate(text)
                    fig, ax = plt.subplots(1,figsize=(13, 13))
                    ax.set_title('WordCloud Negative', fontsize = 18)
                    ax.imshow(wordcloud, interpolation = 'bilinear')
                    plt.axis('off')
                    st.pyplot(fig)
                except:
                    st.write('tidak ada sentiment negatif pada data')
                try:
                    st.write('WordCloud Neutral')
                    train_s0 = df[df["sentiment"] == 'neutral']
                    text = " ".join((word for word in train_s0["text_clean"]))
                    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=700, height=400,colormap='Greens', mode='RGBA').generate(text)
                    fig, ax = plt.subplots(1,figsize=(13, 13))
                    ax.set_title('WordCloud Neutral', fontsize = 18)
                    ax.imshow(wordcloud, interpolation = 'bilinear')
                    plt.axis('off')
                    st.pyplot(fig)
                except:
                    st.write('tidak ada sentiment neutral pada data')

                try:
                    st.write('Pie Chart')
                    def pie_chart(label, data, legend_title):
                        fig, ax = plt.subplots(figsize=(5,7), subplot_kw=dict(aspect='equal'))

                        labels = [x.split()[-1] for x in label]

                        def func(pct, allvals):
                            absolute = int(np.round(pct/100.*np.sum(allvals)))
                            return "{:.1f}% ({:d})".format(pct, absolute)

                        wedges, texts, autotexts = ax.pie(data, autopct = lambda pct: func(pct, data),
                            textprops = dict(color="w"))

                        ax.legend(wedges, labels, title = legend_title,
                    loc = "center left",
                    bbox_to_anchor=(1,0,0.25,1))
                        plt.setp(autotexts, size=6, weight="bold")
                        st.pyplot(fig)

                    label = ['Positif', 'Negatif','Neutral']
                    count_data =[positif, negatif, netral]

                    pie_chart(label, count_data, "status")
                except:
                    st.caption('')
                st.spinner(text="In progress...")

                try:
                    st.write('Word Frequency')
                    top = 11
                    a = df['text_clean'].str.cat(sep=' ')
                    words = nltk.tokenize.word_tokenize(a)
                    Word_dist = nltk.FreqDist(words)
                    rslt = pd.DataFrame(Word_dist.most_common(top), columns=['Word', 'Frequency'])

                    count = rslt['Frequency']

                    fig, x = plt.subplots(1,1,figsize=(11,8))
                    # create bar plot
                    plt.bar(rslt['Word'], count, color=['royalblue'])

                    plt.xlabel('\nKata', size=14)
                    plt.ylabel('\nFrekuensi Kata', size=14)
                    plt.title('Kata yang sering Keluar \n', size=16)
                    st.pyplot(fig)

                except:
                    st.write('error')
                    
                st.write("====================================================================")
                st.subheader("Analisis Perbandingan Tahun dan Alasan Penurunan")

                # Filter Data Berdasarkan Tahun
                df['at'] = pd.to_datetime(df['at'])
                df['year'] = df['at'].dt.year

                # Statistik Awal Berdasarkan Tahun
                year_options = [2021,2022, 2023, 2024,2025]
                for year in year_options:
                    yearly_data = df[df['year'] == year]
                    st.write(f"Tahun {year}:")
                    st.write(f"- Jumlah Ulasan: {len(yearly_data)}")
                    st.write(f"- Rata-Rata Rating: {yearly_data['score'].mean():.2f}")
                    sentiment_counts = yearly_data['sentiment'].value_counts()
                    st.write(f"- Positif: {sentiment_counts.get('positive', 0)}")
                    st.write(f"- Negatif: {sentiment_counts.get('negative', 0)}")
                    st.write(f"- Netral: {sentiment_counts.get('neutral', 0)}")
                    st.write(" ")
                st.write("====================================================================")
                
                # Total jumlah ulasan per tahun menggunakan rating original     
                total_reviews_per_year = df.groupby('year')['score_original'].count()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = total_reviews_per_year.plot(kind='bar', ax=ax, color='royalblue')
                
                ax.set_title("Total Jumlah Ulasan per Tahun", fontsize=16)
                ax.set_xlabel("Tahun")
                ax.set_ylabel("Jumlah Ulasan")
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Tambahkan angka jumlah ulasan di atas setiap batang
                for bar in bars.patches:
                    ax.annotate(f"{int(bar.get_height())}", 
                                (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
                
                st.pyplot(fig)
                                
                # Hitung rata-rata rating asli per tahun
                avg_rating_per_year = df.groupby('year')['score_original'].mean()
                
                # Visualisasi
                fig, ax = plt.subplots(figsize=(10, 5))
                avg_rating_per_year.plot(kind='line', marker='o', ax=ax, color='blue')
                ax.set_title("Tren Rata-Rata Rating Google Play Store per Tahun", fontsize=16)
                ax.set_xlabel("Tahun")
                ax.set_ylabel("Rata-Rata Rating (1-5)")
                ax.grid(True)
                st.pyplot(fig)
                # Visualisasi Rata-Rata Rating per Tahun
                avg_rating_per_year = df.groupby('year')['score'].mean()
                st.write("Rata-Rata Rating per Tahun")
                fig, ax = plt.subplots(figsize=(10, 5))
                avg_rating_per_year.plot(kind='line', marker='o', ax=ax)
                ax.set_title("Rata-Rata Skor Sentimen per Tahun", fontsize=16)
                ax.set_xlabel("Tahun")
                ax.set_ylabel("Rata-Rata Rating (1-5)")
                ax.grid(True)
                st.pyplot(fig)

                # distribusi sentimen per tahun
                sentiment_distribution = df.groupby('year')['sentiment'].value_counts().unstack().fillna(0) 
                st.write("Distribusi Sentimen per Tahun")
                colors = {"positive": "#25993f", "negative": "#bf3d3d", "neutral": "#cfaf3c"}
                fig, ax = plt.subplots(figsize=(10, 6))
                sentiment_distribution.plot(kind='bar', ax=ax, width=0.7, color=[colors[col] for col in sentiment_distribution.columns])
                ax.set_title("Distribusi Sentimen per Tahun", fontsize=16)
                ax.set_xlabel("Tahun", fontsize=12)
                ax.set_ylabel("Jumlah Ulasan", fontsize=12)
                ax.legend(title="Sentimen")  # Menambahkan legenda
                st.pyplot(fig)
        except:
            st.write('Select The Correct File')
            
    with tab4:
        try:
            import seaborn as sns
            data_file = st.file_uploader("Upload labeled CSV file", type=["csv"])
            if data_file is not None:
                df = pd.read_csv(data_file)
                st.dataframe(df)
    
                proseseval = st.button('Start process', key='start_process_btn_tab4')
    
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
    
                    df['sentiment'] = df['sentiment'].apply(score_sentiment)
    
                    X_train, X_test, Y_train, Y_test = train_test_split(
                        df['text_clean'], df['sentiment'], 
                        test_size=0.2, stratify=df['sentiment'], random_state=42
                    )
    
                    jumlah_data_latih_positive = sum(Y_train == "positive")
                    jumlah_data_latih_negative = sum(Y_train == "negative")
                    jumlah_data_latih_neutral = sum(Y_train == "neutral")
    
                    st.write("Data Latih:")
                    st.write(f"Jumlah data latih dengan sentimen positive: {jumlah_data_latih_positive}")
                    st.write(f"Jumlah data latih dengan sentimen negative: {jumlah_data_latih_negative}")
                    st.write(f"Jumlah data latih dengan sentimen neutral: {jumlah_data_latih_neutral}")
    
                    st.write("====================================================================")
    
                    # Konversi text_clean ke fitur numerik menggunakan satu TF-IDF vectorizer
                    vectorizer = TfidfVectorizer()
                    X_train = vectorizer.fit_transform(X_train)
                    X_test = vectorizer.transform(X_test)
    
                    # Buat DataFrame TF-IDF
                    tfidf_df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
    
                    # Hitung rata-rata TF-IDF untuk setiap kata
                    tfidf_mean = tfidf_df.mean().sort_values(ascending=False)
    
                    # Definisikan model SVM
                    clfsvm = svm.SVC(kernel="linear")
                    clfsvm.fit(X_train, Y_train)
                    predict = clfsvm.predict(X_test)
    
                    st.write(f"Jumlah data uji: {X_test.shape[0]}")
                    st.write("SVM Accuracy score  -> ", accuracy_score(Y_test, predict) * 100)
                    st.write("SVM Recall Score    -> ", recall_score(Y_test, predict, average='macro') * 100)
                    st.write("SVM Precision score -> ", precision_score(Y_test, predict, average='macro') * 100)
                    st.write("SVM f1 score        -> ", f1_score(Y_test, predict, average='macro') * 100)
    
                    st.write("===========================================================")
    
                    cm = confusion_matrix(Y_test, predict)
    
                    # Buat heatmap dari confusion matrix
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                                xticklabels=["negative", "neutral", "positive"],
                                yticklabels=["negative", "neutral", "positive"])
                    plt.xlabel("Predicted Labels")
                    plt.ylabel("True Labels")
                    plt.title("Confusion Matrix")
                    st.pyplot(fig)
    
                    st.write("===========================================================")
                    st.text('classification report : \n' + classification_report(Y_test, predict, zero_division=0))
    
        except Exception as e:
            st.write(f'Terjadi kesalahan: {e}')

if __name__ == '__main__':
    main()
