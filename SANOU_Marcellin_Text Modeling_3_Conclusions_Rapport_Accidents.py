#Installation de la librairie Pandas
pip install pandas
#Installation de la librairie Word Cloud pour la visualisation des donnees
pip install wordcloud
#Installation librairie Spacy
pip install spacy
#Téléchargement des modèles Français de la librairie Spacy
#Importations des librairies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk

#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#spacy
import spacy
from spacy.lang.fr.examples import sentences
!python -m spacy download fr_core_news_sm
nlp = spacy.load("fr_core_news_sm")

#Utilisation du Texte 01
#Pré-traitement des données
#Normalisation de la conclusion du Rapport Accident Piper PA34-200T par conversion des mots en minuscules
text1 = "Le pilote s’est désaxé de la trajectoire d’approche aux instruments, se basant probablement sur des références visuelles extérieures erronées, acquises peu avant l’altitude de décision. Celles-ci se situaient significativement à gauche de l’axe suivi lors de l’approche ILS. Cette incohérence ne l’a pas conduit à interrompre immédiatement son approche. Le pilote s’est rendu compte tardivement de sa confusion, à 30 ft au-dessus d’une autoroute. Il a alors débuté une remise de gaz et a perdu le contrôle de son avion au cours de cette manœuvre. Cet accident illustre l’importance de maintenir la cohérence entre les informations issues des instruments de bord et celles fondées sur les références visuelles extérieures, plus particulièrement en conditions IMC et au cours d’une approche ILS pour laquelle les instruments de guidage sont précis. Cette surveillance peut être difficile dans la phase de transition entre l’approche aux instruments et l’approche à vue lorsque les conditions sont marginales."
text1 = text1.lower() #conversion du texte1 en minuscules
print(text1) #Affichage du texte1 en minuscules

#Tokenisation de text1
def return_token(texte):
	# Tokeniser le texte
	doc = nlp(texte)
	# Retourner le texte de chaque token
	return [X.text for X in doc]
return_token(text1)

#Suppression des stopwords
from nltk.corpus import stopwords
stopWords = set(stopwords.words('french'))
print(stopWords) #Affichage en Français de la liste des stopWords

#Pour filtrer le contenu du text1, on enlève tous les stopwords présents dans cette liste :
clean_words = []
for token in return_token(text1):
	if token not in stopWords:
		clean_words.append(token)
clean_words

#Convertion de la liste clean_words en string pour réaliser le stemming
mystring1 = ' '.join(clean_words)
print(mystring1)

#Stemming du text1
#Création du Stemmer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='french')
def return_stem(texte):
	doc = nlp(texte)
	return [stemmer.stem(X.text) for X in doc]

#Appliquons ce stemmer à mystring1
t1 = return_stem(mystring1)
print(t1)

#Suppression des chiffres dans le texte
text1nodigit = ' '.join([i for i in t1 if not i.isdigit()])
print(text1nodigit)

#Convertissons text1nodigit en liste
def convert(string):
	li = list(string.split(" "))
	return li
list1 = convert(text1nodigit)
print(list1)

#Utilisons la librairie sklearn pour importer le vectorizer TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
tfIdfVectorizer=TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(list1)
df1 = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df1 = df1.sort_values('TF-IDF', ascending=False)
print (df1)

#Visualisation des donnees avec la librairie Word Cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
def show_wordcloud(data):
	wordcloud = WordCloud(
		background_color='white',
		stopwords=stopwords,
		max_words=100,
		max_font_size=30,
		scale=3,
		random_state=1)
	wordcloud=wordcloud.generate(str(data))
	fig = plt.figure(1, figsize=(12, 12))
	plt.axis('off')
	plt.imshow(wordcloud)
	plt.show()

show_wordcloud(df1)

#Topic Modeling avec LDA
from nltk.corpus import stopwords  #stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1) 
lda_top=lda_model.fit_transform(df1)

for i,topic in enumerate(lda_top):
  print("Topic ",i,": ",topic*100,"%")

vocab = tfIdfVectorizer.get_feature_names()
for i, comp in enumerate(lda_model.components_):
	 vocab_comp = zip(vocab, comp)
	 sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)
	 print("Topic "+str(i)+": ")
	 for t in sorted_words:
			print(t,end=" ")
			print("\n")

#Topic Modeling sans étape de TF-IDF
#Visualisation des données avec Word Cloud
#Visualisation des donnees avec la librairie Word Cloud
#Importation des librairies Word Cloud et Matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
def show_wordcloud(data):
	wordcloud = WordCloud(
		background_color='white',
		stopwords=stopwords,
		max_words=100,
		max_font_size=30,
		scale=3,
		random_state=1)
	wordcloud=wordcloud.generate(str(data))
	fig = plt.figure(1, figsize=(12, 12))
	plt.axis('off')
	plt.imshow(wordcloud)
	plt.show()
show_wordcloud(list1)

#Topic Modeling avec LDA
def gen_words(texts):
	final = []
	for text in texts:
		new = gensim.utils.simple_preprocess(text, deacc=True)
		final.append(new)
	return (final)
data_words1 = gen_words(list1)
print(data_words1)

id2word = corpora.Dictionary(data_words1)
corpus = []
for text in data_words1:
	new = id2word.doc2bow(text)
	corpus.append(new)
print (corpus[0][0:20])
word1 = id2word[[0][:1][0]]
print (word1)
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
										   id2word=id2word,
										   num_topics=30,
										   random_state=100,
										   update_every=1,
										   chunksize=100,
										   passes=10,
										   alpha="auto")


#Utilisation du Texte02
#Pré-traitement des données
#Normalisation de la conclusion du Rapport ACCIDENT survenu à l’avion immatriculé N429CA par conversion des mots en minuscules
text2 = "L’accident est probablement dû à l’application lors de l’atterrissage à l’aéroport de Limoges d’un effort sur le train avant excédant sa capacité de résistance, affectée par des endommagements précédents. Cet effort a provoqué la progression d’une fissure antérieure, qui s’est poursuivie jusqu’à la rupture statique du bâti. Cet effort n’a pas été rapporté par le pilote. Via la lettre d’information n°00-560-03 éditée en juillet 2000 par le concepteur de la remotorisation, l’exploitant avait été averti :-d’une part de l’occurrence de deux cas d’apparition de criques sur des bâtis moteurs du même type que celui du N429CA,-et d’autre part de la possibilité d’installer sur l’avion une version renforcée du bâti. Malgré cette lettre service, l’exploitant a choisi de ne pas réaliser rapidement cette modification, qui aurait pu éviter l’accident. Le fait que l’exploitant n’ait pu détecter un endommagement présent avant le vol de l’accident a contribué à l’événement."
text2 = text2.lower() #conversion du texte2 en minuscules
print(text2) #Affichage du texte2 en minuscules 

#Tokenisation de text2
def return_token(texte):
	# Tokeniser le texte
	doc = nlp(texte)
	# Retourner le texte de chaque token
	return [X.text for X in doc]
return_token(text2)

#Suppression des stopwords
from nltk.corpus import stopwords
stopWords = set(stopwords.words('french'))
print(stopWords) #Affichage en Français de la liste des stopWords
#Pour filtrer le contenu du text2, on enlève tous les stopwords présents dans cette liste:
clean_words2 = []
for token in return_token(text2):
	if token not in stopWords:
		clean_words2.append(token)
clean_words2
#Convertion de la liste clean_words2 en string pour réaliser le stemming
mystring2 = ' '.join(clean_words2)
print(mystring2)

#Stemming du text2
#Création du Stemmer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='french')
def return_stem(texte):
	doc = nlp(texte)
	return [stemmer.stem(X.text) for X in doc]
#Appliquons ce stemmer à mystring2
t2 = return_stem(mystring2)
print(t2)

#Suppression des chiffres dans le texte
text2nodigit = ' '.join([i for i in t2 if not i.isdigit()])
print(text2nodigit)
#Convertissons text2nodigit en liste
def convert(string):
	li = list(string.split(" "))
	return li
list2 = convert(text2nodigit)
print(list2)

#Installation de la librairie Pandas si il y a lieu
pip install pandas
#Utilisons la librairie pandas et la librairie sklearn pour importer le vectorizer TF-IDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
tfIdfVectorizer=TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(list2)
df2 = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df2 = df2.sort_values('TF-IDF', ascending=False)
print (df2)

#Visualisation des donnees avec Word Cloud
#Visualisation des donnees avec la librairie Word Cloud
#Importation des librairies Word Cloud et Matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
def show_wordcloud(data):
	wordcloud = WordCloud(
		background_color='white',
		stopwords=stopwords,
		max_words=100,
		max_font_size=30,
		scale=3,
		random_state=1)
	wordcloud=wordcloud.generate(str(data))
	fig = plt.figure(1, figsize=(12, 12))
	plt.axis('off')
	plt.imshow(wordcloud)
	plt.show()
show_wordcloud(df2)

#Topic Modeling avec LDA
import nltk
from nltk.corpus import stopwords  #stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1) 
lda_top=lda_model.fit_transform(df2)
for i,topic in enumerate(lda_top):
  print("Topic ",i,": ",topic*100,"%")
vocab = tfIdfVectorizer.get_feature_names()
for i, comp in enumerate(lda_model.components_):
	 vocab_comp = zip(vocab, comp)
	 sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)
	 print("Topic "+str(i)+": ")
	 for t in sorted_words:
			print(t,end=" ")
			print("\n")


#Topic Modeling sans étape de TF-IDF
#Visualisation des donnees avec la librairie Word Cloud
#Importation des librairies Word Cloud et Matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
def show_wordcloud(data):
	wordcloud = WordCloud(
		background_color='white',
		stopwords=stopwords,
		max_words=100,
		max_font_size=30,
		scale=3,
		random_state=1)
	wordcloud=wordcloud.generate(str(data))
	fig = plt.figure(1, figsize=(12, 12))
	plt.axis('off')
	plt.imshow(wordcloud)
	plt.show()
show_wordcloud(list2)

#Topic Modeling avec LDA
#Importation des librairies GENSIM
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
def gen_words(texts):
	final = []
	for text in texts:
		new = gensim.utils.simple_preprocess(text, deacc=True)
		final.append(new)
	return (final)
data_words2 = gen_words(list2)
print(data_words2)

id2word = corpora.Dictionary(data_words2)
corpus = []
for text in data_words2:
	new = id2word.doc2bow(text)
	corpus.append(new)
print (corpus[0][0:20])
word2 = id2word[[0][:1][0]]
print (word2)
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
										   id2word=id2word,
										   num_topics=30,
										   random_state=100,
										   update_every=1,
										   chunksize=100,
										   passes=10,
										   alpha="auto")


#Utilisation du Texte03
#Pré-traitement des données
#Normalisation de la conclusion du Rapport ACCIDENT BEA - Vol à faible hauteur au second régime dans le cadre d’une mission de lutte contre un feu de forêt, perte de référence visuelle, collision avec le relief par conversion des mots en minuscules
text3 = "L’accident résulte d’une décision inappropriée de l’équipage dans le choix du cheminement utilisé pour réaliser un largage de barrage. Ce choix a probablement été dicté par le fait que le copilote, pilote en fonction, pouvait mieux contrôler de sa place, notamment, le virage à droite sur la ville de Burzet. L’évolution à proximité du relief était inadaptée pour un avion de ce gabarit et nécessitait de la part de l’équipage un pilotage aux limites du domaine de vol. La dernière phase de vol, réalisée sans doute au second régime de vol, n’a pas permis au pilote de reprendre suffisamment de hauteur pour s’affranchir du relief. La charge de travail du copilote, pilote en fonction, ne lui a pas laissé le temps d’exécuter un largage d’urgence pour alléger l’avion et dégager dans la vallée. Le soleil de face et la présence de fumée dans la dernière ligne droite ont amené l’équipage à faire une appréciation erronée du relief. L’euphorie du dernier vol de la saison et du dernier largage a pu constituer un facteur contributif."
text3 = text3.lower() #conversion du texte3 en minuscules
print(text3) #Affichage du texte3 en minuscules

#Tokenisation de text3
def return_token(texte):
    # Tokeniser le texte
    doc = nlp(texte)
    # Retourner le texte de chaque token
    return [X.text for X in doc]
return_token(text3)

#Suppression des stopwords
from nltk.corpus import stopwords
stopWords = set(stopwords.words('french'))
print(stopWords) #Affichage en Français de la liste des stopWords
#Pour filtrer le contenu du text3, on enlève tous les stopwords présents dans cette liste:
clean_words3 = []
for token in return_token(text3):
    if token not in stopWords:
        clean_words3.append(token)
clean_words3
#Convertion de la liste clean_words3 en string pour réaliser le stemming
mystring3 = ' '.join(clean_words3)
print(mystring3)

#Stemming du text3
#Création du Stemmer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='french')
def return_stem(texte):
    doc = nlp(texte)
    return [stemmer.stem(X.text) for X in doc]
#Appliquons ce stemmer à mystring3
t3 = return_stem(mystring3)
print(t3)

#Suppression des nombres dans le texte
text3nodigit = ' '.join([i for i in t3 if not i.isdigit()])
print(text3nodigit)
#Convertissons text3nodigit en liste
def convert(string):
    li = list(string.split(" "))
    return li
list3 = convert(text3nodigit)
print(list3)

#Installation de la librairie Pandas si il y a lieu
pip install pandas
#Utilisons la librairie pandas et la librairie sklearn pour importer le vectorizer TF-IDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
tfIdfVectorizer=TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(list3)
df3 = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df3 = df3.sort_values('TF-IDF', ascending=False)
print (df3)

#Visualisation des donnees avec Word Cloud
#Visualisation des donnees avec la librairie Word Cloud
#Importation des librairies Word Cloud et Matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)
    wordcloud=wordcloud.generate(str(data))
    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()
show_wordcloud(df3)

#Utilisation du Modèle LDA
import nltk
from nltk.corpus import stopwords  #stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1) 
lda_top=lda_model.fit_transform(df3)
for i,topic in enumerate(lda_top):
  print("Topic ",i,": ",topic*100,"%")
vocab = tfIdfVectorizer.get_feature_names()
for i, comp in enumerate(lda_model.components_):
     vocab_comp = zip(vocab, comp)
     sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)
     print("Topic "+str(i)+": ")
     for t in sorted_words:
            print(t,end=" ")
            print("\n")

#Visualisation des donnees avec la librairie Word Cloud
#Importation des librairies Word Cloud et Matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)
    wordcloud=wordcloud.generate(str(data))
    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()
show_wordcloud(list3)

#Topic Modeling avec LDA
#Importation des librairies GENSIM
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)
data_words3 = gen_words(list3)
print(data_words3)

id2word = corpora.Dictionary(data_words3)
corpus = []
for text in data_words3:
    new = id2word.doc2bow(text)
    corpus.append(new)
print (corpus[0][0:20])
word3 = id2word[[0][:1][0]]
print (word3)
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=30,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha="auto")

