import spacy
import pandas as pd
import numpy as np
import neuralcoref
import ast
import random
import streamlit as st


@st.cache(allow_output_mutation=True)
def reading_data():
    df_score1=pd.read_csv("final_score_1.csv")
    df_score2=pd.read_csv("final_score_2.csv")
    df_score3=pd.read_csv("final_score_3.csv")
    df_score4=pd.read_csv("final_score_4.csv")

    liste_score1=[]
    for string in df_score1.scores:
        liste_score1.append(ast.literal_eval(string))

    liste_score2=[]
    for string in df_score2.scores:
        liste_score2.append(ast.literal_eval(string))

    liste_score3=[]
    for string in df_score3.scores:
        liste_score3.append(ast.literal_eval(string))

    liste_score4=[]
    for string in df_score4.scores:
        liste_score4.append(ast.literal_eval(string))

    scoring = pd.read_csv('scoring.csv', delimiter = ";")

    df_depeche=pd.read_csv('depeches.csv')

    dict_val = {}
    for i in range(48):
        dict_val[scoring['function'][i]] = scoring['score_norm'][i]

    df2=pd.read_csv('photos_clean_final.csv',index_col='Unnamed: 0')

    df_article = pd.read_csv('base_articles.csv')

    nlp=spacy.load("en_core_web_md")
    neuralcoref.add_to_pipe(nlp,greedyness=0.5)

    return nlp,liste_score1, liste_score2, liste_score3, liste_score4, df_depeche, df2, dict_val, df_article

nlp , liste_score1, liste_score2, liste_score3, liste_score4, df_depeche, df2, dict_val, df_article = reading_data()

df2.reset_index(drop=True,inplace=True)
df_depeche.reset_index(drop=True,inplace=True)

def dep_ent(ent, doc):
    start= ent.start
    end=ent.end
    for k in range(start,end):
        if doc[k].head.text not in ent.text:
            if doc[k].dep_=='conj':
                tok=doc[k]
                while tok.dep_=='conj':
                    tok=tok.head
                return(tok.dep_)

            if doc[k].dep_=='compound':
                tok=doc[k]
                while tok.dep_=='compound':
                    tok=tok.head
                return(tok.dep_)
            if doc[k].dep_ =='' or doc[k].dep_=='predet':
                    print('warning')
                    return 'parataxis' #a modifier
            return(doc[k].dep_)

    if doc[start].dep_=='predet':
        return 'parataxis'
    return doc[start].dep_

def ent_good_type(ent): #filtre les entités selon leur type
    return (ent.label_ == "PERSON" or ent.label_ == "NORP" or ent.label_ == "ORG" or ent.label_ == "GPE" or ent.label_ == "EVENT" or ent.label_ == "LOC" or ent.label_=="FAC")



def score_doc_1(doc):
    """sans poids et sans neuralcoref."""
    res={}
    for ent in doc.ents:
        if ent.text not in res.keys():
            res[ent.text]=1
        else:
            res[ent.text]+=1
    return res

def score_doc_2(doc):
    """avec poids et sans neuralcoref"""
    res={}
    for ent in doc.ents:
        if ent.text not in res.keys():
            res[ent.text]=dict_val[dep_ent(ent,doc)]
        else:
            res[ent.text]+=dict_val[dep_ent(ent,doc)]
    return res

def score_doc_3(doc):
    """sans poids et avec neuralcoref"""
    ent_list=[ent for ent in doc.ents if ent_good_type(ent)]
    clusters=doc._.coref_clusters
    res={}
    ent_treated=[]
    for ent in ent_list:
        if ent.text not in ent_treated:
            if ent._.is_coref:
                for cluster in clusters:
                    if ent in cluster.mentions:
                        for mention in cluster.mentions:
                            if mention in doc.ents and mention.text != ent.text:
                                ent_treated.append(mention.text)
                            if ent.text not in res.keys():
                                res[ent.text]=1
                            else:
                                res[ent.text]+=1
            else:
                flag=0
                for cluster in clusters:
                    ent_in_cluster=False
                    for span in cluster.mentions:
                        if ent.text in span.text:
                            ent_in_cluster=True
                            break
                    if ent_in_cluster and ent.label_ != 'NORP':
                        flag=1
                        for mention in cluster.mentions:
                            if ent.text not in res.keys():
                                res[ent.text]=1
                            else:
                                res[ent.text]+=1
                if flag==0:
                    if ent.text not in res.keys():
                        res[ent.text]=1
                    else:
                        res[ent.text]+=1
    return res

def score_doc_4(doc):
    """avec poids et avec neuralcoref"""
    ent_list=[ent for ent in doc.ents if ent_good_type(ent)]
    clusters=doc._.coref_clusters
    res={}
    ent_treated=[]
    for ent in ent_list:
        if ent.text not in ent_treated:
            if ent._.is_coref:
                for cluster in clusters:
                    if ent in cluster.mentions:
                        for mention in cluster.mentions:
                            if mention in doc.ents and mention.text != ent.text:
                                ent_treated.append(mention.text)
                            if ent.text not in res.keys():
                                res[ent.text]=dict_val[dep_ent(mention,doc)]
                            else:
                                res[ent.text]+=dict_val[dep_ent(mention,doc)]
            else:
                flag=0
                for cluster in clusters:
                    ent_in_cluster=False
                    for span in cluster.mentions:
                        if ent.text in span.text:
                            ent_in_cluster=True
                            break
                    if ent_in_cluster and ent.label_ != 'NORP':
                        flag=1
                        for mention in cluster.mentions:
                            if ent.text not in res.keys():
                                res[ent.text]=dict_val[dep_ent(mention,doc)]
                            else:
                                res[ent.text]+=dict_val[dep_ent(mention,doc)]
                if flag==0:
                    if ent.text not in res.keys():
                        res[ent.text]=dict_val[dep_ent(ent,doc)]
                    else:
                        res[ent.text]+=dict_val[dep_ent(ent,doc)]
    return res

def score_sim(doc,score_doc,liste_scores,k):
    res=0
    score_image=liste_scores[k]
    if len(score_image.values())==0:
        return 0
    max_doc=max(score_doc.values())
    max_im=max(score_image.values())
    for i in score_doc.keys():
        if i in score_image.keys():
            if max_doc==0 or max_im==0:
                return 0
            res += score_doc[i]/max_doc +score_image[i]/max_im
    return res

def kw_in_title(keyword):
    index_list=[]
    for k in range(len(liste_score1)):
        caption_title=df2.title[k]
        for word in caption_title.split('-'):
            if word.lower()==keyword.lower():
                index_list.append(k)
                break
    return index_list


def best_n_image(doc,score,liste_scores,n):
    #rel_descr=related_descr(doc)

    doc_score=score(doc)
    if date_filter:
        if format_date==0:
            ind_dates=[i for i in range(len(liste_scores)) if df2.published_date[i][:4]==date_input]
        if format_date==1:
            date=date_input.split('/')
            month, year = date[0], date[1]
            ind_dates=[i for i in range(len(liste_scores)) if df2.published_date[i][:7]==year+'-'+month]
        if format_date==2:
            date=date_input.split('/')
            day, month, year = date[0], date[1], date[2]
            ind_dates=[i for i in range(len(liste_scores)) if df2.published_date[i][:7]==year+'-'+ month +'-'+day]

        ind_dates

    if kw_in_title_filter:
        rel_descr=kw_in_title(keyword)
    else:
        rel_descr=range(len(liste_scores))

    best_scores=[(i,score_sim(doc,doc_score,liste_scores,i)) for i in rel_descr[:n]]
    best_scores.sort(key = lambda x : x[1])
    for k in rel_descr[n:]:
        score_simil=score_sim(doc,doc_score,liste_scores,k)
        if score_simil > best_scores[0][1]:
            best_scores.pop(0)
            best_scores.append((k,score_sim(doc,doc_score,liste_scores,k)))
            best_scores.sort(key = lambda x : x[1])
    return best_scores



'# Interface Stat\'app'



doc=nlp('Donald trump is a shit cat and Emmanuel Macron too.')
#doc.ents
#
# i=st.slider('indice')
# i=0
# liste_scores4[i]
# test(i)

df2.reset_index(drop=True,inplace=True)
df_depeche.reset_index(drop=True,inplace=True)

url_list=df2.url_extracted
#url_list[0]


doc_type=st.sidebar.radio("Choisir la source :",("Base de dépêches", "Base d'articles","Texte personnalisé"))

if doc_type=="Base de dépêches":
    i=st.sidebar.number_input('Numéro de la dépêche',min_value=0,max_value=len(df_depeche),value=0)
    '#### Depeche n°',str(i),':'
    doc=df_depeche.news2[i]
    doc
elif doc_type=="Base d'articles":
    i=st.sidebar.number_input('Numéro de l\'article',min_value=0,max_value=len(df_article),value=0)
    doc=df_article.Article[i]
    df_article.Title[i]
    doc
elif doc_type=="Texte personnalisé":
    doc=st.text_area('Texte à illustrer:')




n=st.sidebar.slider('Nombre d\'images',min_value=1,max_value=10)



display_desc=st.sidebar.checkbox('Afficher les descriptions des images')

kw_in_title_filter=st.sidebar.checkbox('Filtrer selon un certain mot clé')

if kw_in_title_filter:
   keyword = st.sidebar.text_input('Mot clé:')

date_filter=st.sidebar.checkbox('Filtrer selon une certaine date')

if date_filter:
    date_input=st.sidebar.text_input('Date :','DD/MM/YY, MM/YY or YYYY')

    format_date=date_input.count('/')

num_score=st.sidebar.radio("Choisir le score :",("1","2","3","4"))

doc = nlp(doc)
if num_score=='1':
    best_ind=[tup[0]for tup in best_n_image(doc,score_doc_1,liste_score1,n)]
elif num_score=='2':
    best_ind=[tup[0]for tup in best_n_image(doc,score_doc_2,liste_score2,n)]
elif num_score=='3':
    best_ind=[tup[0]for tup in best_n_image(doc,score_doc_3,liste_score3,n)]
elif num_score=='4':
    best_ind=[tup[0]for tup in best_n_image(doc,score_doc_4,liste_score4,n)]


if n>1:
    '#### Voici les ',str(n), 'meilleures photos illustrant le texte:'
else:
    '#### Voici la meilleure photo illustrant le texte:'
#best_score=[tup[1]for tup in best_n_image4(nlp(df_depeche.news2[i]),n)]


if display_desc:
    st.image([url_list[k] for k in best_ind],width=200, caption=[df2.caption[k] for k in best_ind])
else:
    st.image([url_list[k] for k in best_ind],width=250)











