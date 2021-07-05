
import os, requests, uuid, json, pandas as pd, pycountry, matplotlib.pyplot as plt, seaborn as sns
from IPython.core.display import display, HTML
from zipfile import ZipFile
import sys


key_var_name = 'TRANSLATOR_TEXT_SUBSCRIPTION_KEY'
if not key_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(key_var_name))
subscription_key = os.environ[key_var_name]

endpoint_var_name = 'TRANSLATOR_TEXT_ENDPOINT'
if not endpoint_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(endpoint_var_name))
endpoint = os.environ[endpoint_var_name]

path = '/detect?api-version=3.0'
constructed_url = endpoint + path

headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}


#url = 'https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+1+Discover+the+AI%C2%A0Engineer+Job/Dataset+project+1+AI%C2%A0Engineer.zip'
#r = requests.get(url, allow_redirects=True)
#open('source.zip', 'wb').write(r.content)
#file_name = "source.zip"
#with ZipFile(file_name, 'r') as zip:
#    zip.printdir()
#    zip.extractall()


# extraire les données du fichier texte des paragraphes
data_x = pd.read_fwf("Dataset project 1 AI┬áEngineer/x_test.txt" , header=None)


# In[7]:


#On crée un DataFrame avec les données de la 1ere colonne
x = pd.DataFrame({'Paragraphe': data_x[0]})


# In[8]:


# extraire les données du fichier texte des codes langues
data_y = pd.read_fwf("Dataset project 1 AI┬áEngineer/y_test.txt" , header=None)


# In[9]:


#On crée un DataFrame avec les données
y = pd.DataFrame({'Code': data_y[0]})


# In[10]:


#On fusionne les deux DataFrames
data = y.join(x)

code_lang = ['eng', 'hin', 'spa', 'ara', 'zho', 'fra']

#Extraction des correspondance de code langue a partir du fichier labels.csv
lang = pd.read_csv("Dataset project 1 AI┬áEngineer/labels.csv", sep=';')


#On filtre en gardant que les lignes dont la colonne ISO369-3 contient un des codes dans notre liste code_lang
df = lang.iloc[[index for index,row in lang.iterrows() if row['ISO 369-3'] in (code_lang)]]

#On Garde que les colonnes de la Langue nommée 'English' que l'on renome Langue et la colonne ISO 369-3
top5 = df.loc[df.index[:],["English","ISO 369-3"]].rename(columns={'English': 'Langue'})


#Ajout d'une colonne norme ISO 369-1 (2 lettres) à partir de la conversion ISO 369-3 grace au module Pycountry
top5['ISO 369-1'] = [pycountry.languages.get(alpha_3=row).alpha_2 for row in top5['ISO 369-3']]


# On garde uniquement les lignes dans les langues selectionnées 
data_tri = data.iloc[[index for index,row in data.iterrows() if row['Code'] in (code_lang)]].reset_index()

# On cree une fonction qui prend en entrée la langue et le nombre de paragraphe et qui retourne tous les paragraphes dans cette langue
def get_para_lang(lang,nb_tests):
    return data_tri.iloc[[index for index,row in data_tri.iterrows() if row['Code'] == lang ]][:nb_tests].reset_index().Paragraphe

# On creer le DataFrame vide qui la contenir les paragraphes pour les 5 langues
df5 =  pd.DataFrame()

#On effectue le test de detection n fois avec les paragraphes différents avec le service Azure
def paragraphes():
    global top5
    global df5
    a = 0
    for n in code_lang:

        for i in get_para_lang(n, 2):
            body = [{'text': i}]
            request = requests.post(constructed_url, headers=headers, json=body)
            response = request.json()
            df5 = df5.append(response, ignore_index=True) 
            df5.loc[a, 'Langage Envoyé'] = n
            df5.loc[a, 'Correspondance'] = False
            a += 1

    df6 = df5.copy()

    df6['Langage Envoyé'] = [pycountry.languages.get(alpha_3=row).alpha_2 for row in df6['Langage Envoyé']]

    df6['language'] = [row[:row.index('-')] if '-' in row else row for row in df6['language']]

    df6.rename(columns={'language':'Langage Reconnu'}, inplace=True)

    df6 = df6[['Langage Envoyé', 'Langage Reconnu', 'Correspondance', 'score']]

    df7 = df6.copy()

    for i in [index for index,row in df7.iterrows() if row['Langage Envoyé'] == row['Langage Reconnu']]:
        df7.loc[i, 'Correspondance'] = 'ok'

    for i in [index for index,row in df7.iterrows() if row['Langage Envoyé'] != row['Langage Reconnu']]:
        df7.loc[i, 'Correspondance'] = 'no'

    # On garde la langue et le score si les langues correspandent
    df7 = df7[['Langage Reconnu','score']][df7['Correspondance'] == 'ok']

    #On calcul le pourcentage moyen des scores par langue
    p100 = df7.groupby(by='Langage Reconnu')['score'].mean()*100

    datafinal = p100.to_frame()

    datafinal = datafinal.reset_index()

    top5 = top5.reset_index()

    for i,r in datafinal.iterrows():
        top5.loc[i, 'Score en %'] = r['score']

    top5 = top5[['Langue','Score en %']].set_index('Langue')

    sns.set_theme(style="whitegrid")
    sns.barplot(x=top5.index, 
                y=top5['Score en %'], 
                data=top5)
    return plt.show()

def input_text():
    global top5
    i = input('entrer un texte :')
    body = [{'text': i}]
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    a = response[0]["language"]
    c = response[0]["score"]
    b = top5.loc[(top5['ISO 369-1']==a),['Langue']].iloc[0][0]
    return print(f'The detected language is {b} with a score of {c}')
    
if sys.argv[1] == '--test':
    input_text()
elif sys.argv[1] == '--eval':
    paragraphes()
else:
    print('\nArgument requi : \n--eval pour tester la detection à partir du jeu de donnée \n--text pour taper un texte à tester\n')
# %%
