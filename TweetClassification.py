#Nom: Josep Bravo Bravo
#NIU: 1526453


#processar tweets amb Lancaster Stemmer
import numpy  as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def open_fitxer():    
    df = pd.read_csv("FinalStemmedSentimentAnalysisDataset.csv", delimiter = ';') 
    #df = df.head(3)
    df = df.dropna(how='any') #eliminar totes les files amb algun nullç
    numpy_df = df.to_numpy()
    
    X = numpy_df[:,1] #Quedarte nomes amb tweet text
    Y = numpy_df[:,3] #Quedarte nomes amb lobjectiu sentimentlabel
    Y = Y.astype('int64') 
    return X, Y
    
def create_dicts(X, Y, n):   
    
    dict_X_sentiment0 = dict()
    dict_X_sentiment1 = dict()
    
    
    index = np.where(Y== int(0))
    np_X_sentiment0 = np.take(X, index[0])
    
    index = np.where(Y== int(1))
    np_X_sentiment1 = np.take(X, index[0])

    
    X_flat0 = ''.join(np_X_sentiment0).split()
    X_flat1 = ''.join(np_X_sentiment1).split()
   
    
    for word in X_flat0: 
        if word in dict_X_sentiment0:
            dict_X_sentiment0[word] += 1
        else:
            dict_X_sentiment0[word] = 1
    
    for word in X_flat1:
        if word in dict_X_sentiment1:
            dict_X_sentiment1[word] += 1
        else:
            dict_X_sentiment1[word] = 1
                
    
    #creem diccionaris amb els n elements mes grans
    if n != -1:

        #ordenem el diccionari (tuples de items), per el valor (més frequent).
        #Un cop ordenat creem un diccionari amb els n primers valors, que seran els mes frequents 
        f_dict_X_sentiment0 = dict(sorted(dict_X_sentiment0.items(), key=lambda x: x[1], reverse=True)[:n])
        f_dict_X_sentiment1 = dict(sorted(dict_X_sentiment1.items(), key=lambda x: x[1], reverse=True)[:n])
                      
    else:
        f_dict_X_sentiment0 = dict_X_sentiment0
        f_dict_X_sentiment1 = dict_X_sentiment1
            
    
    return f_dict_X_sentiment0, f_dict_X_sentiment1
    
def prediccio(x_test, dict_sentiment_0, dict_sentiment_1):
    
    sum_dict_0 = sum(dict_sentiment_0.values())
    sum_dict_1 = sum(dict_sentiment_1.values())
    
    prob0_inicial = sum_dict_0/ sum_dict_0 + sum_dict_1
    prob1_inicial = sum_dict_1/ sum_dict_0 + sum_dict_1
    
    len_dict0 = len(dict_sentiment_0)
    len_dict1 = len(dict_sentiment_1)
    
    y_pred = np.zeros(len(x_test), dtype = 'int64')
    for i in range(len(x_test)):
        txt_flat = x_test[i].split() #llista de paraules
        prob0 = prob0_inicial
        prob1 = prob1_inicial
        for word in txt_flat:
            count_word_in_0 = dict_sentiment_0.get(word, 0)
            count_word_in_1 = dict_sentiment_1.get(word, 0)
            
            #laplace smoothing
            prob0 *= ((count_word_in_0 + 1)/(sum_dict_0 + len_dict0))
            prob1 *= ((count_word_in_1 + 1)/(sum_dict_1 + len_dict1))
            
            
            '''
            #sense laplace smoothing
            if count_word_in_0 + count_word_in_1 != 0:
                prob0 *= count_word_in_0 / (count_word_in_0 + count_word_in_1)
                prob1 *= count_word_in_1 / (count_word_in_0 + count_word_in_1)
                '''
        if prob0<=prob1:
            y_pred[i] = 1
    
    return y_pred
    
if __name__ == "__main__":
    t0 = time.time()
    #Obrim el csv i el convertim en diferents datasets i numpy arrays
    X, Y = open_fitxer()

    
    #CREAREM 2 diccionaris, per paraules positives iparaules negatives
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
    dict_sentiment_0, dict_sentiment_1 = create_dicts(x_train, y_train, -1) #if n = -1 no hi ha limitacio de diccionari
    
    
    #Fem la prediccio
    y_pred = prediccio(x_test, dict_sentiment_0, dict_sentiment_1)
    auc = accuracy_score(y_test, y_pred)
    print('Accuracy predita:', auc)
    
    t1 = time.time()
    print('Temps total de execucio:', t1-t0)