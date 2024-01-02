from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os

from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
from nltk.stem import PorterStemmer
import pickle
from imblearn.over_sampling import SMOTE

from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model
from keras.models import model_from_json
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


main = tkinter.Tk()
main.title("Deep Convolutional Forest: A Dynamic Deep Ensemble Approach for Spam Detection in Text")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test
global cnn, filename, dataset, wordembed_vectorizer, cnn_model, dcf
# global X, Y
accuracy = []
precision = []
recall = []
fscore = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

textdata = []
labels = []

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded")

    dataset = pd.read_csv(filename, encoding='iso-8859-1')
    text.insert(END, str(dataset.head()))
    text.update_idletasks()
    label = dataset.groupby('v1').size()
    label.plot(kind="bar")
    plt.show()

def preprocessDataset():
    global X, Y, dataset, textdata, labels, wordembed_vectorizer
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    textdata.clear()
    labels.clear()
    if os.path.exists("model/X.npy"):
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
        with open('model/wordembed.txt', 'rb') as file:
            wordembed_vectorizer = pickle.load(file)
        file.close()
    else:
        for i in range(len(dataset)):
            msg = dataset.get_value(i, 'v2')
            label = dataset.get_value(i, 'v1')
            msg = str(msg)
            msg = msg.strip().lower()
            if label == 'ham':
                labels.append(0)
            elif label == 'spam':
                labels.append(1)
            clean = cleanPost(msg)
            textdata.append(clean)
        textdata = np.asarray(textdata)
        labels = np.asarray(labels)
        wordembed_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=1200)
        wordembed = wordembed_vectorizer.fit_transform(textdata).toarray()
        np.save("model/X", wordembed)
        np.save("model/Y", labels)
        with open('model/wordembed.txt', 'wb') as file:
            pickle.dump(wordembed_vectorizer, file)
        file.close()
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
    text.insert(END,"Preprocess & Word Embedding Vector\n\n")    
    text.insert(END,str(X)+"\n\n")
    classes, count = np.unique(Y, return_counts=True)
    text.insert(END,"Total HAM Messages found in dataset: "+str(count[0])+"\n")
    text.insert(END,"Total SPAM Messages found in dataset: "+str(count[1])+"\n")

def smoteoverSampling():
    global X, Y, dataset, textdata, labels, wordembed_vectorizer
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    smote = SMOTE()
    X, Y = smote.fit_resample(X, Y)
    text.insert(END,"Total classes found in dataset after SMOTE over-sampling\n\n")
    classes, count = np.unique(Y, return_counts=True)
    text.insert(END,"Total HAM Messages found in dataset after SMOTE : "+str(count[0])+"\n")
    text.insert(END,"Total SPAM Messages found in dataset after SMOTE: "+str(count[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train & Test Split Details. 80% dataset used for training and 20% for testing\n\n")
    text.insert(END,"80% records for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% records for training : "+str(X_test.shape[0])+"\n")


def calculateMetrics(algorithm, predict, target):
    acc = accuracy_score(target,predict)*100
    p = precision_score(target,predict,average='macro') * 100
    r = recall_score(target,predict,average='macro') * 100
    f = f1_score(target,predict,average='macro') * 100
    text.insert(END,algorithm+" Precision  : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall     : "+str(r)+"\n")
    text.insert(END,algorithm+" F1-Score   : "+str(f)+"\n")
    text.insert(END,algorithm+" Accuracy   : "+str(acc)+"\n\n")
    text.update_idletasks()
    precision.append(p)
    accuracy.append(acc)
    recall.append(r)
    fscore.append(f)
    LABELS = ['HAM', 'SPAM']
    conf_matrix = confusion_matrix(target, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    


def runExistingAlgorithms():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    svm_cls = svm.SVC()
    svm_cls.fit(X_train,y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)
       

def runCNN():
    global X_train, X_test, y_train, y_test, cnn
    X_trains = np.reshape(X_train, (X_train.shape[0], 20, 20, 3))
    X_tests = np.reshape(X_test, (X_test.shape[0], 20, 20, 3))
    y_trains = to_categorical(y_train)
    y_tests = to_categorical(y_train)

    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn = model_from_json(loaded_model_json)
        json_file.close()    
        cnn.load_weights("model/model_weights.h5")
        cnn._make_predict_function()       
    else:
        cnn = Sequential()
        cnn.add(Convolution2D(32, 3, 3, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))
        cnn.add(Convolution2D(32, 3, 3, activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))
        cnn.add(Flatten())
        cnn.add(Dense(output_dim = 256, activation = 'relu'))
        cnn.add(Dense(output_dim = y_trains.shape[1], activation = 'softmax'))
        cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = cnn.fit(X_trains, y_trains, batch_size=16, epochs=10, shuffle=True, verbose=2, validation_data = (X_tests, y_tests))
        cnn.save_weights('model/model_weights.h5')            
        model_json = cnn.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
    predict = cnn.predict(X_tests)
    predict = np.argmax(predict, axis=1)
    calculateMetrics("Convolution Features Extraction", predict, y_test)

def runDCF():
    global cnn
    global X_train, X_test, y_train, y_test, X, Y, cnn_model, dcf
    X1 = np.reshape(X, (X.shape[0], 20, 20, 3))
    cnn_model = Model(cnn.inputs, cnn.layers[-2].output)#creating cnn model
    deep_features = cnn_model.predict(X1)  #extracting cnn features from test data
    print(Y)
    print(deep_features.shape)
    X_train, X_test, y_train, y_test = train_test_split(deep_features, Y, test_size=0.2)
    dcf = BaggingClassifier(base_estimator=RandomForestClassifier())
    dcf.fit(X_train, y_train)
    predict = dcf.predict(X_test)
    calculateMetrics("DCF", predict, y_test)

def graph():
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                       ['Convolution Features Extraction','Precision',precision[1]],['Convolution Features Extraction','Recall',recall[1]],['Convolution Features Extraction','F1 Score',fscore[1]],['Convolution Features Extraction','Accuracy',accuracy[1]],
                       ['DCF','Precision',precision[2]],['DCF','Recall',recall[2]],['DCF','F1 Score',fscore[2]],['DCF','Accuracy',accuracy[2]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def predict():
    text.delete('1.0', END)
    global cnn_model, wordembed_vectorizer, dcf
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=str(filename)+" Dataset Loaded")
    testData = pd.read_csv(filename, encoding='iso-8859-1')
    print(testData)
    for i in range(len(testData)):
        msg = testData.get_value(i, 'Message')
        review = msg.lower()
        review = review.strip().lower()
        review = cleanPost(review)
        testReview = wordembed_vectorizer.transform([review]).toarray()
        testReview = np.reshape(testReview, (testReview.shape[0], 20, 20, 3))
        print(testReview.shape)
        predict = cnn_model.predict(testReview)
        print(predict.shape)
        predict = dcf.predict(predict)
        predict = predict[0]
        if predict == 0:
            text.insert(END,"Message = "+str(msg)+" PREDICTED AS =========> HAM\n\n")
        if predict == 1:
            text.insert(END,"Message = "+str(msg)+" PREDICTED AS =========> SPAM\n\n")
        

def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='Deep Convolutional Forest: A Dynamic Deep Ensemble Approach for Spam Detection in Text')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload SMS Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess & Word Embedding Vector", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

smoteButton = Button(main, text="Run SMOTE Over-Sampling Dataset", command=smoteoverSampling)
smoteButton.place(x=50,y=200)
smoteButton.config(font=font1)

existingButton = Button(main, text="Run Existing Algorithm", command=runExistingAlgorithms)
existingButton.place(x=50,y=250)
existingButton.config(font=font1)

cnnButton = Button(main, text="Convolution Based Features Extraction", command=runCNN)
cnnButton.place(x=50,y=300)
cnnButton.config(font=font1)

dcfButton = Button(main, text="Run Propose DCF Algorithm", command=runDCF)
dcfButton.place(x=50,y=350)
dcfButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=400)
graphButton.config(font=font1)

predictButton = Button(main, text="Spam Prediction from Test Data", command=predict)
predictButton.place(x=50,y=450)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=500)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
