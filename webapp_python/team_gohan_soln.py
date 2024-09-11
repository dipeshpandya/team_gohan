
def initiate():
# Import required dependencies
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    #from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC 
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from IPython.display import clear_output
    import numpy as np
    import time
    import matplotlib.pyplot as plt
    from prophet import Prophet
    import seaborn as sns

    df=pd.read_csv('https://docs.google.com/spreadsheets/d/1_ODNIn5n1k9RSVr_7gV2On4idZ6v0B2XPeMhCD18kio/pub?gid=1188863554&single=true&output=csv')
        #df.head()

    chiefs = df['chief'].unique()
    x = 0
    for chief in chiefs:
        #print(chief)
        df.loc[df['chief'] == chief, 'chief_term'] = x
        #dfcoyp['chief_term'] = x
        x = x + 1

    #change decision date to DT
    df['dateDecision'] = pd.to_datetime(df['dateDecision'], errors='coerce')

    df['yearDecision'] = df['dateDecision'].dt.year

    dfwork = df[[
    #"issueArea",
    "decisionDirection",
    "decisionType",
    "threeJudgeFdc",
    "certReason",
    #"lcDisposition",
    "lcDispositionDirection",
    "partyWinning",
    "majVotes",
    #"chief_term",
    "minVotes",
    "caseDisposition"
    ]]

    targets = ('partyWinning', 'decisionDirection', 'decisionType', 'caseDisposition')
    model_score = []
    
    for target in targets:
        X_train, X_test, y_train, y_test, target = run_model(target)
        models(X_train, X_test, y_train, y_test, target)

    model_scores = pd.DataFrame(model_score)
    return model_scores


def run_model(target):
    dfwork = df[[
    #"issueArea",
    "decisionDirection",
    "decisionType",
    "threeJudgeFdc",
    "certReason",
    #"lcDisposition",
    "lcDispositionDirection",
    "partyWinning",
    "majVotes",
    #"chief_term",
    "minVotes",
    "caseDisposition"
    ]]

    y=df[[target]]
    X = dfwork.drop(columns=y)
    dfwork= dfwork.sort_index(axis=1)
    columns_to_move = ['majVotes', 'minVotes']
    remaining_columns = [col for col in dfwork.columns if col not in columns_to_move]
    dfwork = dfwork[remaining_columns + columns_to_move]

    #reading csv file from link

    #from sklearn.impute import SimpleImputer

    # Impute missing values with the mean
    imputery = SimpleImputer(strategy='constant', fill_value=0)
    #caseDisposition = df[['caseDisposition']]
    y_imputed = imputery.fit_transform(y).reshape(-1)
    #y_imputed = y_imputed.reshape(-1)
    #print(y_imputed)

    #from sklearn.impute import SimpleImputer

    # Impute missing values with the mean
    imputerx = SimpleImputer(strategy='most_frequent')
    X_imputed = pd.DataFrame(imputerx.fit_transform(X), columns=X.columns)
    #print(X_imputed)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, random_state=42)

    return X_train, X_test, y_train, y_test, target

def models(X_train, X_test, y_train, y_test, target):
    knn(X_train, X_test, y_train, y_test, target)
    lr(X_train, X_test, y_train, y_test, target)
    rf(X_train, X_test, y_train, y_test, target)
    dt(X_train, X_test, y_train, y_test, target)
    #return X_train, X_test, y_train, y_test, target
    #return rt_model

#    return X_train, X_test, y_train, y_test, target
    
    # Create the KNN model with 9 neighbors


def knn(X, xt,y, yt, target):
    model = KNeighborsClassifier(n_neighbors=10)
    model_name = 'KNN'
    function_name = 'knn'
    # Fit the model to the training data
    model.fit(X, y)

    #using score function to find training and testing score
    train_score = model.score(X, y)
    test_score = model.score(xt, yt)
    predictions = model.predict(xt)
    acc_score = accuracy_score(yt, predictions)
    #display(f"{model_name} Train/Test Score for {target} column : {train_score:.3f}/{test_score:.3f}")
    update_score(target,model_name, function_name, train_score, test_score, acc_score)
    #return knn_train_score, knn_test_score, target
    return model

def lr(X, xt,y, yt, target):
    # Create the logistic regression classifier model with a random_state of 1
    model = LogisticRegression(max_iter=2000)
    model_name = 'Logistic Regression'
    function_name = 'lr'
    # Fit the model to the training data
    model.fit(X, y)

    train_score = model.score(X, y)
    test_score = model.score(xt, yt)

    # Validate the model by checking the model accuracy with model.score

    #display(f"{model} Train/Test Score for {target} column : {lr_train_score:.3f}/{lr_test_score:.3f}")
    predictions = model.predict(xt)
    # Calculate the accuracy score
    acc_score = accuracy_score(yt, predictions)
    #print(f"Accuracy Score : {acc_score}")

    update_score(target,model_name, function_name, train_score, test_score, acc_score)
    return model

def rf(X, xt,y, yt, target):
    # Create the logistic regression classifier model with a random_state of 1
    rt_model = RandomForestClassifier(n_estimators=128, random_state=42)
    model_name = 'Random Forest'
    function_name = 'rf'
    # Fit the model to the training data
    rt_model.fit(X, y)
    
    train_score = rt_model.score(X, y)
    test_score = rt_model.score(xt, yt)

    # Validate the model by checking the model accuracy with model.score

    #display(f"{model} Train/Test Score for {target} column : {lr_train_score:.3f}/{lr_test_score:.3f}")
    predictions = rt_model.predict(xt)
    # Calculate the accuracy score
    acc_score = accuracy_score(yt, predictions)
    #print(f"Accuracy Score : {acc_score}")
    #trained_model = model
    #print({trained_model})
    update_score(target,model_name, function_name, train_score, test_score, acc_score)
    return rt_model

def dt(X, xt,y, yt, target):
    from sklearn import tree
    # Create the logistic regression classifier model with a random_state of 1
    model = tree.DecisionTreeClassifier()
    model_name = 'Decision Tree'
    function_name = 'dt'
    # Fit the model to the training data
    model.fit(X, y)

    train_score = model.score(X, y)
    test_score = model.score(xt, yt)

    # Validate the model by checking the model accuracy with model.score

    #display(f"{model} Train/Test Score for {target} column : {lr_train_score:.3f}/{lr_test_score:.3f}")
    predictions = model.predict(xt)
    # Calculate the accuracy score
    acc_score = accuracy_score(yt, predictions)
    #print(f"Accuracy Score : {acc_score}")

    update_score(target,model_name, function_name, train_score, test_score, acc_score)
    return model

def update_score(target, model, fn, trs, tss, acc):
    
    model_score.append({'target_column': target, 'model_name': model, 'function_name' : fn, 'training_score': trs, 'testing_score': tss, 'accuracy_score': acc})
    #display(model_scores.head())
    #return model_scores    

def predict_outcome(trained_model):
    #    #prompts=pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTcUMWXktcB48qH6B77aGPX1XfrkbBc2_RxzJbmCdtDkUZa2m1orezx1pcrGdvfGytfNQL8L-_58SyP/pub?gid=0&single=true&output=csv")
#    prompts = prompts.sort_values(by = ['features', 'selection'])
#    prompts.reset_index(drop=True, inplace=True)
#    prompts
    #selection_prompts = prompts
    selection1 = np.array([])
    for p in prompts['features'].unique():
        #time.sleep(1)
        display(f"Please select a feature from the list of {p} below:")
        
        for i, feature in enumerate((prompts[prompts['features'] == p]['value']), start=1):
            display(f"{i}... {feature}")
        selection1 = np.append(selection1,int(input("Enter the number corresponding to your choice: "))).astype(int)
        
        clear_output()
        
        #time.sleep(1)

    mv = np.append(selection1,int(input("Enter number of majority votes: "))).astype(int)
    nv = np.append(selection1,int(input("Enter number of minority votes: "))).astype(int)
    #selection_df = pd.DataFrame(selection1)
    #selection_df = selection_df.T
    #selection_df.loc[0]
    #predict(selection1)
    #return selection1
    selection_df = pd.DataFrame(selection1)
    selection_df = selection_df.T
    selection_df.loc[0]
    #print(selection_df)
    #X_new1 = np.array([X_imputed2.iloc[13853].to_numpy().astype(int)])
    X_new1 = np.array([selection_df.iloc[0].to_numpy().astype(int)])
    #print(X_new1)
    
    # Get prediction
    #print(model)
    #global trained_model
    #prediction = trained_model.predict(X_new1)
    prediction = trained_model.predict(X_new1)
    #print(f"Based on the data, prediction for the {y.columns[0]} is {prediction[0][0].astype(int)}")
    #print(f"Based on the data, prediction for the {y.columns[1]} is {prediction[0][1].astype(int)}")
    #my_prediction = prompts[(prompts['features'] == y_imputed.columns[0]) & (prompts['selection'] == round(prediction[0][0],0))]['value']
    my_prediction = prompts[(prompts['features'] == target) & (prompts['selection'] == round(prediction[0]))]['value'].iloc[0]
    clear_output()
    print(f"I predict that {my_prediction}")
#prediction = model.predict(rt_model, selection1)





