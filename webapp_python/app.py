from flask import Flask, render_template, request, redirect, Markup
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
import seaborn as sns

# Read in dataset from csv
df=pd.read_csv('https://docs.google.com/spreadsheets/d/1_ODNIn5n1k9RSVr_7gV2On4idZ6v0B2XPeMhCD18kio/pub?gid=1188863554&single=true&output=csv')

# placeholder for selection array being used later
select_array = np.array([])

# creating functions for use in runnin models and predictions
def run_model(target):
    dfwork = df[[
    "decisionDirection",
    "decisionType",
    "threeJudgeFdc",
    "certReason",
    "lcDispositionDirection",
    "partyWinning",
    "majVotes",
    "chief_term",
    "minVotes",
    "caseDisposition"
    ]]

    y=df[[target]]
    X = dfwork.drop(columns=y)
    dfwork= dfwork.sort_index(axis=1)
    columns_to_move = ['majVotes', 'minVotes']
    remaining_columns = [col for col in dfwork.columns if col not in columns_to_move]
    dfwork = dfwork[remaining_columns + columns_to_move]

    # Impute missing values with the mean
    imputery = SimpleImputer(strategy='constant', fill_value=0)
    y_imputed = imputery.fit_transform(y).reshape(-1)

    # Impute missing values with the mean
    imputerx = SimpleImputer(strategy='most_frequent')
    X_imputed = pd.DataFrame(imputerx.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, random_state=42)
    return X_train, X_test, y_train, y_test, target

#creating function to models using data returned by run_model function
def models(X_train, X_test, y_train, y_test, target):
    knn(X_train, X_test, y_train, y_test, target)
    lr(X_train, X_test, y_train, y_test, target)
    rf(X_train, X_test, y_train, y_test, target)
    dt(X_train, X_test, y_train, y_test, target)
   
# Function for KNN model
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
    update_score(target,model_name, function_name, train_score, test_score, acc_score)
    return model

# Function for Logistic Regression model
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

    predictions = model.predict(xt)
    # Calculate the accuracy score
    acc_score = accuracy_score(yt, predictions)

    update_score(target,model_name, function_name, train_score, test_score, acc_score)
    return model

# Function for Random Forest Classifier model
def rf(X, xt,y, yt, target):
    # Create the random forest classifier model with a random_state of 1
    rt_model = RandomForestClassifier(n_estimators=128, random_state=42)
    model_name = 'Random Forest'
    function_name = 'rf'
    # Fit the model to the training data
    rt_model.fit(X, y)
    
    train_score = rt_model.score(X, y)
    test_score = rt_model.score(xt, yt)

    # Validate the model by checking the model accuracy with model.score

    predictions = rt_model.predict(xt)
    # Calculate the accuracy score
    acc_score = accuracy_score(yt, predictions)
    update_score(target,model_name, function_name, train_score, test_score, acc_score)
    return rt_model

# Function for Decision Tree Classifier model
def dt(X, xt,y, yt, target):
    from sklearn import tree
    # Create the decision tree classifier model with a random_state of 1
    model = tree.DecisionTreeClassifier()
    model_name = 'Decision Tree'
    function_name = 'dt'
    # Fit the model to the training data
    model.fit(X, y)

    train_score = model.score(X, y)
    test_score = model.score(xt, yt)

    # Validate the model by checking the model accuracy with model.score

    predictions = model.predict(xt)
    # Calculate the accuracy score
    acc_score = accuracy_score(yt, predictions)

    update_score(target,model_name, function_name, train_score, test_score, acc_score)
    return model

# Function for appending accuracy scores into dataframe
def update_score(target, model, fn, trs, tss, acc):
    
    model_score.append({'target_column': target, 'model_name': model, 'function_name' : fn, 'training_score': trs, 'testing_score': tss, 'accuracy_score': acc})

# Function for predicting outcome from trained model
def predict_outcome(trained_model, selection1, target):
    selection_df = pd.DataFrame(selection1)
    selection_df = selection_df.T
    selection_df.loc[0]
    X_new1 = np.array([selection_df.iloc[0].to_numpy().astype(int)])

    # Get prediction
    prediction = trained_model.predict(X_new1)

    feature_name = prompts[(prompts['features'] == target) & (prompts['selection'] == round(prediction[0]))]['feature_name'].iloc[0]
    my_prediction = prompts[(prompts['features'] == target) & (prompts['selection'] == round(prediction[0]))]['value'].iloc[0]

    prediction_text = f"Based on my training and analysis of the inputs, I predict that outcome of {feature_name} will be : <b>{my_prediction}</b>"

    return prediction_text

# Initializing the prediction variable
prediction = None

# Initial creation of dataframe for analysis
dfwork = df[[
"decisionDirection",
"decisionType",
"threeJudgeFdc",
"certReason",
"lcDispositionDirection",
"partyWinning",
"majVotes",
"minVotes",
"caseDisposition"
]]

chiefs = df['chief'].unique()
x = 0
for chief in chiefs:
    df.loc[df['chief'] == chief, 'chief_term'] = x
    x = x + 1

df['dateDecision'] = pd.to_datetime(df['dateDecision'], errors='coerce')
df['yearDecision'] = df['dateDecision'].dt.year


targets = ('partyWinning', 'decisionDirection', 'decisionType', 'caseDisposition')
model_score = []
for target in targets:

    X_train, X_test, y_train, y_test, target = run_model(target)
    models(X_train, X_test, y_train, y_test, target)

model_scores = pd.DataFrame(model_score)

trained_model = None

# Creating the variables holding information for each feature being displayed in dropdowns.

# Dropdown for Decision Type
dt_dropdown = """<td><label for='dt_select'>Please select a Decision Type:</label></td>
<td><select name='dt_select' placeholder='Decision Type'>
<option disabled selected hidden>Select Decision Type</option>
<option value='1'>opinion of the court (orally argued)</option>
<option value='2'>per curiam (no oral argument)</option>
<option value='4'>decrees</option>
<option value='5'>equally divided vote</option>
<option value='6'>per curiam (orally argued)</option>
<option value='7'>judgment of the Court (orally argued)</option>
<option value='8'>seriatim</option>
</select></td>"""

# Dropdown for Decision Direction
dd_dropdown = """<td><label for='dd_select'>Please select a Decision Direction:</label></td>
<td><select name='dd_select' placeholder='Decision Direction'>
<option disabled selected hidden>Select Decision Direction</option>
<option value='1'>conservative</option>
<option value='2'>liberal</option>
<option value='3'>unspecifiable</option>
</select></td>"""

# Dropdown for Case Disposition
cd_dropdown = """<td><label for='cd_select'>Please select a Case Disposition:</label></td>
<td><select name='cd_select' placeholder='Case Disposition'>
<option disabled selected hidden>Select Case Disposition</option>
<option value='1'>stay, petition, or motion granted</option>
<option value='2'>affirmed (includes modified)</option>
<option value='3'>reversed</option>
<option value='4'>reversed and remanded</option>
<option value='5'>vacated and remanded</option>
<option value='6'>affirmed and reversed (or vacated) in part</option>
<option value='7'>affirmed and reversed (or vacated) in part and remanded</option>
<option value='8'>vacated</option>
<option value='9'>petition denied or appeal dismissed</option>
<option value='10'>certification to or from a lower court</option>
<option value='11'>no disposition</option>
</select></td>"""

# Dropdown for Winning Party
pw_dropdown = """<td><label for='pw_select'>Please select a Winning Party:</label></td>
<td><select name='pw_select' placeholder='Winning Party'>
<option disabled selected hidden>Select Winning Party</option>
<option value='0'>no favorable disposition for petitioning party apparent</option>
<option value='1'>petitioning party received a favorable disposition</option>
<option value='2'>favorable disposition for petitioning party unclear</option>
</select></td>"""

# Dropdown for Three Judge Fiduciary
tjf_dropdown = """<td><label for='tjf_select'>Please select a Three Judge Fiduciary:</label></td>
<td><select name='tjf_select' placeholder='Three Judge Fiduciary'>
<option disabled selected hidden>Select Three Judge Fiduciary</option>
<option value='0'>no mention that a 3-judge ct heard case</option>
<option value='1'>3-judge district ct heard case</option>
</select></td>"""

# Dropdown for Cert Reason
cr_dropdown = """<td><label for='cr_select'>Please select a Reason for granting cert:</label></td>
<td><select name='cr_select' placeholder='Reason for granting cert'>
<option disabled selected hidden>Select Reason for granting cert</option>
<option value='1'>case did not arise on cert or cert not granted</option>
<option value='2'>federal court conflict</option>
<option value='3'>federal court conflict and to resolve important or significant question</option>
<option value='4'>putative conflict</option>
<option value='5'>conflict between federal court and state court</option>
<option value='6'>state court conflict</option>
<option value='7'>federal court confusion or uncertainty</option>
<option value='8'>state court confusion or uncertainty</option>
<option value='9'>federal court and state court confusion or uncertainty</option>
<option value='10'>to resolve important or significant question</option>
<option value='11'>to resolve question presented</option>
<option value='12'>no reason given</option>
<option value='13'>other reason</option>
</select></td>"""

# Dropdown for Case Disposition
ct_dropdown = """<td><label for='ct_select'>Please select a Chief Justice:</label></td>
<td><select name='ct_select' placeholder='Chief Justice'>
<option disabled selected hidden>Select Chief Justice</option>
<option value='0'>Vinson</option>
<option value='1'>Warren</option>
<option value='2'>Burger</option>
<option value='3'>Rehnquist</option>
<option value='4'>Roberts</option>
</select></td>"""

# Dropdown for Lower Course Disposition Direction
ldd_dropdown = """<td><label for='ldd_select'>Please select a Lower Course Disposition Direction	:</label></td>
<td><select name='ldd_select' placeholder='Lower Course Disposition Direction'>
<option disabled selected hidden>Select Lower Course Disposition Direction</option>
<option value='1'>conservative</option>
<option value='2'>liberal</option>
<option value='3'>unspecifiable</option>
</select></td>"""

# Creating variables holding the text boxes for entering Majority and Minority vote information
majvotes = "<tr><td><label for='majVotes'>Enter number of majority votes:</label></td><td><input type=number name='majVotes'></td></tr>"
minvotes = "<tr><td><label for='minVotes'>Enter number of minority votes:</label></td><td><input type=number name='minVotes'></td></tr>"


# Reading in the prompts from csv file
prompts=pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTcUMWXktcB48qH6B77aGPX1XfrkbBc2_RxzJbmCdtDkUZa2m1orezx1pcrGdvfGytfNQL8L-_58SyP/pub?gid=0&single=true&output=csv")
prompts = prompts.sort_values(by = ['features', 'selection'])
prompts.reset_index(drop=True, inplace=True)
# starting_with_0 = ['partyWinning','threeJudgeFdc']
# prompts_0 = prompts[prompts['features'].isin(starting_with_0)]
# prompts_1 = prompts[~prompts['features'].isin(starting_with_0)]


# Initializing Flask application
app = Flask(__name__)

# Defining the index page
@app.route("/")
def app_input_page():
    
    # Displaying model scores dataframe on screen
    df_html = model_scores.to_html(index=False)
    return render_template("index.html", df_html=df_html)

# Defining the  page
@app.route("/featureselection",  methods=['POST'])
def second_page():

    # Initializing the HTML code with data to be displayed on screen
    html_data = "<tr style='display:none'><td>Target selected:</td><td><input type=text name=""tgtselected"" value='"+request.form['tgtselect']+"'/></td></tr><tr style='display:none'><td>Model selected:</td><td><input type=text name=""mdlselected"" value='"+request.form['mdlselect']+"'/></td></tr>"

    # pulling the POST value from the form
    tgtselected = request.form['tgtselect']

    # Initializing a blank variable for holding the information for relevant dropdowns
    dropdowndata=""

    # Populating the dropdowndata variable with relevant dropdowns
    if tgtselected == "decisionType":
        dropdowndata="<tr>" + cd_dropdown + "</tr><tr>" + cr_dropdown + "</tr><tr>" + ct_dropdown + "</tr><tr>" + dd_dropdown+"</tr><tr>" + ldd_dropdown + "</tr><tr>" + pw_dropdown + "</tr><tr>" + tjf_dropdown
    
    elif tgtselected == "decisionDirection":
        dropdowndata="<tr>" + cd_dropdown + "</tr><tr>" + cr_dropdown + "</tr><tr>" + ct_dropdown + "</tr><tr>" + dt_dropdown+"</tr><tr>" + ldd_dropdown + "</tr><tr>" + pw_dropdown + "</tr><tr>" + tjf_dropdown

    elif tgtselected == "partyWinning":
        dropdowndata="<tr>" + cd_dropdown + "</tr><tr>" + cr_dropdown + "</tr><tr>" + ct_dropdown + "</tr><tr>" + dd_dropdown+"</tr><tr>" + dt_dropdown + "</tr><tr>" + ldd_dropdown + "</tr><tr>" + tjf_dropdown

    elif tgtselected == "caseDisposition":
        dropdowndata="<tr>" + cr_dropdown + "</tr><tr>" + ct_dropdown + "</tr><tr>" + dd_dropdown + "</tr><tr>" + dt_dropdown+"</tr><tr>" + ldd_dropdown + "</tr><tr>" + pw_dropdown + "</tr><tr>" + tjf_dropdown
    
    return render_template("featureselection.html", html_data=Markup(html_data + dropdowndata + majvotes +  minvotes))

@app.route("/prediction",  methods=['POST'])
def third_page():
    
    # Reinitializing selection array
    select_array = np.array([])

    # Pulling in the target and models selected from the posted data
    tgtselected = request.form['tgtselected']
    mdlselected = request.form['mdlselected']
    trained_model = None

    # Populating data selected from dropdowns into arrays in correct sequence depending on target selected
    if tgtselected == "decisionType":
        select_array = np.append(select_array,int(request.form["cd_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["cr_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["ct_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["dd_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["ldd_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["pw_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["tjf_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["majVotes"])).astype(int)
        select_array = np.append(select_array,int(request.form["minVotes"])).astype(int)
    
    elif tgtselected == "decisionDirection":
        select_array = np.append(select_array,int(request.form["cd_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["cr_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["ct_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["dt_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["ldd_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["pw_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["tjf_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["majVotes"])).astype(int)
        select_array = np.append(select_array,int(request.form["minVotes"])).astype(int)

    elif tgtselected == "partyWinning":
        select_array = np.append(select_array,int(request.form["cd_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["cr_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["ct_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["dd_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["dt_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["ldd_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["tjf_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["majVotes"])).astype(int)
        select_array = np.append(select_array,int(request.form["minVotes"])).astype(int)

    elif tgtselected == "caseDisposition":
        select_array = np.append(select_array,int(request.form["cr_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["ct_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["dd_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["dt_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["ldd_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["pw_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["tjf_select"])).astype(int)
        select_array = np.append(select_array,int(request.form["majVotes"])).astype(int)
        select_array = np.append(select_array,int(request.form["minVotes"])).astype(int)

    # Reading in X_train, X_test, y_train, y_test and target from run_model function
    Xn, Xt, yn, yt, tgt = run_model(tgtselected)

    # Training selected model with train, test and target data
    if mdlselected == "1":
        trained_model = knn(X_train, X_test, y_train, y_test, tgtselected)
    elif mdlselected == "2":
        trained_model = lr(X_train, X_test, y_train, y_test, tgtselected)
    elif mdlselected == "3":
        trained_model = rf(X_train, X_test, y_train, y_test, tgtselected)
    elif mdlselected == "4":
        trained_model = dt(X_train, X_test, y_train, y_test, tgtselected)
    
    # Reading prediction from the predict_outcome function based on the trained_model, the selection array and target selected
    result = predict_outcome(trained_model, select_array, tgtselected)
    
    # Feeding result to HTML data
    html_data = result

    return render_template("prediction.html", html_data=Markup(html_data))

if __name__ == "__main__":
    app.run(debug=True)
