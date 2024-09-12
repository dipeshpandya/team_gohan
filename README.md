# Team Gohan
# Project 2


## Our Team
Team Gohan is composed of Dhruba Chakrabarti, Dipesh Pandya, and Alexis Wukich.


## The Problem


From the very beginning, our group was committed to using Project 2 to demonstrate how the legal industry could benefit from machine learning. Specifically, our goal was to determine *whether we could use historical case data to train an algorithm to accurately predict the outcome of a Supreme Court case.*


## Our Dataset
For purposes of Project 2, our team elected to use the 
Washington University Law’s [The Supreme Court Database - Version 2023 Release 01](http://supremecourtdatabase.org). A copy of the .csv file can be found [here](https://docs.google.com/spreadsheets/d/1_ODNIn5n1k9RSVr_7gV2On4idZ6v0B2XPeMhCD18kio/pub?gid=1188863554&single=true&output=csv). We specifically focused on the Modern Dataset, which is comprised of 200 pieces of information about each U.S. Supreme Court case decided from 1946-2022. 


## Installation


The following libraries and dependencies are required to run the project successfully:
- !pip install prophet
- import pandas as pd
- from sklearn.impute import SimpleImputer
- from sklearn.model_selection import train_test_split
- from sklearn.metrics import accuracy_score
- from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
- from sklearn.linear_model import LogisticRegression
- from sklearn.svm import SVC 
- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.impute import SimpleImputer
- from IPython.display import clear_output
- import numpy as np
- import time
- import matplotlib.pyplot as plt
- from prophet import Prophet
- import seaborn as sns


**Helpful hint**: After cloning the “Gohan” repository to your computer, add a sub-folder “outputs” to the folder to organize the visualizations and spreadsheet outputs that will be created after running the code.


## Repository Files 


Washington University Law Case Database.csv: A copy of the original [Supreme Court Database]9(https://docs.google.com/spreadsheets/d/1_ODNIn5n1k9RSVr_7gV2On4idZ6v0B2XPeMhCD18kio/pub?gid=1188863554&single=true&output=csv)

function_based_model.ipynb: The completed notebook containing all the code for Project 2

webapp_python folder: we created an extension of our model and predictions which provides a GUI interface prototype for future use on the web

[Powerpoint Presentation](https://docs.google.com/presentation/d/1OPi3H7meqaaGZ5fLT1r_qsfQ5a681IWky1xjDdVXiGI/edit#slide=id.p1)

## Methodology


### Phase 1: Initial Clean-up and Exploration of our Dataset


Our dataset was in very good condition from the outset since the vast majority of variables had already been numerically coded. After reading in our dataset we created our original dataframe “df” which was used to conduct our initial exploration, and come up with a plan for our project. To make this data more useful we completed the following steps:


1. Converted “dateDecision” from an object to a datetime object
2. Used this datetime object to create a new “yearDecision” column
3. Created a new column, “chief_term” that combined “term” and “chief”


We undertook an initial review of the [Documentation](http://scdb.wustl.edu/documentation.php?s=1) and identified one variable, “Disposition of the Case (caseDisposition)” that we thought would be interesting to explore as our “outcome” variable. According to the [Documentation](http://scdb.wustl.edu/documentation.php?s=1), caseDisposition is defined as “The treatment the Supreme Court accorded the court whose decision it reviewed” and was assigned a numerically coded value ranging from 1 to 11: 
> **Values:**
> 1  stay, petition, or motion granted  
> 2  affirmed (includes modified)  
> 3  reversed  
> 4  reversed and remanded  
> 5  vacated and remanded  
> 6  affirmed and reversed (or vacated) in part  
> 7  affirmed and reversed (or vacated) in part and remanded  
> 8  vacated  
> 9  petition denied or appeal dismissed  
> 10  certification to or from a lower court  
> 11  no disposition


#### Exploration Question #1: What is the most common caseDisposition? 
To determine the most common disposition, we elected to visualize the caseDisposition using a bubble plot. To prepare the data for visualization we:
- Filtered the “df” DataFrame by caseDisposition and created a count column for a total count of each class (values 1-11)
- Create components of bubble plot, normalize bubble size and formatted the axes for ease of viewing 
The resulting bubble plot can be found in ![Fig. 1: Case Disposition Bubble Plot]
(https://drive.google.com/file/d/1BAg7sjSXhiRUsz0u3KFP9w8omyQmxrPC/view?usp=drive_link)


>Answer: Using this bubble plot we determined that “2”, or “affirmed”, followed by “4” and “3”, “reversed and remanded” and “remanded”, respectively, were the most common disposition from the Supreme Court from the 1946 term to the 2022 term


#### Exploration Question #2: Have case dispositions changed over time?
In order to determine how case dispositions had changed over time, we:
- Filtered the “df” DataFrame by Year (to reduce the 13,000+ datapoints) and caseDisposition and created a mode column for most frequent classification of caseDisposition
- Applied function to calculate the mode that accounted for multiple modes by selecting the first, and accounting for entries with no mode by filling with NaN
> case_disposition_mode = df.groupby('yearDecision')['caseDisposition'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
- Visualized the data in a bar graph found in ![Fig. 2: Case Disposition by Year]
(https://drive.google.com/file/d/19BJJTx5VtQISxJXfleERHQfUZQc_mA0c/view?usp=drive_link)


>Answer: Using this bar graph we determined that “2” (“affirmed”), followed by “4” ( “reversed and remanded”) were the case dispositions that appeared most frequently from the 1946 term to the 2022 term.


#### Exploration Question #3: How have case dispositions changed over time, based on who is serving as Chief Justice?


Another temporal variable that could be used to explore the data is the “chief_term”. We created this variable to identify patterns or trends in Case Dispositions based on which Chief Justice was presiding over the Supreme Court. To determine the impact of the Chief Justice’s tenure, we:
- Filtered the “df” DataFrame by Term of Chief Justice and Case Disposition
- Created a pivot table using the filtered DataFrame
- Plotted a heatmap using the Seaborn heatmap function which can be visualized in !![Fig. 3:'Heatmap of Case Disposition Types by Chief Justice Term'](https://drive.google.com/file/d/1ebWSdC_o2bH0epWLsLLE-WRyZYAeImNE/view?usp=drive_link)
>Answer: Using the heatmap we could determine which Chief Justice presided over specific case dispositions. For example, Chief Justice Burger prised over the most cases that resulted in a case disposition “2” (“affirmed”). 


####Exploration Question #4: Can we categorize the overall tone (decision direction) of case decisions based on the term of the chief justice?


Even without forecasting, the historical data surrounding caseDisposition supported that most cases would fall into “2” (affirmed) or “4” (reversed or remanded). Since our time was limited and we wanted to explore our data as much as possible, we decided to explore a different variable Though it wasn’t listed as an “outcome variable” in the [Documentation](http://scdb.wustl.edu/documentation.php?s=1), we focused on Decision Direction, or the “ideological direction" of the decision.” In order to determine the ideological tone of decisions, we:
- Filtered the “df” DataFrame by Term of Chief Justice and Decision Direction and add a count column for Decision Direction classes
- Pivoted the DataFrame to get counts for each direction per term
pivot_dd_index = chief_counts_dd_df.pivot_table(index='decisionDirection', columns='chief_term', values='count', fill_value=0)
pivot_dd = pivot_dd_index.sort_index(ascending=False)
- Used the Seaborn heatmap function to plot the heatmap
- Customized and labeled the heatmap when it became clear it was an interesting visualization as shown in ![Fig. 4:''Heatmap of Case Decision Direction during Chief Justice Terms''(]https://drive.google.com/file/d/1ttQzCszi2qg0q7FK0oCG3BJ3A4ZlkZqU/view?usp=drive_link)


>Answer: The heatmap demonstrated that the Court had been rather ideologically balanced during the Chief Justice Vinson years, and had shifted more liberally during Chief Justice Warren’s tenure, before rebounding more conservatively with Chief Justice Burger. The court continued to lean conservatively during Chief Justice Rehnquist’s term, but not to the degree it had prior. In the most recent years, under the tutelage of Chief Justice Roberts, the court has returned to a much more balanced position. 
 
####Exploration Question #5: Can we use this data to predict the ideological direction of future decisions?
To predict the ideological direction of future decisions we:
- We filtered our original “df” DataFrame by “dateDecision” and “decisionDirected” and saved it as anew DataFrame,“dd_by_date_df”.
- To prepare for a time-series forecast we changed the column names of our DataFrame to “ds” and “y” and created variables for a ten-year forecast.
- We created a Prophet model, we fit our DataFrame to the model, made predictions and plotted the forecast and components. The forecast can be found in ![Fig. 5: '''Forecast of Supreme Court Decision Direction'''](https://drive.google.com/file/d/1hj930eBoEoV4NMgyDf1kyHM5518-oOlB/view?usp=drive_link).
>Answer: Our forecasted model predicts will follow a relatively “seasonal” pattern  over the next four years, trending slightly more conservative around 2024, before returning to a more balanced ideological direction around 2028, becoming slightly more conservative again until around 2032-2032, and then again returning to a more balanced ideology. This trend seems to fall in line with the four-year election cycle.


### Phase 2: Preprocessing our Data


After reading in our dataset, we created a working DataFrame known as “dfwork” to make our dataset. To reduce the number of features in our model and improve performance, we:
1. Deleted columns that were objects, i.e. citations to the various case reporters.
2. Deleted columns that contained more than 50% null values.
3. Dropped features with extremely large values (i.e. more than 100 possible values)
4. Removed the remaining highly correlated features
5. Prioritized features that fell into the following categories as identified by the documentation: Outcome, Background, Voting, and Substantive.
6. Identified our multi-class, target variables as “partyWinning,” “decisionDirection”, “decisionType”, and “caseDisposition”. 
7. Created a list of models that we wanted to explore for each target variable, including: Linear Regression, kNN, Decision Tree, Random Forest Classifier


### Phase 3: Streamlining our Process: Training, Testing, and Evaluating Multiple Models and Multiple Variables


With four target variables, four models, and limited time, our group determined that we would need to get creative if we wanted to identify our best performing model. In order to streamline our process, we created a machine learning pipeline to train and evaluate each of the four models, for each of the four target variables. 
#### Preparation: To prepare our data we:
1. Defined a new function, *run_model* and passed in the target variable.
2. We defined our X (features) from the reduced “dfwork” DataFrame, and our y (targets) from the “df” to preserve native formatting.
3. Since the traditional fillna would not work with our models, we imputed NaN values using the “most frequent value” strategy and filling NaN with “0” for our X and y, respectively.
4. We split the data into training and testing data (X_train, X_test, y_train, y_test).


#### Training, Testing, and Evaluating our Model
Rather than training, testing, and evaluating sixteen models, we elected to create a function that would pass our target variables into each of the models, and return the model scores into a DataFrame. Using this output, we could identify which target variable performed best for each model. To complete this step we:
1. Defined a function for each model that fit the model, trained, tested, evaluated/validated, and scored the model.
2. Saved the outputs (scores) from these models in a new model_scores DataFrame, which is also represented here in Table 1.


Table 1: Model Scores


| target_column       | model_name           | function_name | training_score | testing_score |
|---------------------|----------------------|---------------|----------------|---------------|
| partyWinning        | Random Forest        | rf            | 0.9862380907    | 0.9679561201  |
| partyWinning        | Decision Tree        | dt            | 0.9862380907    | 0.9676674365  |
| partyWinning        | KNN                  | knn           | 0.9617938601    | 0.9532332564  |
| decisionDirection   | Random Forest        | rf            | 0.941487826     | 0.9234988453  |
| decisionDirection   | Decision Tree        | dt            | 0.941487826     | 0.9180138568  |
| decisionDirection   | KNN                  | knn           | 0.9028967376    | 0.8865473441  |
| decisionType        | Random Forest        | rf            | 0.8786449812    | 0.8527713626  |
| decisionType        | Decision Tree        | dt            | 0.8786449812    | 0.8493071594  |
| decisionType        | Logistic Regression  | lr            | 0.8322586854    | 0.8325635104  |
| decisionType        | KNN                  | knn           | 0.8420748725    | 0.8293879908  |
| partyWinning        | Logistic Regression  | lr            | 0.7786546049    | 0.7855080831  |
| caseDisposition     | Decision Tree        | dt            | 0.6831873737    | 0.6342378753  |
| caseDisposition     | Random Forest        | rf            | 0.6831873737    | 0.6333718245  |
| decisionDirection   | Logistic Regression  | lr            | 0.6298720046    | 0.6252886836  |
| caseDisposition     | KNN                  | knn           | 0.6327591185    | 0.6050808314  |
| caseDisposition     | Logistic Regression  | lr            | 0.5715523049    | 0.5868937644  |






> After running the function, the output reads “Best target column / model is: partyWinning /Random Forrest Tree.”  Overall, the Random Forest Model, Decision Tree, and kNN models performed well, particularly when using the partyWinning and decisionDirection target variable. Though we initially hypothesized that caseDisposition would be the most informative target variable, it was not effective for purposes of training and testing our models. Logistic Regression did not perform well overall, and could be attributed to the use of multi-class variables.


### Phase 4: Using our Model(s) to Make Predictions
Since we detemrmined that accurate training models was possible, we wanted to stretch ourselves and attempt to make predictions based on our trained data. To do this, we created a new predict_outcome function, that takes user inputs based on the features (X)  and made predictions based on the trained model. We also were able to use this code to assist with developing a GUI interface prototype for future implementation on the web.


## Noteworthy Pain Points, Learning Experiences, and Opportunities for Further Research and Exploration
Starting with a clean dataset is a huge advantage, but it is still incredibly time consuming to explore that data and determine what is necessary and relevant
Multiclass data can be very informative but difficult to manage the increased number of classes
There is no end to how we could explore and use this data and this project could go on forever. For example: training and testing a model on the Cert target variable would be incredibly interesting, so that we could make predictions on the reason that Cert was granted by the court, and potentially expand these predictions to identify which cases would be likely to be granted Cert based on background data. Or we could work to use our trained model in conjunction with our time-series data to make predictions about pending or outcomes for future cases.


## Resources Consulted and Credits
The following credit must be given for assistance with our project:


Harold J. Spaeth, Lee Epstein, Andrew D. Martin, Jeffrey A. Segal, Theodore J. Ruger, and Sara C. Benesh. 2023 Supreme Court Database, Version 2023 Release 01. URL: [http://supremecourtdatabase.org](http://supremecourtdatabase.org).
 
(Supreme Court Database Online Code Book)[http://scdb.wustl.edu/documentation.php?s=1]


Thank you to the following individuals for helping us when the code was tricky, or we wanted to push ourselves!


Zohaib Khawaja, our favorite, former T.A.
Julia Kim
Anthony Inthavong


