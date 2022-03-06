# 🏦 Bank Churn Prediction 2: Project Overview
* End to end project researching the effects customer attributes have on the churn of a bank customer and predicting those customers that may churn.
* Optimised ANN model using GridsearchCV to reach the best model. 
* Built a stakeholder facing visual deployment of model to predict churn of new customers 
* Deployed Model in Power BI for Business Intelligence analysis 
* An artificial neural network (ANN) is a type of machine learning algorithm that is similar to the brain.

[View prerequisite of this project](https://github.com/MattithyahuData/P2-Bank-churn-prediction)

[View Deployed Model in Power BI](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)

## Table of Contents 
*   [Resources](#resources)<br>
*   [Data Collection](#DataCollection)<br>
*   [Data Pre-processing](#DataPre-processing)<br>
*   [Data Warehousing](#DataWarehousing)<br>
*   [Exploratory data analysis](#EDA)<br>
*   [Data Visualisation & Analytics](#Dataviz)<br>
*   [Business Intelligence](#Busintelli)<br>
*   [Feature Engineering](#FeatEng)<br>
*   [ML/DL Model Building](#ModelBuild)<br>
*   [Model performance](#ModelPerf)<br>
*   [Model Optimisation](#ModelOpt)<br>
*   [Model Evaluation](#ModelEval)<br>
*   [Deployment](#ModelDeploy)<br>
*   [Project Management (Agile/Scrum/Kanban)](#Prjmanage)<br>
*   [Project Evaluation](#PrjEval)<br>
*   [Looking Ahead](#Lookahead)<br>
*   [Questions & Contact me](#Lookahead)<br>


<a name="resources"></a>  

## Resources Used
**Python 3, PostgreSQL, PowerBI** 

[**Anaconda Packages:**](requirements.txt) **pandas numpy pandas_profiling ipywidgets sklearn matplotlib seaborn sqlalchemy tensorflow keras kaggle psycopg2 ipykernel**<br><br>
Powershell command for installing anaconda packages used for this project    
```powershell
pip install pandas numpy pandas_profiling ipywidgets sklearn matplotlib seaborn sqlalchemy tensorflow keras kaggle psycopg2 ipykernel
```

<a name="DataCollection"></a>  

## [Data Collection](Code/P6_Code.ipynb)
Powershell command for data import using kaggle API <br>
```powershell
!kaggle datasets download -d kmalit/bank-customer-churn-prediction -p ..\Data --unzip 
```
[Data source link](https://www.kaggle.com/kmalit/bank-customer-churn-prediction)
[Data](Data/Churn_Modelling.csv)
*  Rows: 10000 | Columns: 14
    *   RowNumber
    *   CustomerId  
    *   Surname 
    *   CreditScore
    *   Geography
    *   Gender
    *   Age
    *   Tenure
    *   Balance
    *   NumOfProducts
    *   HasCrCard
    *   IsActiveMember
    *   EstimatedSalary
    *   Exited                   

<a name="DataPre-processing"></a>  

## [Data Pre-processing](Code/P6_Code.ipynb)
After I had all the data I needed, I needed to check it was ready for exploration and later modelling. I made the following changes and created the following variables:   
*   General NULL and data validity checks  
The data contained no null values and all datatypes lined up with their field description. <br>

```python
# Viewing the data types of the columns
data.dtypes

# Viewing dataset shape
data.shape

# 1st check for null values and datatype check 
data.info()
```

<br>

*   Some programming languages can be case sensitive like python and C++ for example, so using lower case letters for variable names allows for straightforward use of data in different programming languages.<br>

```python
# (SQL standard) Formatting column headers by removing potential capital letters and spaces in column headers 
data.columns = data.columns.str.lower()
data.columns = data.columns.str.replace(' ','_')
```

<a name="DataWarehousing"></a>

## [Data Warehousing](Code/P6_Code.ipynb)
I warehouse all data in a Postgre database for later use and reference.

*   ETL in python to PostgreSQL Database.
*   Formatted column headers to SQL compatibility. 

```python 
# Function to warehouse data in a Postgre database 
def store_data(data,tablename):
    """
    :param data: variable, enter name of dataset you'd like to warehouse
    :param tablename: str, enter name of table for data 
    """

    # SQL table header format
    tablename = tablename.lower()
    tablename = tablename.replace(' ','_')

    # Saving cleaned data as csv
    data.to_csv(f'../Data/{tablename}_clean.csv', index=False)

    # Engine to access postgre
    engine = create_engine('postgresql+psycopg2://postgres:password@localhost:5432/projectsdb')

    # Loads dataframe into PostgreSQL and replaces table if it exists
    data.to_sql(f'{tablename}', engine, if_exists='replace',index=False)

    # Confirmation of ETL 
    return("ETL successful, {num} rows loaded into table: {tb}.".format(num=len(data.iloc[:,0]), tb=tablename))
 
# Calling store_data function to warehouse cleaned data
store_data(data,"P6 ANN Bank Churn")
```

<a name="EDA"></a>  

## [Exploratory data analysis](Code/P6_Code.ipynb) 
I looked at the distributions of the data and the value counts for the various categorical variables that would be fed into the model. Below are a few highlights from the analysis.
*   79.63% of customers have churned - Distribution of features and their effects on churning - Some features have outliers, visualising this allows for greater clarity on the extent. 
<img src="images/Churn_barchart_distrib.png" />
<img src="images/independentfeatures_distrib.png" />
<img src="images/boxplots.png" />

*   I looked at the correlation the features have
<img src="images/churn_correlation.png" />

<a name="Dataviz"></a>  

## [Data Visualisation & Analytics](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
[View Interactive Dashboard](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
*   I created an interactive dashboard to deploy the machine learning model to benefit the business.
*   I visualised various key features and highlighted their overall correlation to a customer’s churn. 

<a name="Busintelli"></a>  

## Business Intelligence
On Page 2 of the interactive dashboard, I have provided the stake holders with the new customer names and the customers that are likely to churn due to their characteristics.

*   These customers can be offered subsidised deals and incentives to keep them on
*   Greater engagement with customers could keep some customers on board 
*   Providing quality customer service can also provide customers with long term value and appreciation for the business
*   The complaints team should pay particular attention to complaints from customers who are predicted to churn.
- 96% of unhappy customers don’t complain and 91% of those will simply leave and never come back?

<a name="FeatEng"></a>  

## [Feature Engineering](Code/P6_Code.ipynb) 
I transformed the categorical variable(s) 'geography' and 'gender' into dummy variables. I also split the data into train and tests sets with a test size of 20%.
*   One Hot encoding to encode values
*   Using StandardScaler to scale  

```python
# One Hot encoding for remaining categorical field 
data = pd.get_dummies(data, drop_first = False)
data.head()

# Using train test split to split train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.20, random_state=23, shuffle=True, stratify=y)

# Viewing shape of train / test data
print(X_train.shape)
print(X_test.shape)

# Feature Scaling
# In ANN feature scaling is very important so that all inputs are at a comparable range and only the weights assigned to them are, 
# in fact, the only factor which makes a difference on the predicted value.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

<a name="ModelBuild"></a> 

## [ML/DL Model Building](Code/P6_Code.ipynb)

I used an Artificial Neural Network to attempt to improve the predictive performance for the churn project. ANNs have some key advantages that make them most suitable for certain problems and situations:
*   ANNs have the ability to learn and model non-linear and complex relationships, which is really important because in real-life, many of the relationships between inputs and outputs are non-linear as well as complex
*   Neural Networks have the ability to learn by themselves and produce the output that is not limited to the input provided to them.

```python
# Initialising the ANN - Defining as a sequence of layers or a Graph
classifier = Sequential()

# units - number of nodes to add to the hidden layer.
# Tip: units should be the average of nodes in the input layer (11 nodes) and the number of nodes in the output layer (1 node). For this case is 11+1/2 = 6
# kernel_initializer - randomly initialize the weight with small numbers close to zero, according to uniform distribution.
# activation - Activation function.
# input_dim - number of nodes in the input layer, that our hidden layer should be expecting
# Distribute features of the first observation, from your dataset, per each node in the input layer. Thus, eleven independent variables will be added to our input layer.

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11 ))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN # Cost Function : Measure the generated error by comparing the predicted value with the true value.
classifier.compile(optimizer = 'adam',loss= 'binary_crossentropy',metrics=['accuracy'])

# Fitting the ANN to the Training set  # batch_size : number of observations after which we update the weights  # epochs: How many times you train your model
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
```

<!-- <img src="images/Crossvalidation.png" /> -->

<a name="ModelPerf"></a> 

## [Model performance](Code/P6_Code.ipynb)
*   **Artificial Neural Network** : Accuracy = 84.45% 

```python
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Forming Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy Score: ", accuracy)
```

<a name="ModelOpt"></a> 

## [Model Optimisation](Code/P6_Code.ipynb)
In this step, I used GridsearchCV to find the best parameters to optimise the performance of the model. (* Unless you have an extremely powrful computer using GridsearchCV with a large number of epochs will take a lot of time.)
Using the best parameters, I improved the model accuracy by **1%**

*   **Artificial Neural Network** : Accuracy = 85.45%  

```python
# Hyperparameter tuning 
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              }
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best parameters: ",best_parameters)
print("Best accuracy: ",best_accuracy)
```


<a name="ModelEval"></a> 

## [Model Evaluation](Code/P6_Code.ipynb)
*   A confusion matrix showing the accuracy score of 84.45% achieved by the model. 

* I tested the model out

```python
# Testing data on random instance Use sc.transform to scale our data. Remember above we created the method sc
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
new_prediction
# If False returned then the customer is unlikely to churn 
```
<img src="images/Confusionmatrix.png" />

<!-- <a name="ModelProd"></a> 

## [Model Productionisation](Code/P6_Code.ipynb)
*   A confusion matrix showing the accuracy score of 97.25% achieved by the model. 
<img src="images/Confusionmatrix.png" /> -->

<a name="ModelDeploy"></a> 

## [Deployment](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
I deployed the previous model in Microsoft Power BI for business intellignece use. 
*   I exported the model as a .pkl file and applied it to the unseen data set to get churn predictions and probability predictions.
*   I visualised this in Power BI and using conditional formatting to highlight those new customer sthat are more likely to curn based on the models prediction. 

<a name="Prjmanage"></a> 

## [Project Management (Agile/Scrum/Kanban)](https://www.atlassian.com/software/jira)
* Resources used
    * Jira
    * Confluence
    * Trello 

<a name="PrjEval"></a> 

## [Project Evaluation]() 
*   WWW
    *   The end-to-end process
    *   Use of ANN in project
*   EBI 
    *   Better project management and planning would have made this project faster
    *   Time the coe took to run 

<a name="Lookahead"></a> 

## Looking Ahead
*   What next
*   Business application and better steps to preserve customer that are likely to churn

<a name="Questions"></a> 

## Questions & Contact me 
For questions, feedback, and contribution requests contact me
* ### [Click here to email me](mailto:contactmattithyahu@gmail.com) 
* ### [See more projects here](https://mattithyahudata.github.io/)

