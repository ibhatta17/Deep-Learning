# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report

data = pd.read_csv('census_data.csv')
# data.head()

#------------------------ DATA CLEANSING ------------------------

data.replace(' ?', float('nan'), inplace = True) # replacing missing values(' ?') as nan
data.dropna(axis = 0, how = 'any', inplace = True) # dropping null values

# data.workclass.unique()
data.loc[data['workclass'].str.endswith('-gov'), 'workclass'] = 'Gov'
data.loc[data['workclass'].str.startswith('Self'), 'workclass'] = 'Self Emp'
data.loc[data['workclass'].str.startswith('Without'), 'workclass'] = 'Volunteer'
# data.workclass.unique()

# data.education.unique()
data.loc[data['education'].str.endswith('th'), 'education'] = 'Literate'
data.loc[data['education'].str.startswith('Preschool'), 'education'] = 'Literate'
data.loc[data['education'].str.startswith('HS-grad'), 'education'] = 'High School'
data.loc[data['education'].str.startswith('Assoc'), 'education'] = 'Associate'
data.loc[data['education'].str.startswith('Prof-school'), 'education'] = 'Associate'
data.loc[data['education'].str.startswith('Some-college'), 'education'] = 'Some College'
# data.education.unique()

# Since we already have 'Education' feature, let's drop this feature to avoid redundancy
data.drop('education_num', axis=1, inplace=True)

# data.marital_status.unique()
data.loc[data['marital_status'].str.startswith('Married'), 'marital_status'] = 'Married'
data.loc[data['marital_status'].str.startswith('Never-married'), 'marital_status'] = 'Single'
data.loc[data['marital_status'].str.startswith('Separated'), 'marital_status'] = 'Divorced'
# data.marital_status.unique()

# data.occupation.unique()
data.loc[data['occupation'] == 'Adm-clerical', 'occupation'] = 'Clerical'
data.loc[data['occupation'] == 'Exec-managerial', 'occupation'] = 'Executive'
data.loc[data['occupation'] == 'Handlers-cleaners', 'occupation'] = 'Clerical'
data.loc[data['occupation'] == 'Prof-specialty', 'occupation'] = 'Professional'
data.loc[data['occupation'] == 'Other-service', 'occupation'] = 'Other'
data.loc[data['occupation'] == 'Sales', 'occupation'] = 'Sales'
data.loc[data['occupation'] == 'Transport-moving', 'occupation'] = 'Transport'
data.loc[data['occupation'] == 'Farming-fishing', 'occupation'] = 'Farming'
data.loc[data['occupation'] == 'Machine-op-inspct', 'occupation'] = 'Mechanical'
data.loc[data['occupation'] == 'Tech-support', 'occupation'] = 'Professional'
data.loc[data['occupation'] == 'Craft-repair', 'occupation'] = 'Mechanical'
data.loc[data['occupation'] == 'Protective-serv', 'occupation'] = 'Protective'
data.loc[data['occupation'] == 'Armed-Forces', 'occupation'] = 'Protective'
data.loc[data['occupation'] == 'Priv-house-serv', 'occupation'] = 'Clerical'
# data.occupation.unique()

# data.relationship.unique()
data.loc[data['relationship'] == 'Not-in-family', 'relationship'] = 'Not in Family'
data.loc[data['relationship'] == 'Own-child', 'relationship'] = 'Other'
data.loc[data['relationship'] == 'Unmarried', 'relationship'] = 'Other'
data.loc[data['relationship'] == 'Other-relative', 'relationship'] = 'Other'
# data.relationship.unique()

# data.race.unique()
data.loc[data['race'] == 'Asian-Pac-Islander', 'race'] = 'Asian'
data.loc[data['race'] == 'Amer-Indian-Eskimo', 'race'] = 'American Indian'

# data.native_country.unique()
data.loc[data['native_country'] == 'United-States', 'native_country'] = 'United States'
data.loc[data['native_country'] == 'Puerto-Rico', 'native_country'] = 'United States'
data.loc[data['native_country'] == 'Outlying-US(Guam-USVI-etc)', 'native_country'] = 'United States'
data.loc[data['native_country'] == 'Trinadad&Tobago', 'native_country'] = 'Trinadad & Tobago'
data.loc[data['native_country'] == 'Hong', 'native_country'] = 'Hong Kong'
data.loc[data['native_country'] == 'Holand-Netherlands', 'native_country'] = 'Netherlands'
# data.native_country.unique()

# Converting the 'income_bracket' label to a binary value
data['income_bracket'] = data['income_bracket'].apply(lambda label: int(label == '<=50K'))
# checking if income_bracket is eqaul '<=50K' then converting the boolean value to an integer(0 or 1)

#------------------------ END OF DATA CLEANSING ------------------------


#-------------------------- DATA PREPROCESSING -------------------------

X = data.drop('income_bracket', axis = 1)
y = data['income_bracket']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# data.columns
# Creating feature columns


features = []
# defining numeric columns
num_cols = ['age', 'capital_gain', 'capital_loss', 'hours_per_week'] 
for col in num_cols:
    features.append(tf.feature_column.numeric_column(col))

# defining categorical column
cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country'] 
for col in cat_cols:
    unique_val = list(data[col].unique())
    grouped_col = tf.feature_column.categorical_column_with_vocabulary_list(col, unique_val)
    embeded_col = tf.feature_column.embedding_column(grouped_col, dimension = len(unique_val))    
    features.append(embeded_col)

# Creating the input function for the estimator object
train_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size = 100, 
                                                 num_epochs = 1000, # number of epochs to iterate over data
                                                 shuffle = True # whether to read the records in shuffled order
                                                )

#------------------------ END OF DATA PREPROCESSING ----------------------


#------------------------ TRAINING THE CLASSIFIER ------------------------

model = tf.estimator.DNNClassifier(hidden_units = [16, 8, 4], # number of hidden layers and number of units in each layers
                                   # here 3 hidden layers with 10 units in each layer
                                   n_classes = 2, 
                                   feature_columns = features,
                                   activation_fn = tf.nn.relu # 'relu' activation function
                                  )

# Training the Classifier
model.train(input_fn = train_func, steps = 50000)


#------------------------ EVALUATING THE MODEL ------------------------

# Using the trained model to predict with test data
test_func = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size = len(X_test), shuffle = False)
pred = list(model.predict(input_fn = test_func))
# pred

y_pred = []
for p in pred:
    y_pred.append(p['class_ids'][0])
# y_pred

print(classification_report(y_test, y_pred))