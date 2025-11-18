# To clean data and Preprocess it
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

def clean_process_data(cat,con,X):
    # Creating numerical pipeline that handles missing data and also scales numerical data
    con_pipe = (SimpleImputer(strategy='mean'),StandardScaler())
    # Creating categorical pipeline that handles missing data and scales text data
    cat_pipe = (SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False))

    # Combine both pipeline
    pre = ColumnTransformer([
        ('con',con_pipe,con),
        ('cat',cat_pipe,cat)
    ]).set_output(transform='pandas')

    # use the pre to fit and transform X data
    X_pre = pre.fit_transform(X)

    # return the preprocessed X 
    return pre,X_pre


# WRITE A CODE THAT HANDLES PREPROCESSING AND CLEANING OF NUMERICAL DATA ALONE.
def clean_process_CON_data(X):
    # Creating numerical pipeline that handles missing data and also scales numerical data
    con_pipe = make_pipeline(SimpleImputer(strategy='mean'),StandardScaler()).set_output(transform='pandas')

    # use the pre to fit and transform X data
    X_pre = con_pipe.fit_transform(X)

    # return the preprocessed X 
    return con_pipe,X_pre


# WRITE A CODE THAT HANDLES PREPROCESSING AND CLEANING OF categorical DATA ALONE.
def clean_process_CAT_data(X):
    # Creating numerical pipeline that handles missing data and also scales numerical data
    cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)).set_output(transform='pandas')

    # use the pre to fit and transform X data
    X_pre = cat_pipe.fit_transform(X)

    # return the preprocessed X 
    return cat_pipe,X_pre