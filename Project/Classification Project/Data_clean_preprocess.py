# To clean the data preprocess it
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

def clean_process_data(cat,con,X):
    # Creating numerical pipeline that handles missing data and also scales numerical data
    con_pipe = make_pipeline(SimpleImputer(strategy='mean'),StandardScaler())
    # Creating categorical pipeline that handles missing data and scales text data
    cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False))
    # Combine both the piplines
    pre = ColumnTransformer([
        ('cat',cat_pipe,cat),
        ('con',con_pipe,con)
    ]).set_output(transform='pandas')

    # use the pre to fit and transform X data
    X_pre = pre.fit_transform(X)

    # return the preprocessed X
    return pre,X_pre