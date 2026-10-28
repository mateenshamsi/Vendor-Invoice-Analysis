import sqlite3 
from sklearn.model_selection import train_test_split 
import pandas as pd 
def load_vendor_invoice_data(db_path:str):
    conn = sqlite3.connect("Pytohn/data/inventory.db")
    query = pd.read_sql_query("Select * from vendor_invoice")
    df=pd.read_sql_query(query,conn)
    conn.close()
    return df 

def prepare_features(df:pd.DataFrame):
    x=df[["Dollars"]]
    y=df["Freight"]
    return x,y

def split_data(x,y,test_size=0.2,random_state=42):
    return train_test_split(x,y,test_size,random_state)


