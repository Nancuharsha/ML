#libraries to load into the program
import numpy as np
from sklearn.cluster import MeanShift
import pandas as pd

#Loading Data using pandas libirary  
df = pd.read_csv('Deduplication Problem - Sample Dataset.csv')
#placeing missing data with 0
df.fillna(0,inplace=True)

#Function to convert non-numerical data to numerical data
def handle_non_numerical_data(df):
    #coping all columns into columns variable
    columns = df.columns.values
    # Loopint through each column
    for column in columns:
        #below dictinoary act as hash table for the column
        text_digit_vals = {}
        #Getting the index of value of the element
        def convert_to_int(val):
            return text_digit_vals[val]
        # Checking whether that column data type is int or float.If not we convert the column
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            #Taking all element of the column into the list
            columns_contents = df[column].values.tolist()
            #Making a group of unique elements from the elements set of the column
            unique_elements =set(columns_contents)
            #Here x is the integer value assigned to the each unqiue element in the column
            x=1
            #We loop through unique element assign the integer values
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            #Mapping those unquie values on to the column elements
            df[column] = list(map(convert_to_int,df[column]))
    #returning the modified data         
    return df
#calling the function to convert the non-numerical data into numerical data
df = handle_non_numerical_data(df)
#classifier is called
clf = MeanShift()
#Clustering onthe data is done
clf.fit(df)
#printing the modified data
print(df)
#Getting the centroids of the clusters
cluster_centers = clf.cluster_centers_
#printing the cluster
print(cluster_centers)


