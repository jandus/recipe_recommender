import pandas as pd
import numpy as np

# Function to convert all strings to lower
def clean_data(x):
    '''
    Convert all strings to lower case. 

    Parameters
    ----------
    x : str
        String.
           
    Returns
    -------
    string in lower case or empty string.. 

    '''
    if isinstance(x, list):
        return [str.lower(i) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x)
        else:
            return ''
        
# create a column with all words needed for the recommender 
def create_soup(x):
    '''
    Joins all the words to create a soup of words. 

    Parameters
    ----------
    x : str
        Text information.
           
    Returns
    -------
    soup of words : str.
        Words in a string. 

    '''
    return ' '.join(x['ingredients']) + ' ' + ' '.join(x['tags']) + ' '  + x['description']

# Add two new features ingredients_string and tags_strings - These will be used for the recommendations.
def create_string_feature(x):
    return ' '.join(x)
    