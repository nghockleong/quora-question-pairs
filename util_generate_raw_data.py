import pandas as pd

'''
Takes in a .csv or .tsv file in the data/ 
directory and returns a Pandas dataframe
'''
def read_csv_file(filename):
    if filename.endswith('.tsv'):
        df = pd.read_csv(f'data/{filename}', sep = '\t')
    else:
        df = pd.read_csv(f'data/{filename}')
    return df

'''
Takes in an arbitrary number of filenames as strings
and returns the combined Pandas dataframe
'''
def generate_raw_data(*args):
    # Combine all the .csv or .tsv files together
    df = pd.concat(map(lambda x: read_csv_file(x), args))
    print('train set dimensions: ', df.shape)
    # Remove duplicated data since there might be overlap from 2 different sources
    df = df.drop_duplicates(subset = ['question1', 'question2'])
    print('train set dimensions after dropping duplicates by question pair: ', df.shape)
    # Check for duplicated ID since we are using 2 different sources
    df = df.drop_duplicates(subset = ['qid1', 'qid2'])
    print('train set dimensions after dropping duplicates by qid pair: ', df.shape)
    df = df.drop('id', axis = 1)
    df = df.dropna()
    return df