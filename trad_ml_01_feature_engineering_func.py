import numpy as np
import pandas as pd
from thefuzz import fuzz
import matplotlib.pyplot as plt
import seaborn as sns

def plot_label_distribution(y):
    value_counts = y.value_counts(normalize=True)
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=value_counts.index, y=value_counts.values)
    for i, v in enumerate(value_counts.values):
        ax.text(i, v + 0.01, f'{v:.2%}', ha='center', fontweight='bold')
    plt.xticks([0, 1], ['Not Duplicate (0)', 'Duplicate (1)'])
    plt.ylabel('Proportion')
    plt.title('Class Distribution in y_train')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

'''
Takes in 2 numpy vectors and calculates their cosine similarity
'''
def calculate_cosine_similarity(v1, v2):
    return (v1 @ v2.T).toarray()[0, 0]

'''
Takes in 2 numpy vectors and calculates their manhattan distance
'''
def calculate_manhattan_distance(v1, v2):
    return np.sum(np.abs((v1 - v2).toarray()))

'''
Takes in 2 numpy vectors and calculates their euclidean distance
'''
def calculate_euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2).power(2)))

'''
Takes in 2 numpy vectors and calculates their jaccard distance
'''
def calculate_jaccard_distance(a, b):
    intersection = a.multiply(b).sum()
    union = a.maximum(b).sum()
    return 1 - (intersection / union) if union != 0 else 0

'''
returns a list of tuple pairs,
where first idx in pair is the qid and the second idx is the Object with SpaCy attributes
'''
def get_qid_with_nlp_string_tup_pair(X, nlp):
    # represents ID tagged to question
    id_qn_pair = np.concatenate((X[['qid1', 'question1']].values, X[['qid2', 'question2']].values))
    id_qn_pair = [(qid, nlp(q)) for qid, q in id_qn_pair]
    return id_qn_pair 

'''
returns a 2D numpy array where
each row contains qid1, qid2 and the label
'''
def get_qid_pair_with_label(X, y):
    # stores an array of triples qid1, qid2 and label
    q1q2_id_label_triple = np.concatenate(
        (X[['qid1', 'qid2']].values, 
        y.values.reshape(-1, 1)), axis = 1)
    return q1q2_id_label_triple

'''
For each string, obtain the string in lemmatized form with stop words removed
'''
def get_list_of_lemmatized_string_with_no_stop_words(id_qn_pair):
    # obtain lemmatized non-stop words token from the strings
    id_qn_lemma_pair = [(qid, [token.lemma_ for token in q if not token.is_stop]) for qid, q in id_qn_pair]

    # join back the tokens after removing stop words and lemmatizing
    id_qn_lemma_str_pair = [(qid, ' '.join(lemma_lst)) for qid, lemma_lst in id_qn_lemma_pair]

    # since it is a tuple pair with qid and string, take out only the string
    lemma_str = list(map(lambda x: x[1], id_qn_lemma_str_pair))
    return lemma_str

'''
for each question pair, calculate the following:
1) cosine similarity
2) manhattan distance
3) euclidean distance
4) jaccard distance

and returns them with the question pair and label as 1 new row in a new pandas dataframe
'''
def generate_df(q1q2_id_label_triple_arr, id_to_score_vec_dict, id_to_presence_vec_dict):
    res = []
    for id1, id2, label in q1q2_id_label_triple_arr:
        v1, v2 = id_to_score_vec_dict.get(id1), id_to_score_vec_dict.get(id2)
        cosine_similarity = calculate_cosine_similarity(v1, v2)
        manhattan_distance = calculate_manhattan_distance(v1, v2)
        euclidean_distance = calculate_euclidean_distance(v1, v2)
        v3, v4 = id_to_presence_vec_dict.get(id1), id_to_presence_vec_dict.get(id2)
        jaccard_distance = calculate_jaccard_distance(v3, v4)
        res.append([
            id1,
            id2,
            cosine_similarity,
            manhattan_distance,
            euclidean_distance,
            jaccard_distance,
            label
            ])

    df = pd.DataFrame(res, 
                    columns = ['id1', 
                                'id2', 
                                'cosine_similarity', 
                                'manhattan_dist', 
                                'euclidean_dist', 
                                'jaccard_dist', 
                                'is_duplicate']
                    )
    return df

'''
find fuzz ratio, fuzz partial ratio, token sort ratio, token set ratio
- Fuzz ratio :  String similarity between two strings using edit distance (how many operations it takes to transform one string into another)
- Fuzz partial ratio:  String similarity between the shorter string and best matching substring of the long string
- Token sort ratio: Sorts the tokens alphabetically in case the strings are out of order
- Token set ratio: Similar to token sort ratio, but focus on common words and ignores extra words
'''
def add_thefuzz_features(df, X):
    df_final = df.copy()
    df_final['fuzz_ratio'] = [fuzz.ratio(q1, q2)/100 for q1, q2 in zip(X['question1'], X['question2'])]
    df_final['fuzz_partial_ratio'] = [fuzz.partial_ratio(q1, q2)/100 for q1, q2 in zip(X['question1'], X['question2'])]
    df_final['token_sort_ratio'] = [fuzz.token_sort_ratio(q1, q2)/100 for q1,q2 in zip(X['question1'], X['question2'])]
    df_final['token_set_ratio'] = [fuzz.token_set_ratio(q1, q2)/100 for q1,q2 in zip(X['question1'], X['question2'])]
    return df_final