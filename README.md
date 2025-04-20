# Semantically Similar Classification of Quora Question Pairs through Natural Language Processing and Supervised Learning Techniques

## Project Objective
To identify whether a pair of questions have duplicated meaning using machine learning and deep learning.

## Motivation
Improve search engine performance on various platforms such as Quora when questions of similar semantic meaning are given by the end user. Reduces the chance of Quora users reposting not because the question was not asked before but because the user could not find relevant posts from the past.

## Selected Dataset
- Quora question pair and label from Kaggle and GLUE.
- Slightly imbalanced but still reasonable. Ratio of non-duplicate and duplicate is about 3:2.
- No meaningful numerical features in the raw dataset. Heavy feature engineering is required to formulate ML problem.

## Our Approach
Broken down into:
1. **Data Ingestion**
2. **Feature Engineering**
3. **Modelling**

### Pre-requisites
- `requirements.txt`
  - Contains the necessary Python library dependencies for installation.
- `en_core_web_sm` SpaCy model
  - `python -m spacy download en_core_web_sm`
- Pytorch with CUDA installed
  - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`

### Data Ingestion
Source code:
- `util_generate_raw_data.py`
  - Combines Kaggle and GLUE datasets together with columns qid1, qid2, question1, question2, label.

### Feature Engineering
Source code:
- `trad_ml_01_feature_engineering_func.py`
  - Utility file containing functions to perform feature engineering in `trad_ml_01_feature_engineering.ipynb`.
- `trad_ml_01_feature_engineering.ipynb`
  - Generated TF-IDF vectors and word presence vectors for every question in dataset, where questions have been lemmatized and had their stop words removed.
  - Generated the first dataframe (lets call it df1 for simplicity) using TF-IDF and word presence vectors of each question pair. Metrics generally compared how similar the vectors were via different distance metrics. The metrics are as follows:
    - id1
    - id2
    - cosine_similarity
    - manhattan_dist
    - euclidean_dist
    - jaccard_dist
    - is_duplicate
  - Generated a second dataframe (lets call it df2 for simplicity) containing the additional following features to measure string similarity using thefuzz library. Metrics are as follows:
    - fuzz_ratio: String similarity between two strings using edit distance (how many operations it takes to transform one string into another)
    - fuzz_partial_ratio: String similarity between the shorter string and best matching substring of the long string
    - token_sort_ratio: Sorts the tokens alphabetically in case the strings are out of order
    - token_set_ratio: Similar to token sort ratio, but focus on common words and ignores extra words

### Modelling
We explored the following:
- Traditional ML models
  - XGBoost (XGB)
  - Logistic Regression (LR)
  - K-nearest Neighbours (KNN)
  - Random Forest (RF)
  - Voting Classifier (best 3 base models from the above) (VC)
- Neural Networks
  - Deep neural network
  - Siamese Neural Network with feature learning using LSTM
  - Pre-trained BERT (transformer architecture) with LoRA fine tuning

Rationale for choice of traditional models:
- Linear model: LR
- Bagging: RF, VC
- Boosting: XGB
- Standalone: KNN

Conclusion: We want to begin with a diversity of models first before narrowing down.

Source code for final model evaluation can be found in:
- `util_model_evaluation.py`

#### Traditional ML models:
Motivation: We start from simple models before trying out complex models.

Source code can be found in:
- `trad_ml_02_training_and_eval_func.py`
  - Utility file containing functions for training and evaluation of traditional ML models.
- `trad_ml_02_training_and_eval.ipynb`

Model exploration was generally done using accuracy. Final model evaluation on the true test set was done using accuracy, precision, recall and f1 score. We focused more on accuracy and f1 score.

1. Tested df1 on base XGB, LR, KNN, RF (val data)
   - RF > XGB > KNN > LR
   - We remove LR permanently from consideration as we only want top 3.
   - Models generally look under fitted, hence, we went on to do feature engineering to add thefuzz features.
2. Performance on the 3 selected base models generally improved when tested on df2 (val data). Used only df2 from this point onwards.
   - XGB > RF > KNN now
   - XGB well fitted, RF severely overfitted, KNN slightly overfitted.
3. Performed hyperparameter tuning to address overfitting subsequently
   - Reduced overfitting as a result for both KNN and RF.
   - Performance of KNN improved while RF and XGB performance remained approximately the same.
   - Performance wise: XGB > RF > KNN (where XGB and RF are about the same).
   - Tuning was done using random search as there is [empirical statistical proof](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) that random search is a more efficient option compared to grid search.
4. Used the tuned models and placed into VotingClassifier
   - Tested test data (actual test data) for XGB, RF, KNN and VC
   - By accuracy: VC > RF = XGB > KNN
   - By f1: RF > VC > XGB > KNN
   - VC seems to be the best option as recall and precision are closer together compared to RF.

Areas for improvement (in decreasing order of priority):
1. Find ways to encode semantic meaning and positioning of text.
2. Address the slight imbalance in dataset (not too big of an issue as precision and recall scores were comparable).
3. For KNN, can consider dimensionality reduction to improve the model.

Extra work done in `trad_ml_02_training_and_eval.ipynb`:
- Some follow ups were PCA on dataset for KNN, addressing the data imbalance and finding ways to represent positioning and semantic meaning of text and words.
  - PCA: Did the elbow plot and chose 3 principal components but model did not perform as well, suggesting that curse of dimensionality was not an issue for KNN.
  - Undersampling: Accuracy decreased but F1 score increased in which both were expected. Models are now less biased. Full explanation in the Jupyter notebook.
  - SMOTE: Same for undersampling but accuracy drop was less, likely due to SMOTE keeping more datasets.
  - Addressing positioning and semantic meaning of text done in Siamese Neural Network and fine tuning of pre-trained BERT (see below).

#### Neural Network:
Deep Neural Network
Motivation: Simple models did not look as good. Let’s try complex modelling with deep neural networks (DNN) first.

Source code can be found in:
- `dnn_pytorch_func.py`
  - Utility file, mainly containing the DNN object.
  - DNN mainly contains 2 hidden layers with some dropouts.
- `dnn_pytorch.ipynb`
  - To train and evaluate the DNN.
  - Trained on 5 epochs as performance improvement was slowing down with increasing epochs as shown by the graph we plotted. Had to keep number of epochs low to avoid overfitting.
  - Test accuracy of about 70% and f1-score of about 66%.

Siamese Neural Network
Motivation: Feature learning using LSTM to encode word positioning into numeric features before passing into the Siamese layer for pairwise training where the loss function measures the difference between the 2 input vectors from 2 identical LSTMs.

Source code can be found in:
- `siamese_network_func.py`
  - Utility file containing code to train the Siamese Neural Network.
- `siamese_network.ipynb`
  - Train and evaluate the Siamese model in 5 epochs.
  - 5 epochs chosen in the interest of time (each epoch takes about 6-7 minutes).
  - Test accuracy of about 76% and f1-score of about 70%.

Pretrained transformer neural network (BERT) with LoRA fine tuning
Motivation: We wish to encode not just positioning of text but also capture the semantic meaning behind text. Transformers are computationally expensive to train and hence, we experimented with a pre-trained model and did fine tuning for our specific task of duplicate identification using low ranking adaptation (LoRA) fine tuning where we only train a small portion of weights.

Source code can be found in:
- `pretrained_bert_func.py`
  - Utility file containing functions to fine tune BERT.
- `pretrained_bert.ipynb`
  - Pretrained model: bert-case-uncased.
  - Performed QLoRA for optimal training with a rank of 8 as suggested by the [original research paper](https://arxiv.org/abs/2106.09685), which suggests that low rank is acceptable for most cases.
  - Training and testing on a subset of the dataset due to slow training and inference time.
  - Has excellent performance of 78% accuracy and 74% f1-score. Outperformance could be a result of a smaller dataset which causes the numbers to be slightly more inflated but it could also be because the model has the ability to understand the semantic meaning of text.

## Misc
- `images/` folder contains screenshots of our model performance (confusion matrix and metrics).
- `models/` contains the hyperparameters/weights of our models.

## Moving Forward
We could explore the following:
- More feature engineering for traditional ML models and DNN.
- Explore the outcome of addressing the slight data imbalance (addressed in the end).
- Explore the possibilities of incorporating the transformer architecture into the Siamese neural network (or directly replacing LSTM with transformer).
- Experiment with larger pre-trained models and fine tune with a slightly larger dataset if given the time.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.