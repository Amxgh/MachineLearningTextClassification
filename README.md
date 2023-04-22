# MachineLearningTextClassification
This is a text classification algorithm that uses machine learning. 

It trains a machine learning model that, given an input sentence, determines whether it is a factual statement that is worth fact-checking, i.e., whether the general public would be interested in knowing whether it is true or not.

**Labeled Data**: The labeled file checkworthy_labeled.csv has three columns: Id, text, Category.

**Evaluation**: The given evaluation file checkworthy_eval.csv has two columns: Id, text. This file has 1 header row and 1032 data rows. 

**Prediction**: The prediction will be produced with two columns: Id, label. The prediction file will have 1 header row and 1032 data rows too. Given each Id value in checkworthy_eval.csv, the prediction file will have a corresponding row with the same Id value. The value of column “Category” in that row will provide the prediction of your model for the corresponding sentence.
