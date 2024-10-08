# Classifying emotions in English social media texts
### Capstone Project for UC Berkeley ML/AI Professional Certificate)
This project aims to develop a robust classification model to identify predominant emotion in English texts. By leveraging advanced machine learning techniques and natural language processing algorithms, we attempt to discern emotional undertones and categorize them into distinct states: anger (0), fear (1), joy (2), love (3), sadness (4) and surprise (5). The models will be trained on a large corpus of annotated texts to ensure high accuracy and generalizability. 

<p align="center">
<img src="images/inside_out_2.webp" width="300" height="200">
</p>

 Understanding emotion in English texts is fascinating because it unlocks insights into human feelings, intentions, and social dynamics that are often subtle and complex.The initial idea behind this project was sparked by watching *Inside Out 2* earlier this summer, and my motivation here is to strengthen my foundation in classification techniques, and build my skill-set in text analytics and Natural Language Processing (NLP).


 #### Business Applications/Use Cases:
A model that can detect emotions in texts has valuable business applications, particularly in **human resources**. By analyzing employee engagement surveys or feedback, companies can detect emotions like *anger* or *sadness*, identifying areas of discontentment. This could then be used to improve workplace morale and productivity, and to decrease attrition rates in organizations.

In **customer service**, emotion detection in support tickets or social media can help prioritize more urgent issues. Additionally, in **marketing**, classifying emotions in user-generated content can enhance recommendation systems and ad targeting. For instance, identifying emotions like *joy* or *love* can help brands engage more with supporters of the product/service.

I believe that emotion detection can support more personalized and emotionally intelligent decision-making.

## Data Overview
The data for this project was sourced from 3 different labeled Kaggle datasets:
1. **Emotion Classification NLP:** 7000 rows with 4 labeled emotions: anger, fear, joy, sadness

    https://www.kaggle.com/datasets/anjaneyatripathi/emotion-classification-nlp/data

2. **Emotions dataset for NLP:** 20k rows with all 6 emotion labels.

    https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?select=train.txt

3. **Emotions:** Over 400k rows with all 6 emotion labels.
    https://www.kaggle.com/datasets/nelgiriyewithana/emotions


<p align="center">
<img src="images/data_by_source.png" width="700">
</p>

After aggregating data from all the sources, we have an aggregated dataset with ~444k rows, and the pie chart below displays the breakdown of emotion labels in final dataset. *Joy* is the most represented emotion in 33.7% of observations, followed by *sadness* at 29%. The least detected emotion is *surprise* in only 3.5% of the observations. 

It is worth noting this imbalance in the dataset because it influences how we decide to score our classification models in the later steps.

<p align="center">
<img src="images/pct_emotions_final_data.png" width="300">
</p>

We split this aggregated data into development (75%) and validation (25%) datasets. The former will be split into training and test sets in the model development stage. The validation set will not be used to build models, only to evaluate them.

### Clean and Pre-process Data
NLP preprocessing typically involves the following steps:

1. **Removing unncessary data:** Remove any punctuation marks, URLs/HTML links, numbers, special characters, and emojis.

2. **Convert to Lowercase:** Ensures uniformity and avoid treating the same word in different cases as distinct

3. **Lemmatization:** We use the spaCy english model to tokenize and lemmatize the words.

4. **Stop Words:** In addition to the nltk English stop words, we include words like *'feel'*, *'think*', *'like'*, and so on which are very common in our dataset.


### Word Cloud
Before building any models, we want to see are there any words that are more commonly associated with a certain emotion.

<p align="center">
<img src="images/wordclouds.png" width="700">
</p>

As we would intuitively expect, words like 'anxious', 'scared', 'afraid' are common in texts with the label *fear*, and 'impreseed', 'curious', 'amazed' are common in texts with the label *surprise*. However, there are also words like 'could' and 'things' that appear to be common in multiple emotion labels.

## Models
We need to convert text into finite length vectors to be able to train our machine learning models.

### TF-IDF and Classification
TF-IDF stands for Text Frequency - Inverse Document Frequency and is based on the Bag of Words (BoW) approach, which simply counts the occurence of words in a document. TF-IDF contains insights about the relevancy of words in a documents. It highlights words that are frequent in a specific document (in this case, tweet/text) but rare across the entire corpus, emphasizing their significance.

After converting texts into a vector representation using TF-IDF, we experiment with a variety of classification techniques:

- **Multinomial Naive Bayes**: A classification algorithm based on Bayes' theorem that assumes features are distributed according to a multinomial distribution. This is typically used for text classification where features are word counts or frequencies. It calculates the probability of a document belonging to each class by considering the likelihood of each word given the class and applying Bayes' theorem to make predictions.

- **Logistic Regression**: This is a linear model used for binary classification tasks which estimates the probability that a given input belongs to a certain class. We use this for multi-class classification using 6 One-vs-Rest classifiers, one for each emotion label.

- **Support Vector Machines**: SVMs work by finding the hyperplane that best separates the classes in the feature space with the maximum margin. While non-linear hyperplanes are possible, for our project we use a linear kernel.

- **Decision Trees and Random Forests**: Decision trees build a model by recursively splitting the data into subsets based on the most informative features. They are easy to interpret and visualize but can be prone to overfitting. Random Forests is an ensemble approach that combines multiple decision trees using bootstrapped samples of the data and aggregates those to make the final classification. It helps mitigate overfitting and is robust to noise.

#### How to measure classification performance?

Accuracy, as the name implies, is the percentage of labels that are classified correctly. While a higher accuracy sore is always better, but in cases of highly imbalanced data, this may not capture the model performance accurately. 

$$Recall = \frac{True Positives}{True Positives + False Negatives}$$

We choose to look at recall, in addition to accuracy, since a false negative does not have a higher cost compared to a false positive. In simple terms, *recall*, for a specific class, is the percentage of labels with that class that are classified correctly. 


<p align="center">
<img src="images/tfidf_recall_accuracy.png" width ="500">
</p>

Using TF-IDF vectorization, even a simple (and fast) multinomial naive-bayes model has a high accuracy of 86% on the test data. However, if we look at recall by class, this model severly underperforms for *love* and *surprise* class labels, which are under-represneted in the data. SVM has an accuracy comparable to that of Logistic Regression, but again the recall metrics are lower for less frequent classes.

#### Precision Recall Curve
A precision-recall curve is a graphical representation of a classification model's performance, plotting precision against recall for various threshold values, helping to evaluate the trade-off between correctly identifying positive instances and minimizing false positives.

$$Precision = \frac{True Positives}{True Positives + False Positives}$$
<p align="center">
<img src="images/precision_recall_tfidf.png" width ="850">
</p>

The image corroborates with the story, that <u>**Logistic Regression** seems to be the best classifier for TF-IDF vectorization</u>, particularly beacuse of higher recall and accuracy in the underreprsented class labels.


<!-- <p align="center">
<img src="images/decision_tree_depth_5.png">
</p> -->

#### Evaluating Logistic Regression Classifier using Validation Data

We kept 25% of the total data (~100k observations) separate to be used as a validation dataset to evaluate performance. TF-IDF vectorization + Logistic Regression classifier has a 89% accuracy of this data, and recall/precision/f1 metrics similar to the test dataset.

<p align="center">
<img src="images/tfidf_logreg_validation.png", width= "400">
</p>

One advantage of using TF-IDF vectorization is the high interpretability of the classification problems. For each emotion label, the image below highlights the 15 most important features.

<p align="center">
<img src="images/tfidf_feature_importance.png">
</p>

### Word2Vec Embeddings and Classification using SVM/Logistic Regression
There are two distinct disadvantages of using TF-IDF vectorization.It is a statistical measure with an understanding of relative importance of words, but it fails to capture any semantic understanding of the words in a text. Additionaly, the vector representations are sparse, and can have very high dimensionality. 

On the other hand, Word2Vec is a neural network-based approach that learns dense, continuous vector representations of words by capturing their contextual relationships: words with similar meanings have similar vector representations. To train the model, we use an unsupervised learning technique which scans the entire corpus, and in this process the model determines which words the target word occurs with more often. The vector size does not have to be as large as the vocabulary ( we use 300), but we lose a lot of the interpretability.

*<u>Note</u>:* We still pre-process the text, but for this model we do not use lemmatization.

#### Google News vs Custom Trained Word2Vec Model

Both of these approaches have distinct advantages and trade-offs:

- **Google's Pretrained Model**: This model is trained on a large corpus like Google News, offers high-quality, general-purpose embeddings that can capture a broad range of word meanings and relationships. Using this model is convenient in case of limited data or computational constraints as it does not require additional training.

- **Custom-Trained Model**: This approach allows us to tailor the embeddings to the specific vocabulary and context, which can capture domain-specific nuances and relationships more effectively. 

Since we are dealing with a very specific task here, and a large training sample of over 200k rows, the first instinct was to use a custom-trained model. However, when we look at the 10 words most similar to *affectionate*, it is evident that our custom-model is not able to capture the semantics of words very well. Most of the top 10 most similar words, in fact appear to be antonyms. For this reason, we are biased towards using the Google pre-trained model.


<p align="center">
<img src="images/w2v_affectionate.png" height = "175">
</p>

#### Convert sentences to vectors

To convert Word2Vec word embeddings into sentence embeddings, we have to aggregate the individual word vectors to represent the entire sentence. The most common approach to do so is to average the vectors of all the words in the sentence. Summing up the vectors can bias the results for sentences that are much longer. Alternatively, we can use weighted averaging, where weights are based on term frequency or other factors. 

For our project, we use IDF (Inverse Document Frequency) weights to average the word vectors into a single sentence vector. This has the same advantage as TF-IDF of highlighting words that are present in a sentence, but infrequent across the entire corpus.

<p align="center">
<img src="images/word2vec_similarity.png" height = "250">
</p>

In the image above, we try to visualize the cosine similarity between sentences using Google's pre-trained and our custom-trained Word2Vec embeddings. We decide to proceed with the former, since our custom-trained model apprears to have higher similarity scores for different emotions, especially *anger*/*fear* and *sadness*/*surprise*. 

#### Evaluating classification using Word2Vec

We build two classification models using the Word2Vec embeddings: One-vs-Rest Logistic Regression and a Support Vector Classfier. Upon comparing the accuracy and recall metrics, it is evident that Word2Vec embedding models underperform across the board.

<p align="center">
<img src="images/tfidf_vs_w2v.png" height = "220">
<img src="images/w2v_logreg_precision_recall.png" height = "220">
</p>

While disappointing, this result is not entirely unexpected, due to a variety of reasons:

* *Sparse Dataset*: Despite having a large number of observations, we are dealing with a very limited vocabulary in our dataset. In such cases, TF-IDF may perform better because it directly measures term importance without relying on extensive context.
* *Short Texts*: Since most of the texts we look at are under 100 words, TF-IDF’s term weighting is more straightforward as Word2Vec’s context-based embeddings may not be able to capture sufficient information in such scenarios.
* *Domain-Specific Terminology*: We use Google News trained Word2Vec model, since our custom model did not do a great job at capturing semantics. However the disadvantage is that the training corpus does not accurately capture the relevance of words/terms.


### BERT Transformer and Keras for Classification

So far the best model is a One-vs-Rest Logistic Regression using TF-IDF embeddings. To see if we can improve upon this performance, we experiment with a Neural Network classification model using BERT embeddings.

**BERT** (Bidirectional Encoder Representations from Transformers) is a transformer-based model designed for natural language understanding, introduced by Google. Unlike traditional models that read text sequentially (left-to-right or right-to-left), BERT reads text in both directions simultaneously, capturing the context from both sides of a word or token. This bidirectional approach allows BERT to understand the meaning of words based on their full context, leading to improved performance. 

<p align="center">
<img src="images/bert_training_validation.png" height = "250">
</p>

With this model, we reach an accuracy of over **93%** on both the training and validation datasets.

#### Outperforms Logistic Regression with TF-IDF Vectorization
The Keras NN classifier using BERT embeddings significantly outperforms the Logistic Regression classifier with TF-IDF embeddings due to its ability to capture complex contextual information and nuanced relationships in the text. 

<p align="center">
<img src="images/tfidf_w2v_bert.png" height = "200">
<img src="images/bert_precision_recall_curve.png" height = "200">
</p>

However, like other models, the classfier performs best for frequent labels like *joy*/*sadness*, and underperforms when it comes to *love*/*surprise*. We can clearly see this in the precision-recall curves above.

## Is our model accurate on external data?

We have over 400k samples in our dataset, out of which over 100k were kept aside solely for validation. However, we are still dealing with a very limited vocabulary. In order to test the robustness of our classifier, I generated 20 completely new text samples using ChatGPT - and our model was able to classify 17 of them correctly.

<p align="center">
<img src="images/prediction_gpt_samples.png" height = "450">
</p>

A few things I would like to highlight here:

- 2 of the mis-classified samples belong to the class *surprise*. Since only 3.5% of our training data belongs to this class, it makes sense that the classifier performs the worst here.
- *Every moment spent with you feels like a beautiful dream come true.* is labeled as *love*, whereas the classfier identifies it as *joy*. This may not be entirely inaccurate as the text has a predominant joyous tone, however our model is unable to capture the romance embedded in the text.

## Next Steps
Moving forward, we can incorporate several techniques to increase the model classification and interpretation:

**1. Class Imbalance**: To better handle underrepresented emotions like *surprise* and *love*, we will explore techniques such as oversampling (e.g., SMOTE), undersampling the majority classes, and adjusting class weights in the loss function. These methods can help improve recall for minority classes without sacrificing overall accuracy.

**2. Data Augmentation**: As mentioned before, while we have a large number of samples, most of them share a specific structure and a limited vocabulary. We would like to expand the dataset to include more varied sources of emotion-labeled texts to increase generalizability.

**3. Model Interpretability**: Conducting a deeper error analysis will help identify patterns in misclassifications, particularly in cases where emotions like *love* and *joy* overlap.

**4. Real-Time Classification**: A real-time model can be integrated using data pipelines that stream survey responses, customer chats, etc into the model via Rest APIs. The model could be hosted on cloud services (e.g., AWS Lambda) to process incoming text via BERT embeddings or custom NLP pipelines. Classified emotions can trigger automated actions like escalating tickets or generating alerts by integration with CRM or HR platforms.
