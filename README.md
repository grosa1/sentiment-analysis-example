# Sentiment Analysis on the IMDb dataset

### **Context**
This repository contains two notebooks reporting a small study aimed to build sentiment analysis models for the IMDb dataset
Sentiment analysis is a common problem of Natural Language Processing (NLP) tasks to determine the so-called *sentiment*, i.e. if a word or sentence intrinsically contains a positive or negative tone.
For example, the sentence "Today is a beautiful day" has a positive sentiment, while "I hate broccoli" has a negative sentiment.
The context of the conducted analysis is represented by movie reviews from the IMDb dataset. In detail, the dataset contains several film reviews with the assigned score by the users (i.e., 1-10). Based on that score, they are grouped as "positive" and "negative".
Specifically, the dataset contains a total of 50,000 reviews already divided into train set (D<sub>train</sub>) and test set (D<sub>test</sub>).
The file `download_dataset.py` contains the code which downloads and exports the two datasets in CSV files (i.e. `train.csv` and `test.csv`).

### **Methodology**
The work presented in this notebook aims to address the following challenges:

* C1: Train and test a basic machine learning model that performs sentiment analysis on movie reviews;
* C2: Adjust and tune the model to improve its performance;
* C3: Enhance the machine learning model by using more advanced techniques.

To address the first challenge (C1), we present a simple machine learning model using classical machine learning techniques (e.g., Random Forest, Support Vector Machine, etc.) to build a classifier aimed at assigning a label that defines if the sentiment is positive or negative. The focus is on building a resource-efficient model achieving a good level of accuracy score.
The features are extracted by using a TF-IDF algorithm aimed to transform "words" into "numbers", that are understandable by the machine learning model, based on their frequency and appearance in different reviews. The detailed procedure is reported and described in section 3, based on similar works from the scientific literature dealing with text classification [1,2,3]. The text preprocessing steps and model training are performed on *D<sub>train</sub>*, while the evaluation is performed on *D<sub>test</sub>*.
To address the second challenge (C2), the evaluated NLP pipeline in C1 is tuned by performing parameter optimization via cross-validation on *D<sub>train</sub>*. Specifically, only the parameters of the TF-IDF vectorizer are tuned, by including *n-grams* to enrich the initial feature set [3].
Both C1 and C2 are addressed in `analysis.ipynb`
The last challenge (C3) is addressed by experimenting with *sentence embeddings* to improve the performance of the models in comparison with TF-IDF. The experiment and results are reported in notebook `analysis_with_embeddings.ipynb`.

### **Results and Final Remarks**
The best-performing models resulting from C1 are *LogisticRegression* and *Support Vector Machine (SVM)*, obtaining both an accuracy score of 0.87. After performing the tuning for C2, the accuracy level reached 0.88.
By combining sentence embeddings in C3, the accuracy level reached a score of 0.90 using *SVM*.
However, this is not the best result that can be achieved for this task. This work presents a basic model to perform sentiment analyses aimed at maintaining the resource requirements low, i.e., the model is lightweight and does not require particular hardware (e.g., GPUs).
Better can be done by using deep learning and, specifically, Large Language Models (LLMs). The same dataset is used in the literature as a benchmark to compare state-of-the-art models in the sentiment analysis task. In fact, the best overall is *XLNet* (Transformers architecture) achieving an accuracy score of 96.21 [4].



**References** </br>
[1] Pagano, D., & Maalej, W. (2013, July). User feedback in the appstore: An empirical study. In 2013 21st IEEE international requirements engineering conference (RE) (pp. 125-134). IEEE.</br>
[2] Iacob, C., & Harrison, R. (2013, May). Retrieving and analyzing mobile apps feature requests from online reviews. In 2013 10th working conference on mining software repositories (MSR) (pp. 41-44). IEEE.</br>
[3] Scalabrino, S., Bavota, G., Russo, B., Di Penta, M., & Oliveto, R. (2017). Listening to the crowd for the release planning of mobile apps. IEEE Transactions on Software Engineering, 45(1), 68-86.</br>
[4] PapersWithCode, *Sentiment Analysis on IMDb*, https://paperswithcode.com/sota/sentiment-analysis-on-imdb (Online: Accessed 19-Dec-2023).</br>

## Quick setup

First create a new virtual environment.

For example, using [Anaconda](https://docs.anaconda.com/free/anaconda/install/)

Next, please run:
```
conda create -n sentanalysis python=3.11
conda activate sentanalysis

pip install -r requirements.txt

pip install jupyter
```

Open and execute `analysis.ipynb` to run the NLP pipeline composed by *TF-ID* and *LogisticRegression*.
The second notebook, i.e., `analysis_with_embeddings.ipynb`, contains the NLP pipeline using *SentenceTransformers* to extract sentence embeddings. Note that this second notebook requires a GPU, thus we suggest executing it on Google Colab using the embedded link in the notebook.
Each notebook could be executed independently, downloading the required dataset in each execution. The script `download_dataset.py` could be used to download and store the dataset as CSV.

Note: The reported NLP pipelines are based on my lab lessons for the course "Hands-On Machine Learning". For basic examples of the usage of the NLP libraries, please refer to the source repository [here](https://github.com/grosa1/hands-on-ml-tutorials/tree/master/tutorial_3).
