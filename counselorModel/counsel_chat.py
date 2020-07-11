#!/usr/bin/env python
# coding: utf-8

# # Counselor Generative NLP Model
# This is our code for the counselor generative model. We will begin by analyzing the dataset and doing EDA to find which attributes in our dataset we should train our model on. The dataset includes questions, the links to the posts, the therapist information, and the URL, as well as many other attributes. 
# 
# Our model is based on the HuggingFaceAI model, which we utilized. We adjusted the model in order to fit our dataset more appropriately. 

# In[61]:


from typing import Dict, Any, Callable, List, Tuple, Optional
import dill
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import (
   CountVectorizer, TfidfVectorizer
)
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.pipeline import FeatureUnion, Pipeline
import numpy as np
import torch
import re
import json
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer)
import umap
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.sparse import csr_matrix



from utils import *

sns.set()

from sklearn.utils.multiclass import unique_labels


# In[2]:


df = pd.read_csv("data/20200325_counsel_chat.csv", encoding='utf-8')
df.head()


# ## Descriptive statistics
# This is basic analysis of the dataset and what columns and attributes may be of use to us.

# In[3]:


# Number of counselors
df.groupby("therapistURL").agg("count").shape


# In[4]:


# Number of Topics
len(set(df["topic"].tolist()))


# In[5]:


# Number of responses
df.shape


# In[6]:


# Number of questions
len(set(df["questionLink"].tolist()))


# In[7]:


# Average number of responses to questions
df.groupby("questionLink").agg("count").describe()


# In[8]:


# Distribution of number of responses
fig, axs = plt.subplots(1, 2, figsize=(15, 8))
df.groupby("questionID").agg("count")["questionLink"].plot.hist(bins=15, ax=axs[0], logy=True)
axs[0].set_title("Distribution of Responses per Question", fontsize=18)
axs[0].set_ylabel("log(Count)", fontsize=15)
axs[0].set_xlabel("Number of Responses", fontsize=15)


# Number of responses
df.groupby("questionID").agg("count")["questionLink"].plot.box(ax=axs[1])
axs[1].set_title("Number of Responses per Question", fontsize=18)
axs[1].set_ylabel("Number of Responses", fontsize=15)
axs[1].set_xticklabels("")
axs[1].set_ylim([0, 30])
plt.savefig("figures/number_responses.png")


# In[9]:


df.groupby("questionID").agg("count").sort_values("questionID")


# In[10]:


# Average number of responses to questions by topic type
df.groupby(["questionLink", "topic"]).agg("count").reset_index().head()


# In[11]:


# number of counts topic
df.groupby("topic").agg("count").describe()


# In[12]:


df.groupby("topic").agg("count")["questionID"].describe()


# In[13]:


# Distribution of answers by topic
fig, ax = plt.subplots(figsize=(20, 10))
df.groupby("topic").agg("count")["questionID"].sort_values(ascending=False).plot.bar(ax=ax)
ax.set_title("Number of Responses by Topic", fontsize=30)
ax.set_xlabel("Topic", fontsize=25)
ax.set_ylabel("Number of Responses", fontsize=25)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=17)
plt.tight_layout()
plt.savefig("figures/responses_by_topics_count.png")


# From this chart, it is clear that the majority of the posts are related to depression, and then anxiety as second. 

# In[14]:


# Number of questsions asked per topic
fig, ax = plt.subplots(figsize=(20, 10))
df.groupby(["topic", "questionID"]).agg("count").reset_index().groupby("topic").agg("count")["questionID"].sort_values(ascending=False).plot.bar(ax=ax)
ax.set_title("Number of Questions by Topic", fontsize=30)
ax.set_ylabel("Number of Questions", fontsize=25)
ax.set_xlabel("Topic", fontsize=25)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=17)
plt.tight_layout()
plt.savefig("figures/number_of_questions_by_topic.png")


# In[15]:


# Distribution of response likes
# Distribution of empathy in responses
fig, ax = plt.subplots(figsize=(10, 7))
df["upvotes"].astype(int).plot.hist(bins=10, ax=ax, logy=True)
ax.set_xlabel("Number of Upvotes", fontsize=15)
ax.set_ylabel("Log(Frequency)", fontsize=15)
ax.set_title("Distribution of Upvotes", fontsize=20)
plt.savefig("figures/upvote_hist.png")


# In[16]:


df["upvotes"].describe()


# In[17]:


df["has_upvote"] = df["upvotes"].apply(lambda x: 1 if int(x) > 0 else 0)
df["has_upvote"].sum() / df.shape[0]


# In[19]:


fig, axs = plt.subplots(1, 2, figsize=(15, 7))


# Average Length of Question
df["questionLength"] = df["questionText"].apply(lambda x: len(x.split(" ")))
df["questionLength"].astype(int).plot.hist(bins=25, ax=axs[0], logy=True)
axs[0].set_xlabel("Length of Question (Words)", fontsize=15)
axs[0].set_ylabel("Log(Frequency)", fontsize=15)
axs[0].set_title("Distribution of Question Lengths", fontsize=20)
plt.savefig("figures/question_length_hist.png")

# Average Length of response
df["responseLength"] = df["answerText"].apply(lambda x: len(x.split(" ")))
df["responseLength"].astype(int).plot.hist(bins=25, ax=axs[1], logy=True)
axs[1].set_xlabel("Length of Response (Words)", fontsize=15)
axs[1].set_title("Distribution of Response Lengths", fontsize=20)
axs[1].set_ylabel("", fontsize=15)
plt.savefig("figures/response_length_hist.png")


# In[20]:


# Questions vs statements
titles = list(set(df["questionTitle"].tolist()))
questions = [x for x in titles if x[-1] == "?"]
statements = [x for x in titles if x[-1] != "?"]
print(len(questions))
print(len(statements))


# ## Predict Upvotes
# Now we will try to predict upvotes on each counselor to determine which posts are the best for training. 

# In[21]:


x_train = df[df["split"] == "train"]
y_train = x_train["has_upvote"]
x_val = df[df["split"] == "val"]
y_val = x_val["has_upvote"]


# In[22]:


classifier = svm.LinearSVC(C=1.0, class_weight="balanced")

svm_model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3))),
        ("classifier", classifier),
    ]
)

svm_model.fit(x_train["answerText"], y_train)
svm_preds = svm_model.predict(x_val["answerText"])

svm_perf_df = calculate_classification_metrics(svm_preds, y_val)
svm_ax = visualize_performance(svm_perf_df[svm_perf_df["class"] == 1],
                              ["f_score", "precision", "recall"],
                              use_class_names=False,
                              title="TF-IDF Upvote Prediction Performance")


# In[27]:


# Examine top features
coefs = svm_model.named_steps["classifier"].coef_
if type(coefs) == csr_matrix:
    coefs.toarray().tolist()[0]
else:
    coefs.tolist()
feature_names = svm_model.named_steps["tfidf"].get_feature_names()
coefs_and_features = list(zip(coefs[0], feature_names))
top_features = sorted(coefs_and_features, key=lambda x: abs(x[0]), reverse=True)[:25]


# In[ ]:


# Train BERT Model
classifier = svm.LinearSVC(C=1.0, class_weight="balanced")

dbt = BertTransformer(DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
                      DistilBertModel.from_pretrained("distilbert-base-uncased"),
                      embedding_func=lambda x: x[0][:, 0, :].squeeze(),
                      max_length=150)

bert_model = Pipeline(
    [
        ("vectorizer", dbt),
        ("classifier", classifier),
    ]
)

bert_model.fit(x_train["answerText"], y_train)


# In[65]:


bert_preds = bert_model.predict(x_val["answerText"])

bert_perf_df = calculate_classification_metrics(bert_preds, y_val)
visualize_performance(bert_perf_df[bert_perf_df["class"] == 1],
                      ["f_score", "precision", "recall"],
                      title="BERT Upvote Prediction Performance")


# Now we will start training out Bert Model with this data. *Run this on Colab*

# In[101]:


# Train BERT Model
classifier = svm.LinearSVC(C=1.0, class_weight="balanced")

dbt = BertTransformer(DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
                      DistilBertModel.from_pretrained("distilbert-base-uncased"),
                      embedding_func=lambda x: x[0][:, 0, :].squeeze(),
                      max_length=150)

tf_idf = Pipeline([
    ("vect", CountVectorizer(analyzer='word', ngram_range=(1, 2))),
    ("tfidf", TfidfTransformer())
    ])

combined_model = Pipeline([
    ("union", FeatureUnion(transformer_list=[
        ("bert", dbt),
        ("tf_idf", tf_idf)
        ])),
        ("classifier", classifier),
    ])

combined_model.fit(x_train["answerText"], y_train)


# In[102]:


combined_preds = combined_model.predict(x_val["answerText"])

combined_perf_df = calculate_classification_metrics(combined_preds, y_val)
visualize_performance(combined_perf_df[combined_perf_df["class"] == 1],
                      ["f_score", "precision", "recall"],
                      title="Combined Upvote Performance")


# In[107]:


fig, axs = plt.subplots(1, 3, figsize=(12, 6))
visualize_performance(svm_perf_df[svm_perf_df["class"] == 1],
                      ["f_score", "precision", "recall"],
                      title="TF-IDF Upvote Performance",
                      ax=axs[0],
                      ylim=[0, 1])

visualize_performance(bert_perf_df[bert_perf_df["class"] == 1],
                      ["f_score", "precision", "recall"],
                      title="BERT Upvote Performance",
                      ax=axs[1],
                     ylim=[0, 1])

visualize_performance(combined_perf_df[combined_perf_df["class"] == 1],
                      ["f_score", "precision", "recall"],
                      title="Combined Upvote Performance",
                      ax=axs[2],
                      ylim=[0, 1])
plt.savefig("figures/upvote_performance.png")


# In[106]:


svm_perf_df


# In[105]:


bert_perf_df


# In[104]:


combined_perf_df


# In[ ]:


# Get informative features


# ## Predict Topics

# In[29]:


# Number of questsions asked per topic in validation set
fig, ax = plt.subplots(figsize=(20, 10))
x_val.groupby(["topic", "questionID"]).agg("count").reset_index().groupby("topic").agg("count")["questionID"].sort_values(ascending=False).plot.bar(ax=ax)
ax.set_title("Number of Questions by Topic in Validation Set", fontsize=30)
ax.set_ylabel("Number of Questions", fontsize=25)
ax.set_xlabel("Topic", fontsize=25)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=17)
plt.tight_layout()
plt.savefig("figures/number_of_questions_by_topic_validation_set.png")


# In[ ]:


classifier = svm.LinearSVC(C=1.0, class_weight="balanced")

dbt = BertTransformer(DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
                      DistilBertModel.from_pretrained("distilbert-base-uncased"),
                      embedding_func=lambda x: x[0][:, 0, :].squeeze(),
                      max_length=150)

topics_model = Pipeline(
    [
        ("vectorizer", dbt),
        ("classifier", classifier),
    ]
)

topics_model.fit(x_train["answerText"], x_train["topic"])


# In[ ]:


topics_preds = topics_model.predict(x_val["answerText"])


# In[29]:


topics_perf_df = calculate_classification_metrics(topics_preds, x_val["topic"])
visualize_performance(topics_perf_df, ["f_score", "precision", "recall"], title="Topics Prediction")


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 7))
reducer = umap.UMAP(n_components=2)
reduced = reducer.fit_transform(dbt.transform(df["answerText"]))


# In[72]:


color_names = np.random.choice([k for k in mcolors.CSS4_COLORS.keys()], len(set(df['topic'].tolist())))
color_map = {topic: color for topic, color in zip(set(df["topic"].tolist()), color_names)}
fig, ax = plt.subplots(figsize=(10, 10))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=[color_map[x] for x in df["topic"].tolist()])
ax.set_title("UMAP Embedding of BERT Topic Features")
ax.set_xlim([-6, 7])
patches = []
for topic, color in color_map.items():
    patches.append(mpatches.Patch(color=color, label=topic))
plt.legend(handles=patches)
plt.savefig("figures/umap_topic_bert_features.png")
plt.show()


# In[73]:


df["umap_1"] = reduced[:, 0]
df["umap_2"] = reduced[:, 1]


# In[92]:


# Examine top performers
topics_to_view = ["workplace-relationships", "counseling_fundamentals", "depression", "anxiety", "family-conflict",
                 "marriage", "parenting", "relationship-dissolution"]
temp_df = df[df["topic"].isin(topics_to_view)]
color_names = np.random.choice([k for k in mcolors.CSS4_COLORS.keys()], len(set(temp_df['topic'].tolist())))
color_map = {topic: color for topic, color in zip(set(temp_df["topic"].tolist()), color_names)}


fig, ax = plt.subplots(figsize=(10, 10))
scatter = plt.scatter(temp_df["umap_1"], temp_df["umap_2"], c=[color_map[x] for x in temp_df["topic"].tolist()])
ax.set_title("UMAP Embedding of BERT Features Topic Subset")
ax.set_xlim([-6, 7])
patches = []
for topic, color in color_map.items():
    patches.append(mpatches.Patch(color=color, label=topic))
plt.legend(handles=patches)
plt.savefig("figures/umap_topic_subset_bert_features.png")
plt.show()


# In[32]:


set(x_train["topic"])


# In[37]:


classifier = svm.LinearSVC(C=1.0, class_weight="balanced")

svm_model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3))),
        ("classifier", classifier),
    ]
)

svm_model.fit(x_train["answerText"], x_train["topic"])
svm_preds = svm_model.predict(x_val["answerText"])

svm_perf_df = calculate_classification_metrics(svm_preds, x_val["topic"])
svm_ax = visualize_performance(svm_perf_df,
                              ["f_score", "precision", "recall"],
                              title="TF-IDF Topic Prediction Performance")


# In[41]:


# Examine top features
coefs = svm_model.named_steps["classifier"].coef_
if type(coefs) == csr_matrix:
    coefs.toarray().tolist()[0]
else:
    coefs.tolist()
feature_names = svm_model.named_steps["tfidf"].get_feature_names()
coefs_and_features = list(zip(coefs[0], feature_names))
top_features = sorted(coefs_and_features, key=lambda x: abs(x[0]), reverse=True)[:25]
top_features


# In[59]:


fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.barplot(x="features", y="value", data=pd.DataFrame(top_features, columns=["value", 'features']), ax=ax, color="blue")
ax.set_xticklabels([x[1] for x in top_features], rotation=90)
ax.set_title("Top 25 Features for Topic Prediction", fontsize=25)
ax.set_ylabel("Feature Weight", fontsize=15)
ax.set_xlabel("Feature Name", fontsize=15)


# # Extract Chat Bot Data
# Now we will export the model since we are done training. We are importing the HuggingFace model and connecting the Bert Model with it. Then we will export it as a json and a .bin file. *bin model couldn't be deployed*. 

# In[3]:


tokenizer_class = OpenAIGPTTokenizer
tokenizer = tokenizer_class.from_pretrained("openai-gpt")


# In[6]:


d = convert_df_to_conv_ai_dict(df, [""], ["answerText"], tokenizer, max_tokens=250, n_candidates=8)


# In[188]:


with open("./data/counsel_chat_250-tokens.json", "w") as json_file:
    json.dump(d, json_file)

