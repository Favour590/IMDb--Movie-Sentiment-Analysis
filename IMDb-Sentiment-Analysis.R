#IMDb MOVIE REVIEW SENTIMENT ANALYSIS

#Project: Binary sentiment classification of IMDb movie reviews
#Description: This script imports, explores, cleans and analyses 
#IMDb movie reviews, then trains and evaluates machine learning
#models and applies LIME and a coefficient-based SHAP analysis.


#Install and load requested packages
install.packages(c("tidyverse", "tidytext", "wordcloud", "RColorBrewer"))
install.packages("text2vec")
install.packages("glmnet")
install.packages("e1071")
install.packages("LiblineaR")
install.packages("pROC")
options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages("lime")

library(tidyverse)
library(tidytext)
library(wordcloud)
library(RColorBrewer)
library(text2vec)
library(glmnet)
library(e1071)
library(LiblineaR)
library(pROC)
library(lime)
library(tidyr)
library(dplyr)
library(ggplot2)

#Import the Dataset
file_path<-file.choose()
reviews <-read.csv(file_path, stringsAsFactors = FALSE)

readBin(file_path, what="raw", n=12)
basename(file_path)
tools::file_ext(file_path)

#Initial Data Exploration
dim(reviews)
names(reviews)
head(reviews)
str(reviews)
summary(reviews)

table(reviews$sentiment)
colSums(is.na(reviews))
sum(duplicated(reviews$review))
duplicate_reviews<-reviews[duplicated(reviews$review), ]
head(duplicate_reviews, 10)

table(duplicate_reviews$sentiment)


review_label_check <- aggregate(sentiment~review, data =reviews, FUN =function(x) length(unique(x)))
review_label_check <-review_label_check[review_label_check$sentiment >1, ]
nrow(review_label_check)


#Data cleaning 
reviews_clean <- reviews[!duplicated(reviews$review), ]
dim(reviews_clean)
sum(duplicated(reviews_clean$review))

empty_reviews<- sum(trimws(reviews_clean$review) =="")
empty_reviews
head(reviews_clean$review, 3)

#Text processing and tokenisation
reviews_processed <-reviews_clean
reviews_processed$review <-gsub("<br />", " ", reviews_processed$review)
head(reviews_processed$review, 3)
glimpse(reviews_processed)

reviews_processed$review<- tolower(reviews_processed$review)

reviews_processed$review <-gsub("[[:punct:]]", " ", reviews_processed$review)

reviews_processed$review <-gsub("[[:digit:]]", " ", reviews_processed$review)

reviews_processed$review <-gsub("\\s+", " ", reviews_processed$review)
reviews_processed$review <- trimws(reviews_processed$review)



reviews_tokens <- reviews_processed %>%
  dplyr::mutate(review_id = dplyr::row_number()) %>%
  tidytext::unnest_tokens(word, review) %>%
  dplyr::anti_join(tidytext::stop_words, by ="word")
head(reviews_tokens, 10)
glimpse(reviews_tokens)

#Word frequency analysis
word_counts <- reviews_tokens %>% count(word, sort = TRUE)
head(word_counts)

top_words <- word_counts %>% slice_max(n, n =10)

ggplot(top_words, aes(x = reorder(word, n), y = n))+geom_col() +coord_flip() +
  labs(title = "Top 10 Most Frequent Words in IMDb Reviews",
       x= "Words", y="Frequency")+ theme_minimal()


#Bing lexicon sentiment analysis
bing_words <-get_sentiments("bing")
review_sentiment_words <- reviews_tokens %>% inner_join(bing_words, by = "word")

sum(duplicated(bing_words$word))
bing_words %>%
  count(word)%>%
  filter(n >1)

bing_words %>%
  filter(word%in% c("envious", "enviously","enviousness"))

bing_clean <- bing_words %>%
  group_by(word) %>%
  filter(n_distinct(sentiment)==1) %>%
  ungroup() %>%
  distinct(word, sentiment)

sum(duplicated(bing_clean$word))
review_sentiment_words <-reviews_tokens%>%
  inner_join(bing_clean, by ="word")
head(review_sentiment_words)

sentiment_summary <-review_sentiment_words %>%
  count(sentiment.x, sentiment.y)
sentiment_summary


ggplot(sentiment_summary,
       aes(x = sentiment.x,
           y = n,
           fill =sentiment.y)) +
           geom_col(position ="dodge")+
             labs(title="Word Sentiment Across IMDb Reviews",
                  x= "IMDb Review Sentiment", 
                  y= "Word Frequency",
                  fill ="Bing Sentiment")+
             theme_minimal()


#Distribution of review word counts
reviews_processed$word_count <- stringr::str_count(
  stringr::str_trim(reviews_processed$review),
  "\\S+"
)


review_word_counts <- reviews_tokens %>%
  count(review_id, name = "word_count")

ggplot(review_word_counts, aes(x = word_count)) +
  geom_histogram(
    binwidth = 1,
    fill = "skyblue",
    color = "black"
  ) +
  labs(
    title = "Distribution of Review Word Counts",
    x = "Number of Words",
    y = "Number of Reviews"
  ) +
  theme_minimal()


#Positive and negative word frequencies
sentiment_word_counts <-review_sentiment_words%>%
  count(word, sentiment.y, sort = TRUE)
head(sentiment_word_counts, 10)

positive_words <- sentiment_word_counts %>%
  filter(sentiment.y =="positive") %>%
  slice_max(n, n =10)
head(positive_words)

negative_words <- sentiment_word_counts %>%
  filter(sentiment.y =="negative")%>%
  slice_max(n, n =10)
head(negative_words)


ggplot(positive_words, aes(x =reorder(word, n), y = n)) +
  geom_col(fill ="forestgreen")+
  coord_flip()+
  labs(
    title = "Top 10 Positive Words in IMDb Reviews",
    x = "Words",
    y = "Frequency"
  ) +
  theme_minimal()


ggplot(negative_words, aes(x =reorder(word, n), y = n))+
  geom_col(fill ="firebrick")+
  coord_flip()+
  labs(
    title = "Top 10 Negative words in IMDb Reviews",
    x ="Words",
    y ="Frequency"
  )+
  theme_minimal()


#count total positive and negative words
review_sentiment_count <- review_sentiment_words %>%
  count(sentiment.y, sort = TRUE)
head(review_sentiment_count)

ggplot(review_sentiment_count,
       aes(x = sentiment.y, y = n, fill= sentiment.y)) +
  geom_col()+
  labs(
    title="Positive vs Negative Words in IMDb Reviews",
    x ="Sentiment",
    y ="Word Count"
  ) +
  theme_minimal()


#Compare Bing word sentiment across IMDb review categories
review_comparison<- review_sentiment_words %>%
  count(sentiment.x, sentiment.y)
review_comparison

review_comparison %>%
  group_by(sentiment.x) %>%
  mutate(percentage = round(n / sum(n) *100, 2))

#TF-IDF Analysis
tf_idf_words <-reviews_tokens %>%
  count(review_id, word, sort = TRUE) %>%
  bind_tf_idf(word, review_id, n)
glimpse(tf_idf_words)
head(tf_idf_words, 10)

tf_idf_words %>%
  arrange(desc(tf_idf)) %>%
  select(word, review_id, n, tf, idf, tf_idf) %>%
  head(10)

top_tfidf_words <-tf_idf_words %>%
  group_by(word) %>%
  summarise(mean_tf_idf = mean(tf_idf), .groups = "drop") %>%
  arrange(desc(mean_tf_idf)) %>%
  slice_head(n =10)

top_tfidf_words


ggplot(top_tfidf_words, aes(x= reorder(word, mean_tf_idf),
                            y= mean_tf_idf)) +
  geom_col(fill ="steelblue") +
  coord_flip()+
  labs(
    title = "Top 10 Words by TF_IDF Score",
    x ="Words",
    y ="Mean TF_IDF"
  )+ 
  theme_minimal()

tf_idf_words %>%
  arrange(desc(tf_idf)) %>%
  head(30)
names(tf_idf_words)

tf_idf_summary <- tf_idf_words %>%
  group_by(word) %>%
  summarise(mean_tf_idf =mean(tf_idf), .groups = "drop")

tf_idf_summary %>%
  arrange(desc(mean_tf_idf)) %>%
  head(30)

tf_idf_words %>%
  filter(word %in% c("trivialboring", "barwork", "someplaces")) %>%
  select(review_id, word, tf_idf) %>%
  head(10)

reviews_processed[18633, "review"]
reviews_processed[23538, "review"]

top_tfidf_words <- tf_idf_words %>%
  group_by(word) %>%
  summarise(mean_tf_idf = mean(tf_idf), .groups = "drop")%>%
  arrange(desc(mean_tf_idf))%>%
  slice_head(n=10)
top_tfidf_words

tf_idf_summary <- tf_idf_words %>%
  group_by(word)%>%
  summarise(
    document_frequency = n_distinct(review_id),
    mean_tf_idf =mean(tf_idf),
    .groups = "drop"
  ) %>%
  filter(document_frequency>= 10) %>%
  arrange(desc(mean_tf_idf))

top_tfidf_words <- tf_idf_summary %>%
  slice_head(n =10)
top_tfidf_words


ggplot(top_tfidf_words,
       aes(x =reorder(word, mean_tf_idf),
           y =mean_tf_idf)) +
  geom_col(fill ="steelblue")+
  coord_flip() +
  labs(
    title = "Top 10 Words by TF_IDF Score",
    x ="Words",
    y ="Mean TF_IDF Score"
  ) +
  theme_minimal()

#Word Cloud
positive_wordcloud <- sentiment_word_counts %>%
  filter(sentiment.y =="positive")%>%
  arrange(desc(n))

head(positive_wordcloud, 10)

wordcloud(
  words = positive_wordcloud$word,
  freq = positive_wordcloud$n,
  max.words = 100,
  random.order = FALSE,
  colors = brewer.pal(8, "Dark2")
)

negative_wordcloud <- sentiment_word_counts %>%
  filter(sentiment.y =="negative") %>%
  arrange(desc(n))
head(negative_wordcloud,10)

wordcloud(
  words = negative_wordcloud$word,
  freq = negative_wordcloud$n,
  max.words = 100,
  random.order = FALSE,
  colors = brewer.pal(8, "Reds")[5:8]
)

#Heatmap of positive and negative words
heatmap_words <- sentiment_word_counts %>%
  group_by(sentiment.y)%>%
  slice_max(order_by = n, n=10) %>%
  ungroup()
head(heatmap_words, 20)


ggplot(heatmap_words,aes(x = sentiment.y, y= reorder(word, n), fill =n))+
  geom_tile(color = "white")+
  scale_fill_gradient(low ="white", high = "steelblue") +
  labs(
    title ="Heatmap of Top Positive and Negative Words",
    x ="Sentiment",
    y= "Word",
    fill ="Frequency"
  ) +
  theme_minimal()

#Prepre data for machine learning 

set.seed(123)
reviews_model <- reviews_processed %>%
  mutate(review_id =row_number())

train_data <- reviews_model %>%
  group_by(sentiment) %>%
  slice_sample(prop = 0.8) %>%
  ungroup()

test_data <- reviews_model %>%
  filter(!review_id %in% train_data$review_id)

dim(train_data)
dim(test_data)

table(train_data$sentiment)
table(test_data$sentiment)


#Create training vocabulary
tokens_train <-word_tokenizer(train_data$review)



#Remove stopwords from training tokens
tokens_train <-lapply(tokens_train, function(words) {
  words[!words %in% tidytext::stop_words$word]
})

it_train <-itoken(
  tokens_train,
  progressbar = FALSE
)


vocab <-text2vec::create_vocabulary(it_train)


#Prune vocabulary
vocab <-text2vec::prune_vocabulary(
  vocab,
  term_count_min = 5,
  doc_proportion_max = 0.5
)
vocab


#Create document_term matrix
vectorizer <- text2vec::vocab_vectorizer(vocab)

dtm_train <- text2vec::create_dtm(
  it_train,
  vectorizer
)

dim(dtm_train)

#Create test tokens
tokens_test <-text2vec::word_tokenizer(test_data$review)

#Remove stopwords
tokens_test <-lapply(tokens_test, function(words){
  words[!words %in% tidytext::stop_words$word]
})

#Create test iterator
it_test <- text2vec::itoken(
  tokens_test,
  progressbar = FALSE
)

#Create test document_term matrix
dtm_test <-text2vec:: create_dtm(
  it_test,
  vectorizer
)

dim(dtm_test)


#Prepare sentiment labels 
y_train <-ifelse(train_data$sentiment== "positive", 1, 0)
y_test <-ifelse(test_data$sentiment=="positive", 1, 0)

table(y_train)
table(y_test)


#logistc Regression Model
#Train logistic regression
set.seed(123)

logistic_model <-glmnet::glmnet(
  x=dtm_train,
  y=y_train,
  family="binomial",
  alpha=1,
  nlambda = 20,
  maxit = 10000
)
print(logistic_model)

length(logistic_model$lambda)
logistic_model$lambda

logistic_pred <- predict(
  logistic_model,
  newx = dtm_test,
  s = logistic_model$lambda[7],
  type = "response"
)

head(logistic_pred)

logistic_class <- ifelse(
  logistic_pred >= 0.5,
  1,
  0
)
head(logistic_class)

logistic_accuracy <- mean(logistic_class == y_test)
logistic_accuracy


logistic_cm <- table(
  Actual =y_test,
  Predicted =logistic_class
)
logistic_cm


precision <-4444 / (4444 + 943)
recall <- 4444 / (4444 + 533)
f1_score <- 2 * (precision * recall) / 
  (precision + recall)
precision
recall
f1_score


logistic_result<-data.frame(
  Model ="Logistic Regression",
  Accuracy = logistic_accuracy,
  Precision = precision,
  Recall = recall,
  F1 = f1_score
)
logistic_result


write.csv(
  logistic_result,
  "logistic_regression_results.csv",
  row.names = FALSE
)

#Naive Bayes Model
negative_counts <- Matrix::colSums(
  dtm_train[y_train ==0, , drop = FALSE]
)

positive_counts <- Matrix::colSums(
  dtm_train[y_train ==1, , drop =FALSE]
)


smoothing <- 1

negative_prob <- (negative_counts + smoothing) /
  (sum(negative_counts) + smoothing * ncol(dtm_train))

positive_prob <- (positive_counts + smoothing) /
  (sum(positive_counts) + smoothing * ncol(dtm_train))

head(positive_prob)
head(negative_prob)


prior_negative <- mean(y_train ==0)
prior_positive <- mean(y_train ==1)

prior_positive
prior_negative

negative_score <- log(prior_negative) +
  dtm_test %*% log(negative_prob)

positive_score <- log(prior_positive) +
  dtm_test %*% log(positive_prob)


naive_bayes_pred <- ifelse(
  positive_score > negative_score,
  1,
  0
)


head(naive_bayes_pred)

naive_bayes_cm <- table(
  Actual = y_test,
  Predicted = naive_bayes_pred
)
naive_bayes_cm

naive_bayes_accuracy <- mean(
  naive_bayes_pred ==y_test
)

naive_bayes_precision <- 4174 /(4174 + 618)
naive_bayes_recall <- 4174 / (4174 + 803)

naive_bayes_f1 <- 2* (
  naive_bayes_precision * naive_bayes_recall
) / (
  naive_bayes_precision + naive_bayes_recall
)

naive_bayes_accuracy
naive_bayes_precision
naive_bayes_recall
naive_bayes_f1


naive_bayes_result <- data.frame(
  Model = "Naive Bayes",
  Accuracy = naive_bayes_accuracy,
  Precision = naive_bayes_precision,
  Recall = naive_bayes_recall,
  F1 = naive_bayes_f1
)
naive_bayes_result

#Linear SVM Model
pkg_available <- requireNamespace("LiblineaR", quietly = TRUE)
print(pkg_available)


feature_frequency <- Matrix::colSums(dtm_train)

top_features <- names(
  sort(feature_frequency, decreasing = TRUE)
)[1:3000]


dtm_train_svm <-dtm_train[, top_features]
dtm_test_svm <- dtm_test[, top_features]

dim(dtm_train_svm)
dim(dtm_test_svm)

#Train Linear SVM
set.seed(123)
svm_model <- LiblineaR::LiblineaR(
  data = dtm_train_svm,
  target = y_train,
  type = 1,
  cost = 1,
  bias = TRUE
)

class(svm_model)

svm_pred <- predict(
  svm_model,
  dtm_test_svm
)

head(svm_pred)
class(svm_pred)
dim(svm_pred)
str(svm_pred)

svm_pred_class <- svm_pred$predictions
head(svm_pred_class)

svm_cm <- table(
  Actual = y_test,
  Predicted = svm_pred_class
)
svm_cm

svm_accuracy <-sum(diag(svm_cm) / sum(svm_cm))
svm_precision <- 4351 / (4351 + 715)
svm_recall <- 4351 / (4351 +626)

svm_f1 <- 2* (
  svm_precision * svm_recall
) / (
  svm_precision + svm_recall
)

svm_accuracy
svm_precision
svm_recall
svm_f1


svm_result <- data.frame(
  Model = "Linear SVM",
  Accuracy = svm_accuracy,
  Precision = svm_precision,
  Recall = svm_recall,
  F1 = svm_f1
)
svm_result

#Ridge Logistic Regression Model
set.seed(123)

ridge_model <-glmnet::glmnet(
  x = dtm_train,
  y = y_train,
  family = "binomial",
  alpha = 0,
  nlambda =30,
  maxit = 100000
)
class(ridge_model)

ridge_prob <- predict(
  ridge_model,
  newx = dtm_test,
  type = "response",
  s = ridge_model$lambda[15]
)
head(ridge_model$lambda)
length(ridge_model$lambda)
tail(ridge_model$lambda)


ridge_pred <- ifelse(ridge_prob > 0.5, 1, 0)

head(ridge_prob)
length(ridge_pred)

ridge_cm <- table(
  Actual = y_test,
  Predicted = ridge_pred
)
ridge_cm

ridge_accuracy <- sum(diag(ridge_cm) / sum(ridge_cm))
ridge_precision <- 4455 /(4455 + 643)
ridge_recall <- 4455 /(4455 + 522)

ridge_f1 <- 2 *(
  ridge_precision * ridge_recall
) / (
  ridge_precision + ridge_recall
)

ridge_accuracy
ridge_precision
ridge_recall
ridge_f1

ridge_result<- data.frame(
  Model = "Ridge classifier",
  Accuracy = ridge_accuracy,
  Precision = ridge_precision,
  Recall = ridge_recall,
  F1 = ridge_f1
)
ridge_result

#Model performance comparison
model_comparison <- dplyr::bind_rows(
  logistic_result,
  naive_bayes_result,
  svm_result,
  ridge_result
)
model_comparison

write.csv(
  model_comparison,
  "model_comparison.csv",
  row.names = FALSE
)



model_comparison_long <- model_comparison %>%
  pivot_longer(
    cols = c(Accuracy, Precision, Recall, F1),
    names_to = "Metric",
    values_to = "Score"
  )

ggplot(model_comparison_long,
       aes(x = Model, y = Score, fill = Metric)) +
  geom_col(position = "dodge") +
  labs(
    title = "Machine Learning Model Performance Comparison",
    x = "Machine Learning Models",
    y = "Performance Score",
    fill = "Evaluation Metric "
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 20, hjust = 1)
  )

#Confusiin matrix visualisation
logistic_cm_df <- as.data.frame(logistic_cm)

ggplot(logistic_cm_df,
       aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = 'white') +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = "Logistic Regression Confusion Matrix",
    x = "Predicted Sentiment",
    y = "Actual Sentiment",
    fill = "Count"
  ) +
  theme_minimal()


naive_bayes_plot <- as.data.frame(
  as.table(naive_bayes_cm)
)
names(naive_bayes_plot) <- c(
  "Actual", "Predicted", "Count"
)


ggplot(naive_bayes_plot,
       aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = "Naive Bayes Confusion Matrix",
    x = "Predicted Sentiment",
    y = "Actual Sentiment",
    fill = "Count"
  ) +
  theme_minimal()


svm_plot <-as.data.frame(
  as.table(svm_cm)
)

names(svm_plot) <- c(
  "Actual", "Predicted", "Count"
)


ggplot(svm_plot,
       aes(x = Predicted, y = Actual, fill = Count))+
  geom_tile(color = "white") +
  geom_text(aes(label = Count), size = 6) +
  scale_fill_gradient(low = "white", high ="steelblue") +
  labs(
    title = "Linear SVM Confusion Matrix",
    x = "Predicted Sentiment",
    y = "Actual Sentiment",
    fill = "Count"
  ) +
  theme_minimal()



ridge_plot <- as.data.frame(
  as.table(ridge_cm)
)

names(ridge_plot) <- c(
  "Actual", "Predicted", "Count"
)
head(ridge_plot)

ggplot(ridge_plot,
       aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = "Ridge Classifier Confusion Matrix",
    x = "Predicted Sentiment",
    y = " Actual Sentiment",
    fill = "Count"
  ) +
  theme_minimal()


#ROC Curve and AUC Analysis
logistic_prob <- as.numeric(
  predict(
    logistic_model,
    newx = dtm_test,
    type = "response",
    s = logistic_model$lambda[
      round(length(logistic_model$lambda) / 2)
    ]
  )
)
head(logistic_prob)


logistic_roc <- pROC::roc(
  response = y_test,
  predictor = logistic_prob
)

logistic_auc <- pROC::auc(logistic_roc)
logistic_auc

plot(
  logistic_roc,
  main = "Logistic Regression ROC Curve",
  col = "blue",
  lwd = 2
)

abline(a = 0, b = 1, lty = 2, col = "red")

class(naive_bayes_result)
head(naive_bayes_result)
ls(pattern = "naive")
ls()
ls(pattern = "model")
str(naive_bayes_pred)
ls(pattern = "bayes|nb")


naive_bayes_score<- as.numeric(as.matrix(positive_score - negative_score))
naive_bayes_prob <- plogis(naive_bayes_score)

naive_bayes_roc <- pROC::roc(
  response = y_test,
  predictor = naive_bayes_prob,
  levels = c(0, 1),
  direction ="<"
)
naive_bayes_auc <- pROC::auc(naive_bayes_roc)
naive_bayes_auc

plot(
  naive_bayes_roc,
  main = "Naive Bayes ROC Curve",
  col ="darkgreen",
  lwd = 2
)
abline(a = 0, b = 1, lty = 2, col = "red")


svm_score_obj <- predict(
  svm_model,
  dtm_test_svm,
  decisionValues = TRUE
)
str(svm_score_obj)


svm_score <- svm_score_obj$decisionValues[, "1"]

svm_roc <-pROC::roc(
  y_test,
  svm_score_obj$decisionValues[, "0"]
)

svm_auc <- pROC::auc(svm_roc)
svm_auc

head(svm_score_obj$decisionValues)
colnames(svm_score_obj$decisionValues)

plot(
  svm_roc,
  main = "SVM ROC Curve",
  col = "purple",
  lwd = 2
)
abline(a =0, b = 1, lty =2, col = "red")

ridge_roc <- pROC::roc(
  response = y_test,
  predictor = as.numeric(ridge_prob),
  levels = c(0, 1)
)
ridge_auc <- pROC::auc(ridge_roc)
ridge_auc

plot(
  ridge_roc,
  main = "Ridge Classifier ROC Curve",
  col = "red",
  lwd = 2
)
abline(a =0, b =1, lty = 2, col = "blue")

#Combined ROC curve comparison
plot(
  logistic_roc,
  main = "ROC Curve Comparison of All Models",
  col = "blue",
  lwd = 2
)

plot(naive_bayes_roc, add = TRUE, col = "darkgreen", lwd = 2)
plot(svm_roc, add = TRUE, col = "purple", lwd = 2)
plot(ridge_roc, add = TRUE, col = "red", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "grey")

legend(
  "bottomright",
  legend = c(
    "Logistic Regression (AUC = 0.8594)",
    "Naive Bayes (AUC = 0.9232)",
    "SVM (AUC =0.9340)",
    "Ridge Classifier (AUC = 0.9477)"
  ),
  col =c("blue", "darkgreen", "purple", "red"),
  lwd = 2,
  cex = 0.7
)

#Zoomed ROC curve comparison
plot(
  logistic_roc,
  legacy.axes = TRUE,
  xlim = c(1, 0.8),
  ylim = c(0.8, 1),
  main = "Zoomed ROC Curve",
  col = "blue",
  lwd = 2,
  xlab = "False Positive Rate",
  ylab = "True Positive Rate",
)

plot(naive_bayes_roc, add = TRUE, col = "darkgreen", lwd = 2)
plot(svm_roc, add = TRUE, col = "purple", lwd = 2)
plot(ridge_roc, add = TRUE, col = "red", lwd =2)

legend(
  "bottomright",
  legend = c("Logistic Regression", "Naive Bayes", "SVM", "Ridge Classifier"),
  col = c("blue", "darkgreen", "purple", "red"),
  lwd = 2,
  cex = 0.7
)
grid()


#LIME
class(ridge_model)

ridge_classifier<-list(
  model = ridge_model,
  vectorizer = vectorizer
)
class(ridge_classifier) <- "ridge_classifier"
  
model_type.ridge_classifier <- function(x, ...) {
  "classification"
}
  
predict_model.ridge_classifier <- function(x, newdata, ...) {
  tokens <- text2vec::word_tokenizer(newdata)
  tokens <- lapply(tokens, function(words) {
    words[!words %in% tidytext::stop_words$word]
  })
  it <- text2vec::itoken(
    tokens,
    progressbar = FALSE
  )
  dtm <- text2vec::create_dtm(
    it,
    x$vectorizer
  )
  probabilities <- as.numeric(
    predict(
      x$model,
      newx = dtm,
      type = "response",
      s = x$model$lambda[15]
    )
  )
  data.frame(
    negative = 1 - probabilities,
    positive = probabilities
  )
}
  
explainer <- lime::lime(
  x = train_data$review,
  model = ridge_classifier,
  feature_type = "text"
)

sample_review <- test_data$review[1]

explanation <- lime::explain(
  x = sample_review,
  explainer = explainer,
  n_labels = 1,
  n_features = 10,
  n_permutations = 500
)

plot_features(explanation)

#SHAP
ridge_coef <- as.matrix(
  coef(ridge_model, s = ridge_model$lambda[15])
)

feature_names <- colnames(dtm_train)

ridge_beta <- as.numeric(
  ridge_coef[feature_names, 1]
)

names(ridge_beta) <- feature_names

background_mean <- Matrix::colMeans(dtm_train)

sample_dtm <- dtm_test[1, drop = FALSE]

sample_values <- as.numeric(as.matrix(sample_dtm))

shap_contributions <- (
  sample_values - background_mean
) *ridge_beta

shap_results <- data.frame(
  Feature = colnames(dtm_train),
  SHAP_Value = as.numeric(shap_contributions)
)

shap_results <- shap_results[
  order(abs(shap_results$SHAP_Value), decreasing = TRUE),
]

head(shap_results, 10)

ggplot(shap_results[1:10, ],
       aes(x = reorder(Feature, abs(SHAP_Value)),
           y=abs(SHAP_Value))) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 10 Features by SHAP Importance",
    x = "Features",
    y = "Absolute SHAP Value"
  ) +
  theme_minimal()
