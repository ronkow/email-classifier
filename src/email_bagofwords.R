# EMAIL CLASSIFIER USING THE BAG OF WORDS MODEL

# For step (1), download the raw data (ham and spam text files) from: http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz

# (1) PREPARE THE RAW DATA
# If you do not wish to download and prepare the raw data, you can skip step (1) 
# and proceed to step (2) using ham.csv and spam.csv provided in email-classifier/data

# Ham text files downloaded and stored in d://email-classifier/enron1/ham
setwd("d://email-classifier/enron1/ham")

# Concatenate all text files
files_ham <- list.files()
lines_ham <- lapply(files_ham, readLines)
text_ham <- sapply(lines_ham, paste, collapse=" ")

# Remove duplicates
text_ham_df <- data.frame(text_ham)
text_ham_df_unique <- unique(text_ham_df)

# Check number of spam before and after removing duplicates 
# 3672 reduced to 3432 records
nrow(text_ham_df)
nrow(text_ham_df_unique)

# Create data frame with a new column for the labels
ham_df <- data.frame(text_ham_df_unique, label="ham")

# Write to CSV file so that we can view the data in a spreadsheet
write.csv(ham_df, file="d://email-classifier/data/ham.csv", row.names=FALSE)

# Same process for spam emails
# Spam text files downloaded and stored in d://email-classifier/enron1/spam
setwd("d://email-classifier/enron1/spam")

# Concatenate all text files
files_spam <- list.files()
lines_spam <- lapply(files_spam, readLines)
text_spam <- sapply(lines_spam, paste, collapse=" ")

# Remove duplicates
text_spam_df <- data.frame(text_spam)
text_spam_df_unique <- unique(text_spam_df)

# Check number of spam before and after removing duplicates 
# 1500 reduced to 1463 records
nrow(text_spam_df)
nrow(text_spam_df_unique)

# Create data frame with a new column for the labels 
spam_df <- data.frame(text_spam_df_unique, label="spam")

# Write to CSV file so that we can view the data in a spreadsheet 
write.csv(spam_df, file="d://email-classifier/data/spam.csv", row.names=FALSE)

# ------------------------------------------------------------------
# (2) SHUFFLE ROWS AND SPLIT DATA INTO TRAINING AND TEST DATASETS

# If you wish to skip Step 1 and use ham.csv and spam.csv prepared in Step 1 
# setwd("d://email-classifier/data")
# ham_df <- read.csv("ham.csv", header=TRUE)
# spam_df <- read.csv("spam.csv", header=TRUE)

# If you did Step 1, use the data frames from Step 1
# Change the column names
colnames(ham_df) <- c("text","label")
colnames(spam_df) <- c("text","label")

# Combine the ham and spam data frames
hamspam_df <- rbind(ham_df, spam_df)

# Shuffle the row numbers (1 to 4895)
set.seed(123)
numofrows <-  nrow(hamspam_df)
randomindex <- sample(numofrows, numofrows, replace = FALSE)

# Create a new data frame with shuffled rows
hamspam_df_random <- hamspam_df[randomindex,]

# Re-number the rows in the new data frame
rownames(hamspam_df_random) <- c(1:numofrows)

# 80%/20% training/testing split = 3916/979 
train_hamspam <- hamspam_df_random[1:3916,]
test_hamspam <- hamspam_df_random[3917:4895,]

write.csv(train_hamspam, file="train.csv", row.names=FALSE)
write.csv(test_hamspam, file="test.csv", row.names=FALSE)

# ------------------------------------------------------------------
# (3) CLEAN TRAINING DATA AND CREATE THE DOCUMENT TERM MATRIX

# install.packages("tm")
# install.packages("dplyr")

library(tm)
library(dplyr)

# Function to clean a corpus
clean_corpus <- function(corpus) {
    corpus <- corpus %>% 
    tm_map(removeNumbers) %>%
    tm_map(removePunctuation) %>%
    tm_map(stripWhitespace) %>%
    tm_map(removeWords, stopwords(kind="en")) 
}

# Create corpus and clean data
train_corpus <- VCorpus(VectorSource(train_hamspam$text))
train_corpus_clean <- clean_corpus(train_corpus)

# Create the initial DTM (term frequency) for inspection of words
train_dtm_tf_initial <- DocumentTermMatrix(train_corpus_clean, control=list(weighting=weightTf))
train_dtm_tf_initial_reduced <- removeSparseTerms(train_dtm_tf_initial, 0.96)

inspect(train_dtm_tf_initial)
inspect(train_dtm_tf_initial_reduced)

# List all the words in the initial Bag of Words (BoW)
Terms(train_dtm_tf_initial_reduced)

# Remove names, abbreviations, acronyms from the BoW
train_corpus_clean_final <- train_corpus_clean %>% 
  tm_map(removeWords, c("ami","bob","com","daren","david","don","ect","ena",
                        "enron","farmer","gary","hou","houston","hpl","http",
                        "mary","melissa","michael","mmbtu","nom","noms","pat",
                        "robert","sitara","smith","taylor","teco","texas","vance",
                        "www","xls","subject","Subject"))

# Create the final reduced DTM (tf)
train_dtm_tf <- DocumentTermMatrix(train_corpus_clean_final, control=list(weighting=weightTf))
train_dtm_tf_reduced <- removeSparseTerms(train_dtm_tf, 0.96)

# List all the words in the final BoW
Terms(train_dtm_tf_reduced)

# Create the final DTM (binary)
train_dtm_bin <- DocumentTermMatrix(train_corpus_clean_final, control=list(weighting=weightBin))
train_dtm_bin_reduced <- removeSparseTerms(train_dtm_bin, 0.96)

# Create the final DTM (tfidf)
train_dtm_tfidf <- DocumentTermMatrix(train_corpus_clean_final, control=list(weighting=weightTfIdf))
train_dtm_tfidf_reduced <- removeSparseTerms(train_dtm_tfidf, 0.96)

inspect(train_dtm_tf_reduced)
inspect(train_dtm_bin_reduced)
inspect(train_dtm_tfidf_reduced)

# ------------------------------------------------------------------
# (4) CREATE THE DATA FRAME FOR TRAINING

# Create a column of labels from the training dataset
train_label <- as.character(train_hamspam$label)

# Convert DTM (tf) to data frame
train_df_tf <- as.data.frame(as.matrix(train_dtm_tf_reduced))

# Create the training data frame (tf)
train_tf <- as.data.frame(cbind(train_df_tf, train_label))

# Change the name of the last column
colnames(train_tf)[ncol(train_tf)] <- "label"

# Create the training data frame (binary) 
train_df_bin <- as.data.frame(as.matrix(train_dtm_bin_reduced))
train_bin <- as.data.frame(cbind(train_df_bin, train_label))
colnames(train_bin)[ncol(train_bin)] <- "label"

# Create the training data frame (tfidf)
train_df_tfidf <- as.data.frame(as.matrix(train_dtm_tfidf_reduced))
train_tfidf <- as.data.frame(cbind(train_df_tfidf, train_label))
colnames(train_tfidf)[ncol(train_tfidf)] <- "label"

# ------------------------------------------------------------------
# (5) TRAIN THE MODEL USING LOGISTIC REGRESSION

# install.packages("caret")
library(caret)

# Train the tf model
starttime <- Sys.time()
set.seed(123)
model_tf <- train(label~ . , data = train_tf, method = "glm")
endtime <- Sys.time()
traintime_tf <- endtime - starttime

# Train the binary model
starttime <- Sys.time()
set.seed(123)
model_bin <- train(label~ . , data = train_bin, method = "glm")
endtime <- Sys.time()
traintime_bin <- endtime - starttime

# Train the tfidf model
starttime <- Sys.time()
set.seed(123)
model_tfidf <- train(label~ . , data = train_tfidf, method = "glm")
endtime <- Sys.time()
traintime_tfidf <- endtime - starttime

# Training results
model_tf
model_bin
model_tfidf

traintime_tf
traintime_bin
traintime_tfidf

# ------------------------------------------------------------------
# Compare results
train_results <- resamples(list(tf = model_tf, 
                                binary = model_bin,
                                tfidf = model_tfidf))
summary(train_results)

# ------------------------------------------------------------------
# (6) PREPARE THE TEST DATASET

# Create a vector containing the set of words in the BoW
bagofwords <- colnames(train_df_tf)

# Basic cleaning
test_corpus <- VCorpus(VectorSource(test_hamspam$text))
test_corpus_clean <- clean_corpus(test_corpus)

# DTM with all words
test_dtm_tf <- DocumentTermMatrix(test_corpus_clean, control=list(weighting=weightTf))
test_dtm_bin <- DocumentTermMatrix(test_corpus_clean, control=list(weighting=weightBin))
test_dtm_tfidf <- DocumentTermMatrix(test_corpus_clean, control=list(weighting=weightTfIdf))

# Convert DTM to data frame
test_df_tf <- as.data.frame(as.matrix(test_dtm_tf))
test_df_bin <- as.data.frame(as.matrix(test_dtm_bin))
test_df_tfidf <- as.data.frame(as.matrix(test_dtm_tfidf))

# Create data frame with only the columns with the BoW 
test_df_tf_bagofwords <- test_df_tf[,bagofwords]  
test_df_bin_bagofwords <- test_df_bin[,bagofwords]  
test_df_tfidf_bagofwords <- test_df_tfidf[,bagofwords]  

# ------------------------------------------------------------------
# (7) TEST THE MODELS

test_tf <- predict(model_tf, test_df_tf_bagofwords)
test_bin <- predict(model_bin, test_df_bin_bagofwords)
test_tfidf <- predict(model_tfidf, test_df_tfidf_bagofwords)

# ------------------------------------------------------------------
# (8) ANALYSE THE RESULTS

confusionMatrix(data = test_tf, 
                reference = test_hamspam$label, 
                positive = "spam", 
                dnn = c("Prediction", "Actual"))

confusionMatrix(data = test_bin, 
                reference = test_hamspam$label, 
                positive = "spam", 
                dnn = c("Prediction", "Actual"))

confusionMatrix(data = test_tfidf, 
                reference = test_hamspam$label, 
                positive = "spam", 
                dnn = c("Prediction", "Actual"))

# ------------------------------------------------------------------
# (9) IMPROVING THE MODEL: A SIMPLE ANALYSIS

# Create separate data frames for the BoW with only ham or spam records
train_ham <- train_tf[train_tf$label=="ham",]
train_spam <- train_tf[train_tf$label=="spam",]

# Sum the occurrences of each of the words in the BoW
train_ham_wordcount <- colSums(train_ham[,c(1:ncol(train_df_tf))])
train_spam_wordcount <- colSums(train_spam[,c(1:ncol(train_df_tf))])

# Create a matrix of word counts and the differences and proportions
difference <- train_ham_wordcount - train_spam_wordcount
proportion <- train_ham_wordcount / train_spam_wordcount
train_wordcount <- cbind(train_ham_wordcount, train_spam_wordcount, difference, proportion)
colnames(train_wordcount) <- c("ham", "spam", "ham-spam", "ham/spam")

# List the words 
train_wordcount
