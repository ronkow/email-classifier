# EMAIL CLASSIFIER USING THE BAG OF WORDS MODEL

# For step (1), download the raw data from: http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz

# (1) PREPARE THE RAW DATA
# You can skip step (1) and proceed to step (2) using ham.csv and spam.csv provided here.

# Spam text files stored in d://email_classifier/data/spam/
setwd("d://email_classifier/data/spam")
mydir <- getwd()

# Concatenate all text files
filespam <- list.files(mydir, full.names = TRUE, pattern = "*.txt")
textspam1 <- lapply(filespam, readLines)
textspam2 <- sapply(textspam1, paste, collapse = " ")

# Uncomment if you want to see the output
# head(filespam, n = 2)
# head(textspam1, n = 2)
# head(textspam2, n = 2)

# Remove duplicates
textspam_df <- data.frame(textspam2)
textspam_df2 <- unique(textspam_df)

# Check number of spam before and after removing duplicates 
nrow(textspam_df)
nrow(textspam_df2)

# Create dataframe with a new column for the labels 
spam_df <- data.frame(text = textspam_df2, label = "spam")

# CSV file allows viewing of data in Excel
write.csv(spam_df, "d://email_classifier/data/spam.csv")

# Same process for ham emails

setwd("d://email_classifier/data/ham")
mydir <- getwd()

fileham <- list.files(mydir, full.names = TRUE, pattern = "*.txt")
textham1 <- lapply(fileham, readLines)
textham2 <- sapply(textham1, paste, collapse = " ")

# head(fileham, n = 3)
# head(textham1, n = 3)
# head(textham2, n = 3)

textham_df <- data.frame(textham2)
textham_df2 <- unique(textham_df)

nrow(textham_df)
nrow(textham_df2)

ham_df <- data.frame(text = textham_df2, label = "ham")
write.csv(ham_df, "d://email_classifier/data/ham.csv")

# ------------------------------------------------------------------
# (2) SHUFFLE ROWS AND SPLIT DATA INTO TRAINING AND TEST DATASETS

# If you wish to skip Step 1 and use ham.csv and spam.csv prepared in Step 1 
# setwd("d://email-classifier/data")
# ham_df <- read.csv("ham.csv", header = TRUE)
# spam_df <- read.csv("spam.csv", header = TRUE)

# Combine ham and spam data frames into one data frame
colnames(ham_df) <- c("text","label")
colnames(spam_df) <- c("text","label")
hamspam_df <- rbind(ham_df, spam_df)

# Shuffle the row numbers (1 to 4895)
set.seed(123)
numofrows <-  nrow(hamspam_df)
randomindex <- sample(numofrows, numofrows, replace = FALSE)

# Create a new data frame with shuffled rows
hamspam_df2 <- hamspam_df[randomindex,]

# Re-number the rows in the new data frame
rownames(hamspam_df2) <- c(1:numofrows)

head(randomindex, n = 10)
head(hamspam_df2, n = 3)
head(rownames(hamspam_df2), n = 10)

# 80%/20% training/testing split = 3916/979 
train_hamspam <- hamspam_df2[1:3916,]
test_hamspam <- hamspam_df2[3917:4895,]

# Write to CSV files
write.csv(train_hamspaum, "d://email_classifier/train_hamspam.csv")
write.csv(test_hamspam, "d://email_classifier/test_hamspam.csv")

# ------------------------------------------------------------------
# (3) CLEAN TRAINING DATA AND CREATE THE DOCUMENT TERM MATRIX

# install.packages("tm")
# install.packages("dplyr")

library(tm)
library(dplyr)

# Function to clean a corpus
clean_corpus <- function(corpus){
  corpus <- corpus %>% 
    tm_map(removeNumbers) %>%
    tm_map(removePunctuation) %>%
    tm_map(stripWhitespace) %>%
    tm_map(removeWords, stopwords(kind = "SMART")) 
}

# Create corpus and clean data
train_corpus <- VCorpus(VectorSource(train_hamspam$text))
train_corpus2 <- clean_corpus(train_corpus)

# Create the complete DTM and reduced DTM (term frequency)
train_dtm_tf <- DocumentTermMatrix(train_corpus2, control = list(weighting = weightTf))
train_dtm_tf2 <- removeSparseTerms(train_dtm_tf, 0.96)

inspect(train_dtm_tf)
inspect(train_dtm_tf2)

# List all the words in the Bag of Words (BoW)
Terms(train_dtm_tf2)

# Remove names, abbreviations, acronyms from the BoW
train_corpus3 <- train_corpus2 %>% 
  tm_map(removeWords, c("ami","bob","daren","david","don","ect","ena",
                        "enron","farmer","gary","hou","houston","hpl","http",
                        "mary","melissa","michael","mmbtu","nom","noms","pat",
                        "robert","sitara","smith","taylor","teco","texas","vance",
                        "www","xls","subject","Subject"))

# Create the final reduced DTM (tf)
train_dtm_tf <- DocumentTermMatrix(train_corpus3, control = list(weighting = weightTf))
train_dtm_tf2 <- removeSparseTerms(train_dtm_tf, 0.96)

# List all the words in the final BoW
Terms(train_dtm_tf2)

# Create the final DTM (binary)
train_dtm_bin <- DocumentTermMatrix(train_corpus3, control = list(weighting = weightBin))
train_dtm_bin2 <- removeSparseTerms(train_dtm_bin, 0.96)

# Create the final DTM (tfidf)
train_dtm_tfidf <- DocumentTermMatrix(train_corpus3, control = list(weighting = weightTfIdf))
train_dtm_tfidf2 <- removeSparseTerms(train_dtm_tfidf, 0.96)

inspect(train_dtm_tf2)
inspect(train_dtm_bin2)
inspect(train_dtm_tfidf2)

# ------------------------------------------------------------------
# (4) CREATE THE DATA FRAME FOR TRAINING

# Create a column of labels from the training dataset
train_label <- as.character(train_hamspam$label)

# Convert DTM (tf) to data frame
train_df_tf <- as.data.frame(as.matrix(train_dtm_tf2))

# Create the training data frame (tf)
train_tf <- as.data.frame(cbind(train_df_tf, train_label))

# Change the name of the last column
colnames(train_tf)[ncol(train_tf)] <- "label"

# Create the training data frame (binary) 
train_df_bin <- as.data.frame(as.matrix(train_dtm_bin2))
train_bin <- as.data.frame(cbind(train_df_bin, train_label))
colnames(train_bin)[ncol(train_bin)] <- "label"

# Create the training data frame (tfidf)
train_df_tfidf <- as.data.frame(as.matrix(train_dtm_tfidf2))
train_tfidf <- as.data.frame(cbind(train_df_tfidf, train_label))
colnames(train_tfidf)[ncol(train_tfidf)] <- "label"

# ------------------------------------------------------------------
# (5) TRAIN THE MODEL USING LOGISTIC REGRESSION

# install.packages("caret")
library(caret)

# Train the tf model
starttime <- Sys.time()
set.seed(123)
model_tf <- train(label~ . , data = train_tf, method = "glm", metric = "Accuracy")
endtime <- Sys.time()
traintime_tf <- endtime - starttime

# Train the binary model
starttime <- Sys.time()
set.seed(123)
model_bin <- train(label~ . , data = train_bin, method = "glm", metric = "Accuracy")
endtime <- Sys.time()
traintime_bin <- endtime - starttime

# Train the tfidf model
starttime <- Sys.time()
set.seed(123)
model_tfidf <- train(label~ . , data = train_tfidf, method = "glm", metric = "Accuracy")
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

# Create a vector containing the 105 words in the BoW
bagofwords <- colnames(train_df_tf)

# Basic cleaning
test_corpus <- VCorpus(VectorSource(test_hamspam$text))
test_corpus2 <- clean_corpus(test_corpus)

# DTM with all words
test_dtm_tf_allwords <- DocumentTermMatrix(test_corpus2, control = list(weighting = weightTf))
test_dtm_bin_allwords <- DocumentTermMatrix(test_corpus2, control = list(weighting = weightBin))
test_dtm_tfidf_allwords <- DocumentTermMatrix(test_corpus2, control = list(weighting = weightTfIdf))

# Convert DTM to data frame
test_df_tf_allwords <- as.data.frame(as.matrix(test_dtm_tf_allwords))
test_df_bin_allwords <- as.data.frame(as.matrix(test_dtm_bin_allwords))
test_df_tfidf_allwords <- as.data.frame(as.matrix(test_dtm_tfidf_allwords))

# Create data frame with only the 105 columns 
test_df_tf_bagofwords <- test_df_tf_allwords[,bagofwords]  
test_df_bin_bagofwords <- test_df_bin_allwords[,bagofwords]  
test_df_tfidf_bagofwords <- test_df_tfidf_allwords[,bagofwords]  

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
