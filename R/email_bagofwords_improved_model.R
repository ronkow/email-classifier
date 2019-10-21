# EMAIL CLASSIFIER USING THE BAG OF WORDS: IMPROVED MODEL

# (1) USE train.csv and test.csv CREATED FOR PREVIOUS MODEL

setwd("d://email-classifier/data")
train_hamspam <- read.csv("train.csv", header=TRUE)
test_hamspam <- read.csv("test.csv", header=TRUE)

# ------------------------------------------------------------------
# (2) CLEAN TRAINING DATA AND CREATE THE DOCUMENT TERM MATRIX

# install.packages("tm")
# install.packages("dplyr")
# install.packages("SnowballC")

library(tm)
library(dplyr)
library(SnowballC)

# Function to clean a corpus
clean_corpus <- function(corpus) {
    corpus <- corpus %>% 
    tm_map(removeWords, stopwords(kind="en")) %>% 
    tm_map(removeNumbers) %>%
    tm_map(removePunctuation) %>%
    tm_map(stripWhitespace)
}

# Create corpus and clean data
train_corpus <- VCorpus(VectorSource(train_hamspam$text))
train_corpus_clean <- clean_corpus(train_corpus)

train_dtm_tf <- DocumentTermMatrix(train_corpus_clean, control=list(weighting=weightTf))
train_dtm_tf_reduced <- removeSparseTerms(train_dtm_tf, 0.98)


# Function to clean a corpus
clean_corpus_words <- function(corpus) {
    corpus <- corpus %>% 
    tm_map(removeWords, c("allen","ami","aimee","anita","bob","brian","beaumont","brenda","bill",
                          "clynes","cotten","cotton","charlie","carlos","chokshi","cheryl","cynthia",
                          "david","duke","darren","daren","don","donald","dave","donna","devon",
                          "ena","edward","farmer","fred","graves","gary","george","howard","hanks",
                          "james","jackie","julie","jones","john","jim","jeff",
                          "katy","ken","katherine","karen","lloyd","lisa","lee","lauri","luong",
                          "mike","melissa","mary","megan","michael","meyers","mark","parker","pat",
                          "rita","rodriguez","robert","ray","richard",
                          "smith","steve","stone","susan","sitara","stacey","sherlyn","schumack","scott","stephanie",
                          "tenaska","tom","thomas","taylor","tess","vance","victor",
                          "wynne","weissman","william")) %>% 
    tm_map(removeWords, c("adobe","aep","albrecht","canada","cec","clem","computron","carthage","com",
                          "doc","eastrans","exxon","ees","entex","ect","enron","enronxgate","equistar",
                          "gco","gcs","href","hplo","hsc","houston","hou","hplc","http","hpl","hotmail","hplno",
                          "iferc","lst","lsk","lannou","microsoft","mmbtu","methanol","nom","noms",
                          "pec","papayoti","Subject","subject","teco","texas","txu","vlt","www","xls","yahoo")) %>% 
    tm_map(stemDocument, language="english")
}

train_corpus_clean_final <- clean_corpus_words(train_corpus_clean)

train_dtm_tf <- DocumentTermMatrix(train_corpus_clean_final, control=list(weighting=weightTf))
train_dtm_tf_reduced <- removeSparseTerms(train_dtm_tf, 0.98)

inspect(train_dtm_tf_reduced)
Terms(train_dtm_tf_reduced)


# ------------------------------------------------------------------
# (3) SELECT OPTIMUM BAG OF WORDS

# Create a column of labels from the training dataset
train_label <- as.character(train_hamspam$label)

# Convert DTM to data frame
train_df_tf <- as.data.frame(as.matrix(train_dtm_tf_reduced))

# Create the training data frame
train_tf <- as.data.frame(cbind(train_df_tf, train_label))

# Change the name of the last column
colnames(train_tf)[ncol(train_tf)] <- "label"

# Create separate data frames for the BoW with only ham or spam records
train_ham <- train_tf[train_tf$label=="ham",]
train_spam <- train_tf[train_tf$label=="spam",]

# Sum the occurrences of each of the 475 words
train_ham_wordcount <- colSums(train_ham[,c(1:ncol(train_df_tf))])
train_spam_wordcount <- colSums(train_spam[,c(1:ncol(train_df_tf))])

# Create a matrix of word counts and the proportion: counts in ham / counts in spam
proportion <- train_ham_wordcount/train_spam_wordcount
train_wordcount <- cbind(train_ham_wordcount, train_spam_wordcount, proportion)
colnames(train_wordcount) <- c("ham", "spam", "ham/spam")

# Choose words
train_wordcount <- data.frame(train_wordcount)
train_wordcount_reduced <- train_wordcount[(train_wordcount$proportion > 17.0 | train_wordcount$proportion < 0.4),]

nrow(train_wordcount_reduced)

bagofwords <- rownames(train_wordcount_reduced)


# ------------------------------------------------------------------
# (4) TRAIN THE MODEL USING LOGISTIC REGRESSION

# install.packages("caret")

library(caret)

train_tf_bow <- train_df_tf[,bagofwords]

# Train the tf model
set.seed(123)
model_tf <- train(train_tf_bow, train_label, method="glm", trControl=trainControl(method="cv", number=5))

# Training results
model_tf

# ------------------------------------------------------------------
# (4) TEST THE MODEL

# cleaning
test_corpus <- VCorpus(VectorSource(test_hamspam$text))
test_corpus_clean <- clean_corpus(test_corpus)
test_corpus_clean_final <- clean_corpus_words(test_corpus_clean)

# DTM with all words
test_dtm_tf <- DocumentTermMatrix(test_corpus_clean_final, control=list(weighting=weightTf))

# Convert DTM to data frame
test_df_tf <- as.data.frame(as.matrix(test_dtm_tf))

# Create data frame with only the 75 columns 
test_df_tf_bow <- test_df_tf[,bagofwords]  

test_tf <- predict(model_tf, test_df_tf_bow)

# ------------------------------------------------------------------
# (5) ANALYSE THE RESULTS

confusionMatrix(data=test_tf, 
                reference=test_hamspam$label, 
                positive="spam", 
                dnn=c("Prediction", "Actual"))

