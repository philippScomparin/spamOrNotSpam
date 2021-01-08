##################### IMPORT LIBRARIES ##################### 

library(tm)
library(wordcloud)
library(SnowballC)
library(RWeka)
library(ggplot2)
library(reshape2)
library(sentimentr)
library(qdap)
library(e1071)
library(gmodels)
library(RTextTools)



##################### LOAD DATA ##################### 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
emails <- read.csv('emails.csv')

##################### Explore DATA ##################### 

head(emails$text)
length(emails$text)
table(factor(emails$spam))
table(is.na(emails$spam))

##################### HELPER FUNCTIONS ##################### 

removeEmailAddress <- function(x) {
  gsub(pattern = "\\w*[[:blank:]][[:punct:]][[:blank:]]\\w*[[:blank:]][[:punct:]][[:blank:]]com[[:blank:]]", replacement = "", x)
}

removeNumeration <- function(x) {
  gsub(pattern = "[[:digit:]][[:blank:]]th", replacement = "", x)
}

# removeURLs <- function(x){
#   
#   gsub(pattern= "http[[:blank:]]www\\w*com", replacement = "", x)
#   
# }

pal = brewer.pal(8, "Blues")
pal = pal[-(1:3)]
set.seed(1234)

##################### CLEAN CORPUS ##################### 

corpus <- VCorpus(VectorSource(emails$text))
inspect(corpus[[4]])
corpus_lowercase <- tm_map(corpus, content_transformer(tolower))
inspect(corpus_lowercase[[1]])
corpus_noEmailAddress <- tm_map(corpus_lowercase, content_transformer(removeEmailAddress))
inspect(corpus_noEmailAddress[[4]])
corpus_noNumeration <- tm_map(corpus_noEmailAddress, content_transformer(removeNumeration))
corpus.ngrams = tm_map(corpus_noNumeration,removeWords,c(stopwords(),"re", "ect", "hou", "e", "mail", "kaminski", "hou", "cc", "subject", "vince", "j", "enron", "http"))
corpus.ngrams = tm_map(corpus.ngrams,removePunctuation)
# corpus.ngrams = tm_map(corpus.ngrams, removeURLs)
corpus.ngrams = tm_map(corpus.ngrams,removeNumbers)


##################### UNIGRAM ##################### 

tdm = TermDocumentMatrix(corpus.ngrams)
tdm
tdm.small <- removeSparseTerms(tdm, sparse = 0.9)
tdm.small
freq = rowSums(as.matrix(tdm.small))
head(freq,10)
freq = sort(rowSums(as.matrix(tdm.small)), decreasing = T)
inspect(tdm)
word.cloud = wordcloud(words=names(freq), freq=freq, min.freq=500,
                       random.order=F, colors=pal)

##################### WORD DOCUMENT FREQUENCY GRAPH #####################

tdm.mini <- removeSparseTerms(tdm.small, sparse = 0.8)
matrix.tdm = melt(as.matrix(tdm.mini), value.name = "count")
head(matrix.tdm)

ggplot(matrix.tdm, aes(x = Docs, y = Terms, fill = log10(count))) +
  geom_tile(colour = "black") +
  scale_fill_gradient(high="#ffffff" , low="#000000")+
  ylab("Terms") +
  xlab("E-Mails") +
  theme(panel.background = element_blank()) +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

##################### BIGRAM ##################### 

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tdm.bigram <- TermDocumentMatrix(corpus.ngrams,
                                control = list (tokenize = BigramTokenizer, stripWhitespace = T))
tdm.bigram
tdm.bigram.small <- removeSparseTerms(tdm.bigram, 0.99)
inspect(tdm.bigram.small)
freq = sort(rowSums(as.matrix(tdm.bigram.small)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)
wordcloud(freq.df$word,freq.df$freq,max.words=100,random.order = F, colors=pal)
ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Bigram") + ylab("Frequency") +
  ggtitle("Most Frequent Bigrams")

##################### TRIGRAM ##################### 

TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
tdm.trigram = TermDocumentMatrix(corpus.ngrams,
                                 control = list(tokenize = TrigramTokenizer, stripWhitespace = T))
tdm.trigram
tdm.trigram.small <- removeSparseTerms(tdm.trigram, 0.999)
inspect(tdm.trigram.small)
freq = sort(rowSums(as.matrix(tdm.trigram.small)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)
wordcloud(freq.df$word,freq.df$freq,max.words=100,random.order = F, colors=pal)
ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Trigram") + ylab("Frequency") +
  ggtitle("Most Frequent Trigrams")

##################### SENTIMENT ANALYSIS ##################### 

df = as.data.frame(corpus.ngrams)
sentiment = sentiment_by(df$text)
summary(sentiment$ave_sentiment)
qplot(sentiment$ave_sentiment, geom="histogram",binwidth=0.1,main="Review Sentiment Histogram")
sentiments <- sentiment$ave_sentiment

##################### BUILD MODEL TO CLASSIFY E-MAILS AS SPAM OR NOT SPAM ##################### 

ySplits <- sort(sample(nrow(emails), nrow(emails)*.7))
yTrain <- emails[ySplits,]$spam
yTest <- emails[-ySplits,]$spam
prop.table(table(yTrain))
prop.table(table(yTest))

xTrain <- tdm[1:4009,]
xTest <- tdm[4010:5728,]

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}
train <- apply(xTrain, MARGIN = 2,
               convert_counts)
test <- apply(xTest, MARGIN = 2,
              convert_counts)

classifier <- naiveBayes(train, yTrain)
classifier[2]$tables
testPredict <- predict(classifier, test)

CrossTable(testPredict, yTest,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))

cMatrix <- table(testPredict, testPredict)
confusion_matrix(cMatrix)


##################### BUILD MODEL TO CLASSIFY E-MAILS AS SPAM OR NOT SPAM ##################### 

m <- data.frame(emails$text,emails$spam)
tdmNew <- create_matrix(m$emails.text, language="english", removeNumbers=TRUE,
                        stemWords=TRUE, removeSparseTerms=.998)

container <- create_container(tdmNew, m$emails.spam,
                              trainSize = 1:4009, testSize = 4010:5728, virgin = F)
svm_model <- train_model(container,"SVM")
svm <- classify_model(container, svm_model)


##################### BUILD MODEL TO CLASSIFY E-MAILS AS SPAM OR NOT SPAM ##################### 
dtm = DocumentTermMatrix(corpus.ngrams)
dtm
dtm.small <- removeSparseTerms(dtm, sparse = 0.9)
dtm.small
xMatrix <- as.matrix(dtm.small)
y <- emails$spam
data <- as.data.frame(cbind(y,xMatrix, sentiments))

# split into test and train
train.index <- sample(1:length(y), size=floor(.8*length(y)), replace=FALSE) 
train <- data[train.index,]
test <- data[-train.index,]

# fit the svm and do a simple validation test. Cost parameter should be tuned.
sv <- svm(y~., train, type="C-classification", kernel="linear", cost=1)
table(Pred=predict(sv, test[,-1]) , True=test$y)



