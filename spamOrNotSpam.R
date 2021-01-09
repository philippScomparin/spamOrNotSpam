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
library(caret)


##################### LOAD DATA ##################### 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
emails <- read.csv('emails.csv')


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
corpus <- tm_map(corpus, content_transformer(tolower))
inspect(corpus[[1]])
corpus <- tm_map(corpus, content_transformer(removeEmailAddress))
inspect(corpus[[4]])
corpus <- tm_map(corpus, content_transformer(removeNumeration))
corpus <- tm_map(corpus,removeWords,c(stopwords(),"re", "ect", "hou", "e", "mail", "kaminski", "hou", "cc", "subject", "vince", "j", "enron", "http"))
corpus <- tm_map(corpus,removePunctuation)
# corpus.ngrams = tm_map(corpus.ngrams, removeURLs)
corpus = tm_map(corpus,removeNumbers)


##################### UNIGRAM ##################### 

tdm = TermDocumentMatrix(corpus)
tdm
inspect(tdm)
tdm.small <- removeSparseTerms(tdm, sparse = 0.9)
tdm.small
freq <- rowSums(as.matrix(tdm.small))
head(freq,5)
freq <- sort(rowSums(as.matrix(tdm.small)), decreasing = T)
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
tdm.bigram <- TermDocumentMatrix(corpus,
                                control = list (tokenize = BigramTokenizer, stripWhitespace = T))
tdm.bigram
tdm.bigram.small <- removeSparseTerms(tdm.bigram, 0.99)
inspect(tdm.bigram.small)
freq <- sort(rowSums(as.matrix(tdm.bigram.small)),decreasing = TRUE)
freq.df <- data.frame(word=names(freq), freq=freq)
head(freq.df, 20)
wordcloud(freq.df$word,freq.df$freq,max.words=100,random.order = F, colors=pal)
ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Bigram") + ylab("Frequency") +
  ggtitle("Most Frequent Bigrams")

##################### TRIGRAM ##################### 

TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
tdm.trigram = TermDocumentMatrix(corpus,
                                 control = list(tokenize = TrigramTokenizer, stripWhitespace = T))
tdm.trigram
tdm.trigram.small <- removeSparseTerms(tdm.trigram, 0.999)
inspect(tdm.trigram.small)
freq <- sort(rowSums(as.matrix(tdm.trigram.small)),decreasing = TRUE)
freq.df <- data.frame(word=names(freq), freq=freq)
head(freq.df, 20)
wordcloud(freq.df$word,freq.df$freq,max.words=100,random.order = F, colors=pal)
ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Trigram") + ylab("Frequency") +
  ggtitle("Most Frequent Trigrams")

##################### SENTIMENT ANALYSIS ##################### 

corpus.df <- as.data.frame(corpus)
sentiment <- sentiment_by(corpus.df$text)
summary(sentiment$ave_sentiment)
qplot(sentiment$ave_sentiment, geom="histogram",binwidth=0.1,main="Review Sentiment Histogram")
sentiments <- sentiment$ave_sentiment

##################### BUILD MODEL TO CLASSIFY E-MAILS AS SPAM OR NOT SPAM ##################### 
dtm = DocumentTermMatrix(corpus)
dtm
dtm.small <- removeSparseTerms(dtm, sparse = 0.9)
dtm.small
xMatrix <- as.matrix(dtm.small)
spam <- emails$spam
wholeData <- as.data.frame(cbind(spam,xMatrix, sentiments))

splits <- sample(1:length(spam), size=floor(.75*length(spam)), replace=FALSE) 
trainData <- wholeData[splits,]
testData <- wholeData[-splits,]

sv <- svm(spam~., trainData, type="C-classification", kernel="radial", cost=100)
prediction <- predict(sv, testData[,-1])
table(prediction , True=testData$spam)
confusionMatrix(table(prediction, True=testData$spam))

