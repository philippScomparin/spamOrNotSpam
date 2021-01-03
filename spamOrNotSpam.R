

##################### IMPORT LIBRARIES ##################### 

library(tm)
library(wordcloud)
library(SnowballC)
library(RWeka)
library(ggplot2)

##################### LOAD DATA ##################### 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
emails <- read.csv('emails.csv')

##################### Explore DATA ##################### 

head(emails$text)
length(emails$text)


##################### UNIGRAM ##################### 

corpus <- VCorpus(VectorSource(emails$text))
tdm = TermDocumentMatrix(corpus, control=list(removePunctuation = T, removeNumbers = T, stopwords = c(stopwords("en"), "subject"), stemming = T))
removeSparseTerms(tdm, sparse = 0.5)
freq = rowSums(as.matrix(tdm))
tail(freq,10)
pal = brewer.pal(8, "Blues")
pal = pal[-(1:3)]
set.seed(1234)
freq = sort(rowSums(as.matrix(tdm)), decreasing = T)
word.cloud = wordcloud(words=names(freq), freq=freq, min.freq=400,
                       random.order=F, colors=pal)
inspect(tdm)



##################### BIGRAM ##################### 

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
tdm.bigram = TermDocumentMatrix(corpus,
                                control = list (tokenize = BigramTokenizer, removePunctuation = T, removeNumbers = T, stopwords = c(stopwords("en"), "subject"), stemming = T))
freq = sort(rowSums(as.matrix(tdm.bigram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)
wordcloud(freq.df$word,freq.df$freq,max.words=100,random.order = F, colors=pal)

ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Bigrams") + ylab("Frequency") +
  ggtitle("Most frequent bigrams")

