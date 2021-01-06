##################### IMPORT LIBRARIES ##################### 

library(tm)
library(wordcloud)
library(SnowballC)
library(RWeka)
library(ggplot2)
library(reshape2)

##################### LOAD DATA ##################### 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
emails <- read.csv('emails.csv')

##################### Explore DATA ##################### 

head(emails$text)
length(emails$text)

##################### HELPER FUNCTIONS ##################### 

removeEmailAddress <- function(x) {
  gsub(pattern = "\\w*[[:blank:]][[:punct:]][[:blank:]]\\w*[[:blank:]][[:punct:]][[:blank:]]com[[:blank:]]", replacement = "", x)
}

removeNumeration <- function(x) {
  gsub(pattern = "[[:digit:]][[:blank:]]th", replacement = "", x)
}

removeURLs <- function(x){
  
  gsub(pattern= "http[[:blank:]]www\\w*com", replacement = "", x)
  
}

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
corpus.ngrams = tm_map(corpus.ngrams, removeURLs)
corpus.ngrams = tm_map(corpus.ngrams,removeNumbers)


##################### UNIGRAM ##################### 

tdm = TermDocumentMatrix(corpus.ngrams, control=list(stripWhitespace= T))
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




