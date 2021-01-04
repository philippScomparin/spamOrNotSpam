

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


##################### HELPER FUNCTION ##################### 

# removeRegex <- function(x) {
#   gsub(pattern = "[A-z]*\\s[@]\\s[A-z]*\\s[.]\\scom", "", x)
# }

# removeRegex <- function(x) {
#   gsub(pattern = "\\w*[[:blank:]][[:punct:]][[:blank:]]\\w*[[:blank:]][[:punct:]][[:blank:]]com", "", x)
# }


removeRegex <- function(x) {
  gsub(pattern = "com", "", x)
}

##################### UNIGRAM ##################### 

corpus <- VCorpus(VectorSource(emails$text))
corpus_lowercase <- tm_map(corpus, content_transformer(tolower))
corpus_noEmail <- tm_map(corpus_lowercase, content_transformer(removeRegex))

length(corpus_lowercase)
length(corpus_noEmail)

tdm = TermDocumentMatrix(corpus_lowercase, control=list(removePunctuation = T, removeNumbers = T, stopwords = c(stopwords(), "subject", "subject re"), stripWhitespace= T))
tdm.small <- removeSparseTerms(tdm, sparse = 0.9)
tdm.small

freq = rowSums(as.matrix(tdm.small))
head(freq,10)
pal = brewer.pal(8, "Blues")
pal = pal[-(1:3)]
set.seed(1234)
freq = sort(rowSums(as.matrix(tdm.small)), decreasing = T)
word.cloud = wordcloud(words=names(freq), freq=freq, min.freq=500,
                       random.order=F, colors=pal)
inspect(tdm)


##################### BIGRAM ##################### 

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tdm.bigram <- TermDocumentMatrix(corpus_noEmail,
                                control = list (tokenize = BigramTokenizer, removeNumbers = T, stripWhitespace = T ))
tdm.bigram
tdm.bigram.small <- removeSparseTerms(tdm.bigram, 0.9)
freq = sort(rowSums(as.matrix(tdm.bigram.small)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)
wordcloud(freqBigram.df$word,freqBigram.df$freq,max.words=100,random.order = F, colors=pal)

ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Bigrams") + ylab("Frequency") +
  ggtitle("Most frequent bigrams")



# m2 <- as.matrix(tdm.bigram.small)
# v2 <- sort(rowSums(m2),decreasing=TRUE)
# tdm_bigrams <- data.frame(word = names(v2),freq=v2)
# word.cloud = wordcloud(words=names(v2), freq=v2, min.freq=1, 
#                        random.order=F, colors=pal)
