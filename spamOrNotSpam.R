

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

removeEmailAddress <- function(x) {
  gsub(pattern = "\\w*[[:blank:]][[:punct:]][[:blank:]]\\w*[[:blank:]][[:punct:]][[:blank:]]com[[:blank:]]", replacement = "", x)
}

removeNumeration <- function(x) {
  gsub(pattern = "[[:digit:]][[:blank:]]th", replacement = "", x)
}



corpus <- VCorpus(VectorSource(emails$text))
inspect(corpus[[4]])
corpus_lowercase <- tm_map(corpus, content_transformer(tolower))
inspect(corpus_lowercase[[1]])
corpus_noEmailAddress <- tm_map(corpus_lowercase, content_transformer(removeEmailAddress))
inspect(corpus_noEmailAddress[[4]])
corpus_noNumeration <- tm_map(corpus_noEmailAddress, content_transformer(removeNumeration))


##################### UNIGRAM ##################### 

tdm = TermDocumentMatrix(corpus_noEmailAddress, control=list(removePunctuation = T, removeNumbers = T, stopwords = c(stopwords(), "subject", "subject re", "vince", "hou", "ect", "kaminski"), stripWhitespace= T))
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
corpus.ngrams = tm_map(corpus_noNumeration,removeWords,c(stopwords(),"re", "ect", "hou", "e", "mail", "kaminski", "hou", "cc", "subject", "vince", "j", "enron"))
corpus.ngrams = tm_map(corpus.ngrams,removePunctuation)
corpus.ngrams = tm_map(corpus.ngrams,removeNumbers)


BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
# tdm.bigram <- TermDocumentMatrix(corpus_noNumeration,
#                                 control = list (tokenize = BigramTokenizer, removeNumbers = T, stripWhitespace = T , removePunctuation = T, stopwords = c(stopwords(), " ect", "subject re", "ect ", "hou", "vince", "hou ", " hou", "cc subject", " am", "on ", "enron ", " enron", " pm", "j kaminski", "kaminski ", "http", "ect cc", " to")))
tdm.bigram <- TermDocumentMatrix(corpus.ngrams,
                                control = list (tokenize = BigramTokenizer))

tdm.bigram
tdm.bigram.small <- removeSparseTerms(tdm.bigram, 0.99)
inspect(tdm.bigram.small)

freqBigram = sort(rowSums(as.matrix(tdm.bigram.small)),decreasing = TRUE)
freqBigram.df = data.frame(word=names(freqBigram), freq=freqBigram)
head(freqBigram.df, 20)
wordcloud(freqBigram.df$word,freqBigram.df$freq,max.words=100,random.order = F, colors=pal)

ggplot(head(freqBigram.df,15), aes(reorder(word,freqBigram), freqBigram)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Bigrams") + ylab("Frequency") +
  ggtitle("Most frequent bigrams")



# m2 <- as.matrix(tdm.bigram.small)
# v2 <- sort(rowSums(m2),decreasing=TRUE)
# tdm_bigrams <- data.frame(word = names(v2),freq=v2)
# word.cloud = wordcloud(words=names(v2), freq=v2, min.freq=1, 
#                        random.order=F, colors=pal)
