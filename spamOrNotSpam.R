library(tm)
library(wordcloud)


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
emails <- read.csv('emails.csv')
length(emails$text)
tdm = TermDocumentMatrix(emails)
corpus <- VCorpus(VectorSource(emails$text))
tdm = TermDocumentMatrix(corpus)

freq = rowSums(as.matrix(tdm))
tail(freq,10)
pal = brewer.pal(8, "Blues")
pal = pal[-(1:3)]
set.seed(1234)

freq = sort(rowSums(as.matrix(tdm)), decreasing = T)
word.cloud = wordcloud(words=names(freq), freq=freq, min.freq=400,
                       random.order=F, colors=pal)
