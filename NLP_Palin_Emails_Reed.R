### Import Libraries
library(tm)
library(qdap)
library(tibble)
library(ggplot2)
library(RWeka)
library(wordcloud)
library(lubridate)
library(lexicon)
library(tidytext)
library(lubridate)
library(gutenbergr)
library(stringr)
library(dplyr)
library(radarchart)

library(readtext)
library(tidyr)

library(lexicon)

# Set path to directory with a list of .txt files that are the emails
# 393.txt example file name
path = "Insert Your Path Here"
setwd(path)
text_read = readtext(paste0(path,"\\*.txt"))


# remove non alpha numeric characters 
text_read$text <- iconv(text_read$text, from = "UTF-8", to = "ASCII", sub = "")

# Here, set the total number of emails you want to look at, for example, I have it at 2000 so I am 
# looking at the first 2000 emails. Around 3000 emails during parts of the code your computer may
# run out of memory, especially during the TDF to matrix parts
emails_to_read = 2000
text_read = text_read[1:emails_to_read,] 

# Here we realize that most emails have useless information before the first 'Subject:'
# If you are looking to keep that information, skip this for loop
cnt = 1
for (text in text_read$text){
  text_read$text[cnt] = str_split(text,'Subject')[[1]][2]
  cnt = cnt + 1
}

# Turn the vector of email text bodies into a vector for later use
review_corpus <- VCorpus(VectorSource(text_read$text))

# This function takes in a corpus, like review_corpus above, and returns a cleaned corpus
# There are many stopwords I included below, these should be changed if the user is looking
# to keep certain things like day of the week or emails where Sarah Palin comes up
clean_corpus <- function(cleaned_corpus){
  removeURL <- content_transformer(function(x) gsub("(f|ht)tp(s?)://\\S+", "", x, perl=T))
  cleaned_corpus <- tm_map(cleaned_corpus, removeURL)
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(replace_abbreviation))
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(tolower))
  cleaned_corpus <- tm_map(cleaned_corpus, removePunctuation)
  cleaned_corpus <- tm_map(cleaned_corpus, removeNumbers)
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, stopwords("english"))

  # Used custom stopwords based on trial and error seeing a lot of strange words show up in the word clouds
  custom_stop_words = c("alaska", "ak","re:","www.crivellawest.com"
                        ,"www.yahoo.com",'yahoo','dot','com','message','original',
                        'gov','sent',"msnbc.com","posted","searchable","analytics",
                        'governor','sarah','palin','will','one','state','can','date','sarahyahoo',
                        'thank','thanks','email','pra','may','april','know','need','govsarahyahoo',
                        'get','just','like','also','let','frye','togovsarahyahoo','mailtowebmailgovstateakus'
                        ,'akus','msnbc','tuesday','friday','monday','wednesday','saturday','sunday','thursday',
                        'thu','mon','httppalinemailmsnbcmsn','publica','www','crivellawest','k','e','january',
                        'osoaoufirst','pm','w','y','r','subject','fw','february','am','page','unknown',
                        'pragsp','ftbyahoo','wed','pro','cellular','device','blackberry',
                        'webmailgovstateakus','december','dnr','conocophilips','tues','see','said','jan',
                        'feb','mar','sat','sun','apr','may','jun','jul','aug','sep','oct','nov','dec')
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, custom_stop_words)
  cleaned_corpus <- tm_map(cleaned_corpus, stripWhitespace)
  return(cleaned_corpus)
}

# Clean the original corpus using the previously created function
cleaned_review_corpus <- clean_corpus(review_corpus)

# From the corpus, create a term document matrix and set it as a matrix data type
# This is a part where your computer could have a memory issue if you include too much in the corpus
# by either including too many emails at the beginning or having little to no cleaning take place in 
# your custom clean_corpus function
TDM_reviews <- TermDocumentMatrix(cleaned_review_corpus)
TDM_reviews_m <- as.matrix(TDM_reviews)


# Here we add up the counts of each word without worrying about what email they came from
# For example, if 'hello' showed up 10 times in the whole group of emails, there would
# be a 10 next to the word 'hello' in term_frequency
term_frequency <- rowSums(TDM_reviews_m)

# Sort term_frequency in descending order so we can easily view the top words
term_frequency <- sort(term_frequency,dec=TRUE)

# View the top 20 most common words
top20 <- term_frequency[1:20]

# Plot a barchart of the 20 most common words
barplot(top20,col="darkorange",las=2,horiz = TRUE,cex.names= (0.6))


############ Word Cloud
# Create word_freqs dataframe because the wordcloud function below accepts that data type
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)

# Create a wordcloud for the values in word_freqs, here we are only taking into consideration words
# that show up at least 5 times and we are taking the top 100 words
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,max.words=100,colors=brewer.pal(8, "Paired"))


############## bigrams and trigrams
# This function takes in a corpus and returns grams of length min to max, so by
# setting min and max to the same value, you only get those size grams
# Example, setting min = 2 and max = 2 returns only bigrams
tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=2,max=2))

# Here we use the TermDocumentMatrix function again to get a TDM of Bigrams now and set
# its datatype to matrix so later functions will work on it like rowsums
bigram_tdm <- TermDocumentMatrix(cleaned_review_corpus,control = list(tokenize=tokenizer))
bigram_tdm_m <- as.matrix(bigram_tdm)

# Here we add up the total number of times a bigram shows up and sort them so wordcloud picks up the top values
term_frequency <- rowSums(bigram_tdm_m)
term_frequency <- sort(term_frequency,dec=TRUE)

# Here we change the datatype of term_frequency into a dataframe so wordcloud works
# Then we create the bigram word cloud
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,max.words=150,colors=brewer.pal(8, "Paired"))




############# Tri-Gram ################
# Here we do the same as the bigram process above except now min = 3 and max = 3 gives us tri-grams
tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=3,max=3))
bigram_tdm <- TermDocumentMatrix(cleaned_review_corpus,control = list(tokenize=tokenizer))
bigram_tdm_m <- as.matrix(bigram_tdm)
term_frequency <- rowSums(bigram_tdm_m)
term_frequency <- sort(term_frequency,dec=TRUE)
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,max.words=150,colors=brewer.pal(8, "Paired"))

##########tf-idf weighting word cloud
# Here we create a term frequency inverse document frequency unigram word cloud to show
# which words show up most often with less weight on long emails which may bias our count of words
# because a few emails include them very often

# The process is exactly similar to the tri-gram and bi-gram cases above, except in the
# TermDocumentMatrix function the control is weightTfIdf
tfidf_tdm <- TermDocumentMatrix(cleaned_review_corpus,control=list(weighting=weightTfIdf))
tfidf_tdm_m <- as.matrix(tfidf_tdm)
term_frequency <- rowSums(tfidf_tdm_m)
term_frequency <- sort(term_frequency,dec=TRUE)
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,max.words=1000,colors=brewer.pal(8, "Paired"))

##############################################################################################
################ Sentiment Analysis using Bing Lexicon  ######################################
#############################################################################################

# Here we use the Tidy package to create a different approach to viewing the words so we
# can take advantage of giving each word a polarity score, add them all up, and then
# join them on email to get an overall score for each email

# Here, we are kind of starting over and resetting this new variable, mytext, equal to our emails from
# before the wordcloud bit above
mytext = text_read
#Tidy TDM
mytext$text <- iconv(mytext$text, from = "UTF-8", to = "ASCII", sub = "")
mytext_corpus <- VCorpus(VectorSource(mytext$text))
cleaned_mytext_corpus <- clean_corpus(mytext_corpus)
TDM_mytext <- TermDocumentMatrix(cleaned_mytext_corpus)
# Here, we now have a matrix called mytext_tidy which has three columns - word, email, count
mytext_tidy <- tidy(TDM_mytext)

# bing
# Load in Bings pre-defined rules for which words are positive or negative
bing_lex <- get_sentiments("bing")

# Here we join the words in the Bing lexicon with the tidy matrix
mytext_bing_lex <- inner_join(mytext_tidy, bing_lex, by = c("term" = "word"))

#Here we put a -1 if the sentiment joined over from Bing is negative and a 1 if otherwise (positive) 
mytext_bing_lex$sentiment_n <- ifelse(mytext_bing_lex$sentiment=="negative", -1, 1)

# By multiplying the count column in mytext_tidy by the 1 or -1 now introduced, we get a total score
# for that word in that email, and by aggregating it into big_aggdata, we now have a total score for
# each email
mytext_bing_lex$sentiment_value <- mytext_bing_lex$sentiment_n * mytext_bing_lex$count
bing_aggdata <- aggregate(mytext_bing_lex$sentiment_value, list(index = mytext_bing_lex$document), sum)

# Change index column to numeric data type
bing_aggdata$index <- as.numeric(bing_aggdata$index)
# Change column names for later on and for consistency
colnames(bing_aggdata) <- c("index","bing_score")

# First, graph sentiment as a function of email number
ggplot(bing_aggdata, aes(index, bing_score)) + geom_point()

# Then, graph sentiment in the same fashion but use smoothing to get a more general view of sentiment
# as emails go on
ggplot(bing_aggdata, aes(index, bing_score)) + geom_smooth() + theme_bw()+ xlab("email")+ylab("sentiment")+
  ggtitle("Sentiment Versus Email Number")



####################### Radar Chart Emotional Analysis
# Here we use the NRC pre-defined definitions of words for 8 different emotions and plot them on a radar chart

# Load the pre-defined definitions
nrc_lex <- get_sentiments("nrc")

# Join the mytext_tidy with the lexicon rules so each word gets its own emotion
story_nrc <- inner_join(mytext_tidy, nrc_lex, by = c("term" = "word"))

# Take out any emotions that are positive or negative because we only want to look at emotions
story_nrc_noposneg <- story_nrc[!(story_nrc$sentiment %in% c("positive","negative")),]

# Sum the number of each emotion by word, then plot
aggdata <- aggregate(story_nrc_noposneg$count, list(index = story_nrc_noposneg$sentiment), sum)
chartJSRadar(aggdata)



##################### Comparison Clouds based on sentiment
# Again, kind of restarting here, going to compare positive emails to negative emails based on the
# Bing lexicon score, not NRC from above

# Put in a numeric index column for later use
mytext$index = c(1:2000)

# Go through the Bing text from before and create a new column where an email gets a postive or negative flag
bing_aggdata$posneg = ifelse(bing_aggdata$bing_score < 0, 'Negative','Positive')

# Seperate out two subsets that are either only positive or only negative
positives = bing_aggdata[ which(bing_aggdata$posneg == 'Positive'),]
negatives = bing_aggdata[ which(bing_aggdata$posneg == 'Negative'),]

# Join positves and negatives with the actual text from before
pos_texts = inner_join(positives,mytext,by = c('index' = 'index'))
neg_texts = inner_join(negatives,mytext,by = c('index' = 'index'))

# Turn all the positive and negative emails into a giant string
pos_texts2 = pos_texts$text
neg_texts2 = neg_texts$text

pos_string =''
neg_string = ''
for (i in pos_texts$text){
  pos_string = paste(pos_string,i)
}

for (j in neg_texts$text){
  neg_string = paste(neg_string,j)
}

# Combine the two long strings into a single matrix
posneg_text <- c(pos_string,neg_string)

# Turn the text matrix into a corpus by using the vecorization function here below 
posneg_corpus <- VCorpus(VectorSource(posneg_text))

# Clean the corpus using the same function from way before
cleaned_posneg_corpus <- clean_corpus(posneg_corpus)

# Turn corpus into a TDM again, with datatype equal to matrix, watch computer memory here too!
TDM_posneg <- TermDocumentMatrix(cleaned_posneg_corpus)
TDM_posneg_m <- as.matrix(TDM_posneg)

############################# Commonality Cloud
# Here, what words show up in both positive and negative emails, this is so
# not just one class of email can dominate the term frequency matrix
commonality.cloud(TDM_posneg_m,colors=brewer.pal(8, "Dark2"),max.words = 100)

################# Comparison Cloud
# Here, we compare postive to negative emails to see which words show up most in positives and negatives 
# seperately
TDM_posneg <- TermDocumentMatrix(cleaned_posneg_corpus)
colnames(TDM_posneg) <- c("Positive Emails","Negative Emails")
TDM_posneg_m <- as.matrix(TDM_posneg)
comparison.cloud(TDM_posneg_m,colors=brewer.pal(8, "Dark2"),max.words = 100)




