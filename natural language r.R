dataset_original=read.delim('Restaurant_Reviews.tsv',quote = '',stringsAsFactors = FALSE)
#cleaning th text
library(tm)
library(SnowballC)

corpus=VCorpus(VectorSource(dataset$Review))
corpus= tm_map(corpus,content_transformer(tolower))
corpus= tm_map(corpus,removeNumbers)
corpus= tm_map(corpus,removePunctuation)
corpus= tm_map(corpus,removeWords,stopwords(kind='en'))
corpus= tm_map(corpus,stripWhitespace)
corpus= tm_map(corpus,stemDocument)

#creating bag of word models
dtm=DocumentTermMatrix(corpus)
dtm=removeSparseTerms(dtm,0.999)

dataset=as.data.frame(as.matrix(dtm))
dataset$Liked=dataset_original$Liked

dataset$Liked=factor(dataset$Liked,levels = c(0,1))
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling

#fitting lclassifier

classifier=randomForest(x= training_set[-692],y=training_set$Liked,ntree = 10)

'predicting the test set results'

y_pred=predict(classifier,newdata = test_set[-692])


#evauation of results by making confusion matrix
cm=table(test_set[,692],y_pred)