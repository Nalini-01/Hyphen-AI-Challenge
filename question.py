from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer           
import re
import pandas as pd 

# Function to extract questions 
def prntquest(l):
    l1=[]
    for i in range(len(l)):
        if (i%2)==0:
            l1.append(l[i])
    return l1

# Function to extract answers

def prntans(l):
    l2=[]
    for i in range(len(l)):
        if (i%2)!=0:
            l2.append (l[i])
    return l2    

#########################  TRAINING #################################

#Create first training data frame
f=open('Training_set1.txt')
lines=f.readlines()
raw_data1={'Questions':prntquest(lines),'Answers':prntans(lines)}
train_file1=pd.DataFrame(raw_data1,columns=['Questions','Answers'])
f.close()

#Create second training data frame
f=open('Training_set2.txt')
lines=f.readlines()
raw_data2={'Questions':prntquest(lines),'Answers':prntans(lines)}
train_file2=pd.DataFrame(raw_data2,columns=['Questions','Answers'])

#Combine both of the training dataframes
frames_train = [train_file1,train_file2]
train_file= pd.concat(frames_train,ignore_index=True)     
f.close()


# Function to convert raw questions/answers into sring of meaningful words only 
def trainfile_string(raw_quesans):
     #Remove non-letters    
     letters_only = re.sub("[^a-zA-Z]", " ", raw_quesans)           
     #Convert to lower case, split into individual words
     words = letters_only.lower().split()                          
     #convert the stop words to a set
     stops = set(stopwords.words("english"))                       
     #Remove stop words
     meaningful_words = [w for w in words if not w in stops]       
     #Join the words back into one string separated by space, 
     #and return the result.
   
     return( " ".join( meaningful_words ))  
     
                                                   
# Get the number of reviews based on the dataframe column size    
num_Questions = train_file["Questions"].size  

# Initialize an empty list to hold the clean questions
clean_train_questions = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 

for i in xrange( 0, num_Questions ):
    if( (i+1)%1000 == 0 ):
        print "Question %d of %d\n" % ( i+1, num_Questions )                                                                    
    clean_train_questions.append(trainfile_string( train_file["Questions"][i] ))
    
     
# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.       

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 
# fit_transform()
train_data_features = vectorizer.fit_transform(clean_train_questions)

train_data_features = train_data_features.toarray()
print "Training the random forest..."

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 
# Fit the forest to the training set
forest = forest.fit( train_data_features, train_file["Answers"] )




#############################  TESTING ##########################
# Data frame for frst test file
f=open('test_data1.txt')
lines=f.readlines()
raw_data01={'Questions':prntquest(lines),'Answers':prntans(lines)}
test_file1=pd.DataFrame(raw_data1,columns=['Questions','Answers'])
f.close()

# Data frame for second test file
f=open('test_data2.txt')
lines=f.readlines()
raw_data02={'Questions':prntquest(lines),'Answers':prntans(lines)}
test_file2=pd.DataFrame(raw_data1,columns=['Questions','Answers'])
f.close()

# Combining both of the test_data frames
frames_test= [test_file1,test_file2]
test_file= pd.concat(frames_test,ignore_index=True) 
  
# Get the number of reviews based on the dataframe column size   
num_Questions = len(test_file["Questions"])


# Initialize an empty list to hold the clean questions
clean_test_questions = [] 

print "Cleaning and parsing the test set questions ...\n"
for i in xrange(0,num_Questions):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_Questions)
    clean_review =trainfile_string( test_file["Questions"][i] )
    clean_test_questions.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_questions)
test_data_features = test_data_features.toarray()
# Use the random forest to make Answers predictions
result = forest.predict(test_data_features)

#Copy the results to a pandas dataframe with an "Questions" column and a "Answers" column
output = pd.DataFrame( data={"Questions":test_file["Questions"], "Answers":result} )

#Use pandas to write the output file
output.to_csv('Result.txt',header=['Questions','Answers'], index=None, columns=('Questions','Answers'),sep='\n',encoding='utf-8')


                     
