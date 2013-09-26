This program clusters posts in the MLComp 20news-18828 dataset using sklearn's KMeans via 
an augmentation to the sklearn TfidfVectorizer that utilizes 
the SnowballStemmer from the NTLK.  

Given a new post in the form a string, get_related(post) will
predict which cluster post belongs to and then fetch three
posts from said cluster with various similarities.  

The mlcomp data needs to be downloaded, and then put in a directory
data/, or else the MLCOMP_DIR variable in the Python script needs to be changed
to point to the location where the mlcomp data is downloaded.
