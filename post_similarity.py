from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
import nltk.stem, os
import sklearn.datasets
from sklearn.cluster import KMeans


def main():
    vectorizer = StemmedTfidfVectorizer(
            min_df = 10, 
            max_df=.5, 
            stop_words = 'english', 
            decode_error = 'ignore')
    
    # Get the 20 newsgroups dataset and use sklearn's customer loaders for mlcomp data
    
    MLCOMP_DIR = 'data/' # points to the data directory
    full_set =  sklearn.datasets.load_mlcomp("20news-18828", mlcomp_root=MLCOMP_DIR)
    train_set = sklearn.datasets.load_mlcomp("20news-18828", "train", mlcomp_root=MLCOMP_DIR)
    test_set =  sklearn.datasets.load_mlcomp("20news-18828", "test", mlcomp_root=MLCOMP_DIR)
    # vectorize the data using StemmedTfidfVectorizer
    vect_train_data = vectorizer.fit_transform(train_set.data)
    vect_test_data = vectorizer.transform(test_set.data)
    vect_full_set = vectorizer.transform(full_set.data)
    # cluster the data
    num_clusters = 50
    km = KMeans(n_clusters = num_clusters, init='random', n_init=1, verbose=1)
    km.fit(vect_train_data)
    
    # testing
    new_post = "Hey guys.  I think that polar bears are pretty cool animals."
     
    
def get_related(post):
    """Given a string consisting of a post, predicts its cluster using the already trained km object.
    Returns three related posts from the same cluster.
    
    Args: 
        post (string) - a string consisting of a post to be assigned to a cluster
    
    Return: An index array corresponding to three posts in the same cluster as post of varying similarities.    
    """
    post_vec = vectorizer.transform([post])
    post_label = km.predict(post_vec)[0]
    # get all indicies of posts in the same cluster as the input post
    cluster_indices = (km.labels_ == post_label).nonzero()[0]
    related = []
    for i in cluster_indices:
        dist = sp.linalg.norm((post_vec - vect_full_set[i]).toarray())
        related.append((dist,full_set.data[i]))
    related = sorted(related)
    post_1 = related[0]
    post_2 = related[len(related)/2]
    post_3 = related[-1]
    posts = [related[0], related[len(related)/2], related[-1]]
    for item in posts:
        print("Similarity:",item[0], '\n Post:', item[1])
    

# Overwrite the build_analyzer method of sklearn.CountVectorizer to include stemming
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer,self).build_analyzer()
        eng_stem = nltk.stem.SnowballStemmer('english')
        return lambda doc: (eng_stem.stem(w) for w in analyzer(doc))


main()

        
