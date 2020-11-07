import numpy as np
import cv2
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from os import listdir
import sys

#this function will get SIFT descriptors from training images and
#train a k-means classifier
def read_and_clusterize(train_folder, num_clusters):

    sift_keypoints = []

    for file in listdir(train_folder):
        #read image
        image = cv2.imread(train_folder + file, 1)
        # Convert them to grayscale
        image =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # SIFT extraction
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(image, None)
        #append the descriptors to a list of descriptors
        sift_keypoints.append(descriptors)

    sift_keypoints = np.asarray(sift_keypoints)
    sift_keypoints = np.concatenate(sift_keypoints, axis=0)
    #with the descriptors detected, lets clusterize them
    print("Training kmeans")
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=100, random_state=0,  init_size=1000).fit(sift_keypoints)
    #return the learned model
    return kmeans

#with the k-means model found, this code generates the feature vectors
#by building an histogram of classified keypoints in the kmeans classifier
def calculate_centroids_histogram(folder, model, num_clusters):

    feature_vectors = []
    class_vectors = []

    for file in listdir(folder):
        #read image
        image = cv2.imread(folder + file, 1)
        #Convert them to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #SIFT extraction
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(image, None)
        #classification of all descriptors in the model
        predict_kmeans = model.predict(descriptors)
        #calculates the histogram
        hist, bin_edges = np.histogram(predict_kmeans, bins=num_clusters)
        #histogram is the feature vector
        feature_vectors.append(hist)
        #define the class of the image (elephant or electric guitar)
        class_sample = define_class(file)
        class_vectors.append(class_sample)

    feature_vectors = np.asarray(feature_vectors)
    class_vectors = np.asarray(class_vectors)
    #return vectors and classes we want to classify
    return class_vectors, feature_vectors


def define_class(img_name):

    # 1 = dog, 0 = cat
    if img_name.startswith('dog'):
        class_image = 1

    elif img_name.startswith('cat'):
        class_image = 0

    return class_image

def main(train_folder, test_folder, num_clusters, method):
    #step 1: read and detect SIFT keypoints over the input image (train images) and clusterize them via k-means
    print("Step 1: Extracting SIFT features and calculating Kmeans classifier")
    model = read_and_clusterize(train_folder, num_clusters)

    print("Step 2: Generating histograms of training and testing images")
    print("Training")
    [train_class, train_featvec] = calculate_centroids_histogram(train_folder, model, num_clusters)
    print("Testing")
    [test_class, test_featvec] = calculate_centroids_histogram(test_folder, model, num_clusters)

    if method == 'SVM-poly':
        print("Step 3: Training the SVM classifier")
        clf = svm.SVC(kernel='poly', degree=2)
        clf.fit(train_featvec, train_class)
    elif method == 'SVM-rbf':
        print("Step 3: Training the SVM classifier")
        clf = svm.SVC(kernel='rbf')
        clf.fit(train_featvec, train_class)
    elif method == 'SVM-linear':
        print("Step 3: Training the SVM classifier")
        clf = svm.SVC(kernel='linear')
        clf.fit(train_featvec, train_class)
    elif method == 'LR':
        print("Step 3: Training the Logistic Regression classifier")
        clf = LogisticRegression()
        clf.fit(train_featvec, train_class)
    elif method == 'KNN':
        print("Step 3: Training the Logistic Regression classifier")
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(train_featvec, train_class)

    print("Step 4: Testing the classifier")
    predict = clf.predict(test_featvec)

    score = accuracy_score(np.asarray(test_class), predict)

    file_object = open("results.txt", "a")
    file_object.write("%f\n" % score)
    file_object.close()

    print("Accuracy:" + str(score))

if __name__ == "__main__":
    method = sys.argv[1]
    train_folder = 'train/'
    test_folder = 'test/'
    main(train_folder, test_folder, 100, method)
