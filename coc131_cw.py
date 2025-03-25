import numpy as np
from PIL import Image


# Please write the optimal hyperparameter values you obtain in the global variable 'optimal_hyperparm' below. This
# variable should contain the values when I look at your submission. I should not have to run your code to populate this
# variable.
optimal_hyperparam = {}

    class COC131:

        
       def __init__(self, dataset_root):
        """
        Initializes the dataset and loads images.
        :param dataset_root: The root directory containing class folders.
        """
        self.dataset_root = dataset_root
        self.x, self.y = self.load_dataset()

    def load_dataset(self):
        """Loads images, resizes them to 32x32 using PIL, and stores as numpy arrays."""
        images, labels = [], []
        
        dataset_dir = Image.Path(self.dataset_root)  # Use Pillow's path handling
        class_folders = dataset_dir.listdir()  # Get all items in dataset_root
        
        for class_folder in class_folders:
            if class_folder.is_dir():  # Ensure it's a folder (class label)
                for img_file in class_folder.listdir():  # List images inside folder
                    img = Image.open(img_file).resize((32, 32))  # Read & resize
                    images.append(np.array(img).astype(float))  # Convert to NumPy array
                    labels.append(class_folder.name)  # Use folder name as label

        self.x = np.array(images)
        self.y = np.array(labels)
        return self.x, self.y

    def q1(self, filename=None):
        """
        This function loads and flattens an image, returning the image array and its class label.
        
        :param filename: Name of an image file to locate inside the dataset.
        :return res1: Flattened 1D numpy array of the image (dtype=float, row-major order).
        :return res2: Corresponding class label (folder name).
        """
        if filename:
            dataset_dir = Image.Path(self.dataset_root)
            for class_folder in dataset_dir.listdir():  # Search all class folders
                if class_folder.is_dir():
                    img_path = class_folder / filename  # Construct full path
                    try:
                        img = Image.open(img_path).resize((32, 32))  # Read & resize
                        res1 = np.array(img).astype(float).flatten()  # Convert & flatten
                        res2 = class_folder.name  # Get class name
                        return res1, res2
                    except Exception:
                        continue  # If file not found, continue searching
        
        return np.zeros(1), ""  # Default return if filename not found


    def q2(self, inp):
        """
        This function should compute the standardized data from a given 'inp' data. The function should work for a
        dataset with any number of features.

        :param inp: an array from which the standardized data is to be computed.
        :return res2: a numpy array containing the standardized data with standard deviation of 2.5. The array should
        have the same dimensions as the original data
        :return res1: sklearn object used for standardization.
        """
        res1 = np.zeros(1)
        res2 = ''

        return res2, res1

    def q3(self, test_size=None, pre_split_data=None, hyperparam=None):
        """
        This function should build a MLP Classifier using the dataset loaded in function 'q1' and evaluate model
        performance. You can assume that the function 'q1' has been called prior to calling this function. This function
        should support hyperparameter optimizations.

        :param test_size: the proportion of the dataset that should be reserved for testing. This should be a fraction
        between 0 and 1.
        :param pre_split_data: Can be used to provide data already split into training and testing.
        :param hyperparam: hyperparameter values to be tested during hyperparameter optimization.
        :return: The function should return 1 model object and 3 numpy arrays which contain the loss, training accuracy
        and testing accuracy after each training iteration for the best model you found.
        """

        # normalize data

        res1 = object()
        res2 = np.zeros(1)
        res3 = np.zeros(1)
        res4 = np.zeros(1)

        return res1, res2, res3, res4

    def q4(self):
        """
        This function should study the impact of alpha on the performance and parameters of the model. For each value of
        alpha in the list below, train a separate MLPClassifier from scratch. Other hyperparameters for the model can
        be set to the best values you found in 'q3'. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: res should be the data you visualized.
        """

        alpha_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]

        res = np.zeros(1)

        return res

    def q5(self):
        """
        This function should perform hypothesis testing to study the impact of using CV with and without Stratification
        on the performance of MLPClassifier. Set other model hyperparameters to the best values obtained in the previous
        questions. Use 5-fold cross validation for this question. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: The function should return 4 items - the final testing accuracy for both methods of CV, p-value of the
        test and a string representing the result of hypothesis testing. The string can have only two possible values -
        'Splitting method impacted performance' and 'Splitting method had no effect'.
        """

        res1 = 0
        res2 = 0
        res3 = 0
        res4 = ''

        return res1, res2, res3, res4

    def q6(self):
        """
        This function should perform unsupervised learning using LocallyLinearEmbedding in Sklearn. You can assume that
        the function 'q1' has been called prior to calling this function.

        :return: The function should return the data you visualize.
        """

        res = np.zeros(1)

        return res