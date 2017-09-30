import numpy as np
import random
#loading the load from a textfile in the form of numpy array
data = np.loadtxt('irisdata.txt', delimiter=',')
# extracting all the input_values excluding the class labels from loaded data which is basically all rows and columns from 1 to 4 or 0 to 3
input_values = data[0:, 0:5]
input_values = np.append(np.ones((150, 1)), input_values, axis=1)
#getting the class labels from the loaded data which is the last column of the data
Target_values = data[0:, 4]
#initialization
# this is basically the value of k which describes the k-fold cross validation
m = len(Target_values)
k = 10
accuracy_mean = 0
initial_col = 0
values_k = int(m/k)
# generating the unique indices in the range of 0 to the no of rows and generating values equal to 150 ie no of rows
list_indices = random.sample(range(len(Target_values)), len(Target_values))
#doing cross validation by generating the test and train data from a iris data
for i in range(k):
    # getting the test indices as the first fold from list indices and then getting other folds on every iteration
    a =list_indices[initial_col:values_k]
    # getting test data from input_values as per generated test indices
    test_data = input_values[a, 0:]
    # getting the train indices from list_indices which is not there in a ie test indices
    b = [list_indices[i] for i in range(len(list_indices)) if list_indices[i] not in a]
    # finally getting the first train data and similarly getting other in other iterations
    train_data = input_values[b,0:]
    #getting the classes column from the train data
    traindata_y = train_data[:, 5]
    #getting feature columns from the train data
    traindata_x = train_data[0:,0:5]
    # getting the optimised weight for features through normal equation deduction
    B = np.dot(np.dot((np.linalg.pinv(np.dot(np.transpose(traindata_x),(traindata_x)))),np.transpose(traindata_x)),traindata_y)

    #hypothesis value ie predicted output values
    output_values = np.dot(B,np.transpose(test_data[:,0:5]))

    #rounding all the values in the output_values matrix as our classes are 1,2,3
    output_values = np.round(output_values)
    # checking the accuracy so getting the test labels and comparing it with our predicted labels
    test_data_y = test_data[:,5]
    # parameter for checking how many labels are correctly predicted
    accurate_count = 0
    for i in range(len(output_values)):
        # checking condition for predicted values equal to actual labels
        if(test_data_y[i] == output_values[i]):
            # incrementing my count for this variable every time i am getting the predicted labels same as actual labels
            accurate_count = accurate_count+1
    #getting the accuracy
    accuracy = (float(accurate_count)/float(len(output_values)))*100
    print(accuracy)
    #summinig up all the accuracies in order to find mean accuracy
    accuracy_mean = accuracy_mean + accuracy
    # incrementing the initial_col values to values_k value which is 15 in first, 30 in second and 45 and so on
    initial_col = values_k
    #similarly geting the values_k and incrementing it ie 30,45,60 and so on
    values_k = values_k + int(m/k)
accuracy_mean_avg = accuracy_mean/k
print("this is your mean accuracy:")
print(accuracy_mean_avg)
