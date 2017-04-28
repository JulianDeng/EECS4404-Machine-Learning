from __future__ import print_function
from collections import OrderedDict, Counter
from numpy import *
from operator import itemgetter
import matplotlib.pyplot as plt
import csv

def classify(alpha, dataset, poi_fq, edi_fq, prior_poi, prior_edi):
    global reverse_maps
    global char_maps

    #Calculate lp_poi and lp_edi
    lp_poi = []
    lp_edi = []

    for i,v in enumerate(dataset):
        temp_poi = prior_poi
        temp_edi = prior_edi
        for j,w in enumerate(v):
            if j != 0:
                if (poi_fq[0][0] == 0) and (alpha == 1.0):                            # circumvent the log zero problem when alpha = 1
                    temp_poi += 0
                else:
                    temp_poi += log((poi_fq[j][w] + alpha - 1) / (poi_fq[0][0] + len(reverse_maps[j]) * alpha - len(reverse_maps[j])))
                if (edi_fq[0][1] == 0) and (alpha == 1.0):                            # circumvent the log zero problem when alpha = 1
                    temp_edi += 0
                else:
                    temp_edi += log((edi_fq[j][w] + alpha - 1) / (edi_fq[0][1] + len(reverse_maps[j]) * alpha - len(reverse_maps[j])))
        lp_poi.append(temp_poi)
        lp_edi.append(temp_edi)

    # Calculate posterior for poisonous and edible and assign the greater one to result set (big B).
    poi = []
    edi = []
    result = []

    for i in range(len(dataset)):
        poi.append(exp(lp_poi[i] - (max(lp_poi[i],lp_edi[i]) + log(exp(lp_edi[i] - max(lp_poi[i],lp_edi[i])) + exp(lp_poi[i] - max(lp_poi[i],lp_edi[i]))))))
        edi.append(exp(lp_edi[i] - (max(lp_poi[i],lp_edi[i]) + log(exp(lp_poi[i] - max(lp_poi[i],lp_edi[i])) + exp(lp_edi[i] - max(lp_poi[i],lp_edi[i]))))))
        if (poi[i] > edi[i]):
            result.append(char_maps[0]['p'])   #assign poisonous flag if poi[i] is greater
        else:
            result.append(char_maps[0]['e'])   #otherwise assign edible flag
    return result

def get_accuracy(result,dataset):
    #Calculate the accuracy of the classification result for the given dataset
    correct_count = 0
    total_count = len(dataset)

    for i,v in enumerate(dataset):
        if(v[0] == result[i]):
            correct_count += 1
    accuracy = correct_count / (total_count * (1.0))
    return accuracy

def plot_accuracy_alpha(alpha_set, accuracy_train, accuracy_test):
    plt.plot(alpha_set, accuracy_train, label="Train Set")
    plt.plot(alpha_set, accuracy_test, label="Validation Set")
    plt.legend(loc='upper right')
    ax = plt.axes()
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("alpha")
    plt.show()
    plt.close()
    return None

def plot_feature_frequency(fq_set, p_e):
    global reverse_maps
    for i, v in enumerate(fq_set):
        pos = arange(len(reverse_maps[i]))
        width = 1.0

        ax = plt.axes()
        ax.set_xticks(pos + (width / 2))
        ax.set_xticklabels(reverse_maps[i])
        ax.set_ylim(0, 1.0)
        title = "Feature %d " % i + p_e
        ax.set_title(title)

        norm = [float(i) / sum(v) for i in v]

        plt.bar(pos, norm, width, align='center', color='r')
        plt.show()
        plt.close()
    return None


#=========================================Code start from here==========================================================


if __name__ == "__main__":
    # Professor's code start from here
    filename = 'mushrooms.csv'
    with open(filename, 'rb') as raw_file:
        raw_data = csv.reader(raw_file, delimiter=',', quoting=csv.QUOTE_NONE)
        data_list = list(raw_data)

    ndims = len(data_list[0])
    npts = len(data_list)

    char_maps = [OrderedDict() for i in range(ndims)]
    reverse_maps = [[] for i in range(ndims)]
    data_mat = empty((npts,ndims),dtype=int32)
    for i,cdata in enumerate(data_list):
        for j,cstr in enumerate(cdata):
            if cstr not in char_maps[j]:
                char_maps[j][cstr] = len(char_maps[j])
                reverse_maps[j].append(cstr)
            data_mat[i,j] = char_maps[j][cstr]
    del data_list

    random.seed(0)
    data_perm = random.permutation(npts)
    data_train = data_mat[data_perm[0:(8*npts/10)],:]
    data_test = data_mat[data_perm[(8*npts/10):],:]
    data_ranges = data_mat[:,1:].max(axis=0)

    #=====================================My code start here===========================================================

    # Split original training data into poisonous set and edible set
    poi_set = [[] for i in range(len(char_maps))]
    edi_set = [[] for i in range(len(char_maps))]

    for i, v in enumerate(data_train):
        if v[0] == char_maps[0]['p']:
            for j in range(len(char_maps)):
                poi_set[j].append(v[j])
        else:
            for j in range(len(char_maps)):
                edi_set[j].append(v[j])

    # Use Counter class to generate frequency information matrix for both poisonous and edible set
    poi_counter = [() for i in range(len(char_maps))]
    edi_counter = [() for i in range(len(char_maps))]

    for i in range(0, len(char_maps)):
        poi_counter[i] = Counter(poi_set[i])
        edi_counter[i] = Counter(edi_set[i])

    poi_fq = [[] for i in range(len(char_maps))]
    edi_fq = [[] for i in range(len(char_maps))]
    for i, v in enumerate(char_maps):
        for j in range(len(char_maps[i])):
            poi_fq[i].append(poi_counter[i][j])
            edi_fq[i].append(edi_counter[i][j])

    # Plot histograms for each feature in poisonous and edible set                                         <<<<<<<<<<==========================Step 1 plotting histograms
    plot_feature_frequency(poi_fq,"poisonous")
    plot_feature_frequency(edi_fq,"edible")

    # Acquire accuracy for both train set and test set given random alpha between 1 and 2                  <<<<<<<<<<==========================Step 2 start from here
    alpha = 1.0                                   # initialize alpha as 1
    experiment_time = 100                         # experiment 100 times to acquire the optimal alpha

    alpha_set = []
    accuracy_train = []
    accuracy_test = []

    # For each experiment... (alpha incrementally increasing)
    for i in range(experiment_time + 1):
        alpha_set.append(alpha)

        # Calculate the prior
        prior_poi = (poi_fq[0][0] + alpha - 1) / float(len(data_train) + 2 * alpha - 2)
        prior_edi = (edi_fq[0][1] + alpha - 1) / float(len(data_train) + 2 * alpha - 2)

        # Classify train set and test set
        result_train = classify(alpha, data_train, poi_fq, edi_fq, prior_poi, prior_edi)
        result_test = classify(alpha, data_test, poi_fq, edi_fq, prior_poi, prior_edi)

        # Calculate the accuracy based on result
        accuracy_train.append(get_accuracy(result_train, data_train))
        accuracy_test.append(get_accuracy(result_test, data_test))
        print("No. %d: "%i, "alpha=", alpha, ", train_accurarcy=", accuracy_train[i], ", validation_accuracy=", accuracy_test[i])

        alpha += 1 / (1.0 * experiment_time)  # increment alpha by 1 / experiment_time for each loop

    print("optimal alpha is: ", alpha_set[argmax(accuracy_train)])

    # Plot the accuracy diagram for both train set and validation set with respect to the specified alpha
    plot_accuracy_alpha(alpha_set, accuracy_train, accuracy_test)

    # Find the most discriminative feature...Use the optimal alpha                  <<<<==================================Step 3 start from here
    alpha = alpha_set[argmax(accuracy_train)]       # optimal alpha
    impact_list = []
    for i, v in enumerate(reverse_maps):
        for j, w in enumerate(v):
            impact = log((edi_fq[i][j] + alpha - 1) / (edi_fq[0][1] + len(reverse_maps[i]) * alpha - len(reverse_maps[i]))) - log((poi_fq[i][j] + alpha - 1) / (poi_fq[0][0] + len(reverse_maps[i]) * alpha - len(reverse_maps[i])))
            impact_list.append((i, j, impact, abs(impact)))

    impact_list = impact_list[2:]                              # remove the class item from the list

    impact_list = sorted(impact_list, key=itemgetter(2))       # sort the tuple list by value from small to large

#    Below are deprecated as there are multiple features of highly discriminative
#    disc_poi_feature = impact_list[0][0]
#    disc_edi_feature = impact_list[len(impact_list) - 1][0]
#    disc_poi_option = impact_list[0][1]
#    disc_edi_option = impact_list[len(impact_list) - 1][1]

    impact_list = sorted(impact_list, key=itemgetter(3))       # sort by aboslute value

    for i, v in enumerate(impact_list):
        print(v)

    high_disc_list = []
    for i,v in enumerate(impact_list):
        if (v[3] == float("inf")):
            high_disc_list.append(v)

#    Below are deprecated as there are multiple features of highly discriminative
#    print("Most discriminative indicator for poisonous mushroom is: feature " , disc_poi_feature , ", option '" , reverse_maps[disc_poi_feature][disc_poi_option] , "'")
#    print("Most discriminative indicator for edible mushroom is: feature " , disc_edi_feature , ", option '" , reverse_maps[disc_edi_feature][disc_edi_option] , "'")


    for i,v in enumerate(high_disc_list):
        if v[2] == float("-inf"):
            print("Discriminative feature for poisonous mushroom is: feature ", v[0], ", option ", v[1], "   ", reverse_maps[v[0]][v[1]])
        if v[2] == float("inf"):
            print("Discriminative feature for edible mushroom is: feature ", v[0], ", option ", v[1], "   ", reverse_maps[v[0]][v[1]])