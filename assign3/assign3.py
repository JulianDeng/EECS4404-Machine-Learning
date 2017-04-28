#========================WARNING: Please use 2.7.12 64-bit python to work around memory error======================

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy import linalg
from scipy import io as scio
from numpy import linalg as nla

import random


def plot_img(img, grayscale=False):
    if grayscale is False:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.show()
    plt.close()


def plot_gen(x, y, x_label, y_label, title):
    plt.plot(x, y)
    plt.axes().set_xlabel(x_label)
    plt.axes().set_ylabel(y_label)
    plt.axes().set_title(title)
    plt.axes().set_xlim(-0.02 * max(x), 1.02 * max(x))
    plt.axes().set_ylim(-0.02 * max(y), 1.02 * max(y))
    plt.show()
    plt.close()


def get_w_basis(c_dim, eig_vec):
    return eig_vec.T[0:c_dim].T


# =========================================Code start from here========================================================
#========================WARNING: Please use 2.7.12 64-bit python to work around memory error==========================
# ======================================step1========================================
if __name__ == "__main__":
    inputdata = scio.loadmat("face_images.mat")
    train_ids = inputdata["train_ids"]
    test_ids = inputdata["test_ids"]
    test_imgs = inputdata["test_imgs"]
    train_imgs = inputdata["train_imgs"]
    print train_imgs.shape
    rnd_num = random.randint(0, len(train_imgs))
    print "index ", rnd_num
    plt.axes().set_title("Random face image from train set")
    plot_img(train_imgs[rnd_num])  # randomly select a image from train set

    train_avg = np.mean(train_imgs, axis=0)  # average the train set
    plt.axes().set_title("Average face image from train set")
    plot_img(train_avg)


    #======================================step2========================================
    # Calculate eigenvalue
    sample_n, img_row, img_col = train_imgs.shape
    train_flat = train_imgs.reshape(sample_n, img_row * img_col)     # flatten 2-dimensional image to 1-dimension
    print "y=", train_flat.shape                                # N by D

    b_mean_vec = np.mean(train_flat, axis=0)                    # get mean from the flattened train set
    print "b=", b_mean_vec.shape                                # D

    z_coordinate = np.array([train_flat[ith] - b_mean_vec for ith in range(sample_n)]).T     # get the coordinates z
    print "z=", z_coordinate.shape                              # D by N, why conflicts with theory???

    x_svd = (sample_n ** (-0.5)) * z_coordinate
    print "x=", x_svd.shape

    sgl_vec_lft, sgl_val, sgl_vec_rgt_t = linalg.svd(x_svd,full_matrices=False)     # use thin SVD to get singular values and vectors
    print "left singular=", sgl_vec_lft.shape
    print "singular_val=", sgl_val.shape
    print "right_singular=", sgl_vec_rgt_t.shape

    eig_val = sgl_val ** 2                                      # get eigenvalues from singular values
    print "eig_val=", eig_val.shape

    plot_gen(range(0, len(eig_val)), eig_val, "index", "eigenvalue", "Screen Plot")


    # plot fraction of variance explained
    eig_cumsum = np.cumsum(eig_val)                            # get cumulative sum for eigenvalues
    r_frac_expl = np.array([eig_cumsum[ith] / eig_cumsum.max() for ith in range(sample_n)])      # get the fraction of variance explained
    plot_gen(range(0, r_frac_expl.size), r_frac_expl, "index", "cumulative sum", "Fraction of Variance Explained")


    # Display first 10 principle components
    eig_vec = sgl_vec_lft                         # D by N. eigenvector of XX^T equals to left singular vector of X
    print "eig_vec=", eig_vec.shape

    ten = 10                                      # The question requires 10 principle components
    w_basis_vec_ten = get_w_basis(ten, eig_vec)   # D by C
    print "w_10=", w_basis_vec_ten.shape

    w_revert_ten = w_basis_vec_ten.T.reshape(ten, img_row, img_col)   # revert the flattened image to 2-d
    print "w_rev=", w_revert_ten.shape

    for i, v in enumerate(w_revert_ten):
        plt.axes().set_title("%dth principal component"%i)
        plot_img(v)


    # Select a subspace dimensiona where at least 90% of variances are explained.
    conf_var = 0.9
    for i, v in enumerate(eig_cumsum):
        if (
                v / eig_cumsum.max() >= 0.9):  # when cumulative variance reaches 90%, loop exits and we get the subspace dimension
            sub_dim_opt = i + 1
            break
    print "subspace dimensionality selected is: ", sub_dim_opt


    #======================================step3========================================
    # Calculate reconstructed y and the differece between actual y and it
    w_optimal = get_w_basis(sub_dim_opt, eig_vec)        # w: d by c; get the optimal basis vectors.
    print "w=", w_optimal.shape

    x = np.dot(w_optimal.T, z_coordinate)   # w: d by c; z: d by n;
    print "x=", x.shape                     # x: c by n

    y_recon = (b_mean_vec.reshape(img_row * img_col, 1) + np.dot(w_optimal, x)).T    # b: d by n  w: d by c x: c by n
    print "y_rec=", y_recon.shape

    y = train_flat
    print "y=", y.shape

    diff_y = np.absolute(y - y_recon)      # difference between actual y and reconstructed y

    # Display 5 randomly selected face images and their squared reconstruction errors respectively
    for i in range(0, 5):
        rnd_index = random.randint(0,len(diff_y))
        print "face index:", rnd_index
        sqr_recon_err = nla.norm(diff_y[rnd_index]) ** 2
        print "sqr_recon_err:", sqr_recon_err
        plt.axes().set_title("%dth random original face image"%i)
        plot_img(y[rnd_index].reshape(img_row, img_col), True)
        plt.axes().set_title("%dth random reconstructed face image"%i)
        plot_img(y_recon[rnd_index].reshape(img_row, img_col), True)
        plt.axes().set_title("Squared Reconstruction Error: %.2f" %sqr_recon_err)
        plot_img(diff_y[rnd_index].reshape(img_row, img_col))

    sre1 = [nla.norm(diff_y[ith]) ** 2 for ith in range(0, sample_n)]      # store squared reconstruction errors in a list for each data point

    # Re-do everything for nonface_image; but use the same basis vectors w
    inputdata2 = scio.loadmat("nonface_images.mat")
    train_imgs2 = inputdata2["nonface_imgs"]

    # Use basis vector w and mean vector inherited from face image.
    w_optimal2 = w_optimal
    b_mean_vec2 = b_mean_vec

    # Re-calculate the rest of params.
    sample_n2, img_row2, img_col2 = train_imgs2.shape
    train_flat2 = train_imgs2.reshape(sample_n2, img_row2 * img_col2)
    z_coordinate2 = np.array([train_flat2[ith] - b_mean_vec2 for ith in range(sample_n2)]).T
    x2 = np.dot(w_optimal2.T, z_coordinate2)
    y_recon2 = (b_mean_vec2.reshape(img_row2 * img_col2, 1) + np.dot(w_optimal2, x2)).T
    y2 = train_flat2
    diff_y2 = np.absolute(y2 - y_recon2)

    for i in range(0, 5):
        rnd_index = random.randint(0, len(diff_y2))
        print "nonface index:", rnd_index
        sqr_recon_err = nla.norm(diff_y2[rnd_index]) ** 2
        print "sqr_recon_err:", sqr_recon_err
        plt.axes().set_title("%dth random original nonface image" % i)
        plot_img(y2[rnd_index].reshape(img_row2, img_col2), True)
        plt.axes().set_title("%dth random reconstructed nonface image" % i)
        plot_img(y_recon2[rnd_index].reshape(img_row2, img_col2), True)
        plt.axes().set_title("Squared Reconstruction Error: %.2f" % sqr_recon_err)
        plot_img(diff_y2[rnd_index].reshape(img_row2, img_col2))

    sre2 = [nla.norm(diff_y2[ith]) ** 2 for ith in range(0, sample_n2)]

    plt.plot(range(0, len(sre1)), sre1, '.r', label="face")
    plt.plot(range(0, len(sre2)), sre2, '.b', label="nonface")
    plt.axes().set_title("Squared Reconstruction Error Comparison")
    plt.legend(loc='upper right')
    plt.show()
    plt.close()


    # Find the optimal threshold that minimizes the rate of misclassification for face image and nonface image
    threshold = 0
    misclsf_total = len(sre1) + len(sre2)
    for i in range(2500, 12500):
        misclsf_face_index = [ith for ith in range(0, len(sre1)) if nla.norm(diff_y[ith]) ** 2 > i]
        misclsf_nonface_index = [ith for ith in range(0, len(sre2)) if nla.norm(diff_y2[ith]) ** 2 < i]
        if(len(misclsf_face_index) + len(misclsf_nonface_index) < misclsf_total):
            misclsf_total = len(misclsf_face_index) + len(misclsf_nonface_index)
            threshold = i
            min_misclsf_face_index = misclsf_face_index
            min_misclsf_nonface_index = misclsf_nonface_index
    print "The optimal threshold is: ", threshold
    print "The minimum misclassfied images count is:", misclsf_total


    # Display the misclassified images, totally 14 images to be displayed
    for i, v in enumerate(min_misclsf_face_index):
        print "face ", v
        sqr_recon_err = nla.norm(diff_y[v]) ** 2
        print "sqr_recon_err:", sqr_recon_err
        plt.axes().set_title("%dth misclassified original face image"%i)
        plot_img(y[v].reshape(img_row, img_col), True)
        plt.axes().set_title("%dth misclassified reconstructed face image"%i)
        plot_img(y_recon[v].reshape(img_row, img_col), True)
        plt.axes().set_title("Squared Reconstruction Error: %.2f" %sqr_recon_err)
        plot_img(diff_y[v].reshape(img_row, img_col))
    for i, v in enumerate(min_misclsf_nonface_index):
        print "nonface ", v
        sqr_recon_err = nla.norm(diff_y2[v]) ** 2
        print "sqr_recon_err:", sqr_recon_err
        plt.axes().set_title("%dth misclassified original nonface image"%i)
        plot_img(y2[v].reshape(img_row2, img_col2), True)
        plt.axes().set_title("%dth misclassified reconstructed nonface image"%i)
        plot_img(y_recon2[v].reshape(img_row2, img_col2), True)
        plt.axes().set_title("Squared Reconstruction Error: %.2f" %sqr_recon_err)
        plot_img(diff_y2[v].reshape(img_row2, img_col2))


    #======================================step4========================================
    # Plot the subspace coefficients for all samples
    c4 = 4
    w4 = get_w_basis(c4, eig_vec)
    z4 = z_coordinate
    x4 = np.dot(w4.T, z4)

    # Take step = 10 since the train set is constructed in a way that every 10 imgs belong to the same person
    for i_sub in range(0, 30, 10):
        x_sub = np.array([x4.T[ith] for ith in range(0, sample_n) if train_ids[ith] == train_ids[i_sub]]).T
        x_sub_rest = np.array([x4.T[ith] for ith in range(0, sample_n) if train_ids[ith] != train_ids[i_sub]]).T
        # Take the dimension from (0, 1), (1, 2) and (2, 3)
        for dim in range(0, 3):
            plt.plot(x_sub_rest[dim], x_sub_rest[dim + 1], 'y.')
            plt.plot(x_sub[dim], x_sub[dim + 1], 'k.')
            plt.axes().set_title("Subject %d"%(i_sub) + ", Subspace Dimensions (%d"%(dim) + ", %d)"%(dim+1))
            plt.show()
            plt.close()


    # Use KNN where k=1 to do classification
    c10 = 10
    w10_train = get_w_basis(c10, eig_vec)
    z10_train = z_coordinate
    x10_train = np.dot(w10_train.T, z10_train)

    # Use basis vector w and mean vector inherited from train set
    w10_test = w10_train
    b10_test = b_mean_vec

    # Calculate the subspace coefficient x10_test for test set
    n_test, row_test, col_test = test_imgs.shape
    test_flat = test_imgs.reshape(n_test, row_test * col_test)
    z_test = np.array([test_flat[ith] - b10_test for ith in range(n_test)]).T
    x10_test = np.dot(w10_test.T, z_test)

    # implement the 1NN algorithm
    result_actual = test_ids.reshape(test_ids.size).tolist()  # list holds actual labels for test set
    result_clsf = []  # list holds classfied labels for test set
    test_train_index_map = []  # list holds the index of train image which a certain test image is classified to

    for i in range(0, n_test):
        x_test = x10_test.T[i]  # loop for each item in test set
        dists = []  # a list holds euclidean distances between a specified item in test set and all items in train set
        for j in range(0, sample_n):
            dist = distance.euclidean(x10_test.T[i], x10_train.T[j])  # calculate the euclidean distance
            dists.append(dist)  # store it in the distance list
        min_dist = min(dists)  # find the minimal distance from all
        min_index = dists.index(min_dist)  # get the index for the minimal distance
        test_train_index_map.append(min_index)  # maps the test id to train id
        label_clsf = result_actual[min_index]  # get the corresponding label for the index from train set
        result_clsf.append(label_clsf)  # store the label in the classification list

    misclsf_count = 0
    misclsf_indexes = []
    for i in range(0, len(result_actual)):
        if result_actual[i] != result_clsf[i]:
            misclsf_count += 1  # misclassification plus one if labels does not match
            misclsf_indexes.append(i)
    accuracy = (len(result_actual) - misclsf_count) / float(len(result_actual))

    print "count of misclassifying is: ", misclsf_count
    print "accuracy= ", accuracy

    # plot misclassified test original & reconstructed image, train original & reconstructed image.
    test_recon = (b10_test.reshape(row_test * col_test, 1) + np.dot(w10_test, x10_test)).T
    test_diff = np.absolute(test_flat - test_recon)

    for i in range(0, len(misclsf_indexes)):
        index = misclsf_indexes[i]
        plt.axes().set_title("%dth misclassified original test face image" % i)
        plot_img(test_flat[index].reshape(row_test, col_test), True)
        plt.axes().set_title("%dth reconstructed misclassified test face image" % i)
        plot_img(test_recon[index].reshape(row_test, col_test), True)
        plt.axes().set_title("%dth 1NN train face image" % i)
        plot_img(train_flat[test_train_index_map[index]].reshape(row_test, col_test), True)
        plt.axes().set_title("%dth reconstructed 1NN train face image" % i)
        plot_img(y_recon[test_train_index_map[index]].reshape(row_test, col_test), True)


