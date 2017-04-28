from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


def poly_phi(x, power):
    return x ** power


# matrix transpose
def t(mat):
    return np.matrix(mat).transpose()


# matrix inverse
def inv(mat):
    return np.linalg.inv(mat)


# matrix outer product
def mx(mata, matb):
    return np.dot(mata, matb)


def phi(x_input, d_size):
    phi_tmp = [[] for i in range(len(x_input))]
    for i in range(len(x_input)):
        for d in range(d_size):
            phi_tmp[i].append(poly_phi(x_input[i][0], d))
    return np.matrix(phi_tmp)  # convert phi to numpy matrix 100 * 20


def mle_w(phi, ou1):
    return mx(mx(inv(mx(t(phi), phi)), t(phi)), ou1)


def map_w(phi, ou1, lumda, d):
    mat_lumda = np.identity(d) * lumda
    return mx(mx(inv(mat_lumda + mx(t(phi), phi)), t(phi)), ou1)


def plot_mse(s, title):
    plt.plot(range(1, len(s) + 1), s, '+')
    plt.axes().set_xlabel("D")
    plt.axes().set_ylabel("mean squared error")
    plt.axes().set_title(title)
    plt.show()
    plt.close()


def plot_step_4(in1, in_mle, in_map, ou1, f_mle, f_map, d):
    plt.plot(in1, ou1, '+g', label="Data")
    plt.plot(in_mle, f_mle, '-r', label="ML")
    plt.plot(in_map, f_map, '-b', label="MAP")
    plt.axes().set_title("D= %d " % d)
    plt.axes().set_xlim(-1.5, 1.5)
    plt.legend(loc='upper right')
    plt.show()
    plt.close()


# get estimate F(x)
def get_estimate(x_train, y_train, d_size, lumda=None, x_val=None):
    # create phi matrix
    fx = []  # regular list 20 * 100

    for big_d in range(1, d_size + 1):

        # derive phi from train set
        phi_train = phi(x_train, big_d)

        # derive weight w matrix, if lumda is none, use mle; otherwise, use map
        if lumda is None:
            w = mle_w(phi_train, y_train)  # numpy matrix 20 * 1
        else:
            w = map_w(phi_train, y_train, lumda, big_d)

        if x_val is None:
            phi_times_w_tolist = t(mx(phi_train, w)).tolist()  # regular list 1 * 100
            fx.append(phi_times_w_tolist[0])
        else:
            # derive phi from val set
            phi_val = phi(x_val, big_d)

            # multiply phi_derived_from_val_set with w_derived_from_train_set and convert to regular then add to fx
            phi_times_w_tolist = t(mx(phi_val, w)).tolist()  # regular list 1 * 100
            fx.append(phi_times_w_tolist[0])

    fx_result = np.array(fx)  # convert to numpy ndarray 20 * 100

    return fx_result


def get_mean_squared_error(fx_result, y_val, d_size, lumda=None):
    # derive mean squared error s
    s = []
    for d in range(d_size):
        fx_d = fx_result[d].reshape(fx_result[d].size, 1)  # numpy ndarray 100 * 1
        pow_diff = (y_val - fx_d) ** 2
        s.append(pow_diff.sum() / pow_diff.size)

    s_result = np.array(s)  # numpy ndarray 20 * 1
    return s_result


#=========================================Code start from here==========================================================


if __name__ == "__main__":
    filename_in1 = 'dataset1_inputs.txt'
    filename_ou1 = 'dataset1_outputs.txt'
    filename_in2 = 'dataset2_inputs.txt'
    filename_ou2 = 'dataset2_outputs.txt'

    # do not forget to RESHAPE when creating ndarray
    input_data = np.loadtxt(filename_in1)
    output_data = np.loadtxt(filename_ou1)

    # ======================================step 1 plot points=====================================
    plt.plot(input_data, output_data, '+g')
    plt.show()

    d_basis_functions = 20
    lumda = 0.001
    in1 = input_data.reshape(len(input_data), 1)  # numpy ndarray 100 * 1
    ou1 = output_data.reshape(len(output_data), 1)  # numpy ndarray 100 * 1

    # ===========================step 2 print mean squared error on trainning data using mle to evaluate w=============================
    fx_mle_train = get_estimate(in1, ou1, d_basis_functions)  # w/o param lumda, use mle
    s_mle = get_mean_squared_error(fx_mle_train, ou1, d_basis_functions)
    plot_mse(s_mle, "step 2 mle")

    # ===========================step 3 instead, use map to evaluate w=============================================
    fx_map_train = get_estimate(in1, ou1, d_basis_functions, lumda=lumda)  # w/ param lumda, use map
    s_map = get_mean_squared_error(fx_map_train, ou1, d_basis_functions, lumda=lumda)
    plot_mse(s_map, "step 3 map")

    # =============================step 4 plot original data, mle, map====================================
    d_range = [5, 10, 15, 20]
    # generate random validation set in range(-1, 1) (slightly widening) of the population 1000.
    x_ran = 2.01 * np.random.rand(1000, 1) - 1.005
    fx_mle_val = get_estimate(in1, ou1, d_basis_functions, x_val=x_ran)
    fx_map_val = get_estimate(in1, ou1, d_basis_functions, lumda=lumda, x_val=x_ran)

    for i, d in enumerate(d_range):
        data_dic = dict(zip(in1.reshape(1, len(in1))[0], ou1.reshape(1, len(ou1))[0]))
        mle_dic = dict(zip(x_ran.reshape(1, len(x_ran))[0], fx_mle_val[d - 1].reshape(1, len(fx_mle_val[d - 1]))[0]))
        map_dic = dict(zip(x_ran.reshape(1, len(x_ran))[0], fx_map_val[d - 1].reshape(1, len(fx_map_val[d - 1]))[0]))

        x, y_data = zip(*sorted(data_dic.items()))
        x_mle, y_mle = zip(*sorted(mle_dic.items()))
        x_map, y_map = zip(*sorted(map_dic.items()))

        plot_step_4(x, x_mle, x_map, y_data, y_mle, y_map, d)

    # =============================step 5 10-fold cross validation==============================
    fold_size = 10
    lumda = 0.001
    d_size = 20
    # trainset_split = np.split(input_data, fold_size)
    mse_list = []

    # get random
    np.random.seed(10)
    data_perm = np.random.permutation(len(input_data))
    index_split = np.split(data_perm, fold_size)

    for i in range(fold_size):
        train_index_raw = [index_split[ith] for ith in range(fold_size) if ith != i]
        val_index_raw = [index_split[ith] for ith in range(fold_size) if ith == i]
        train_index = np.concatenate(train_index_raw).ravel()
        val_index = np.concatenate(val_index_raw).ravel()
        train_x = np.array([input_data[ith] for ith in train_index]).reshape(90, 1)
        train_y = np.array([output_data[ith] for ith in train_index]).reshape(90, 1)
        val_x = np.array([input_data[ith] for ith in val_index]).reshape(10, 1)
        val_y = np.array([output_data[ith] for ith in val_index]).reshape(10, 1)

        y_est = get_estimate(train_x, train_y, d_size, lumda, val_x)
        mse_list.append(get_mean_squared_error(y_est, val_y, d_size, lumda))

    mse_avg_list = [np.average(i) for i in zip(*mse_list)]
    plot_mse(mse_avg_list, "step 5 mse average list")

    # =============================step 6 plot the points================================
    i2_data = np.loadtxt(filename_in2)
    o2_data = np.loadtxt(filename_ou2)

    plt.plot(i2_data, o2_data, '+g')
    plt.show()
    plt.close()

    # =========================step 7 bayesian regression========================
    in2 = i2_data.reshape(len(i2_data), 1)
    ou2 = o2_data.reshape(len(o2_data), 1)
    d_basis_functions = 13
    sigma = 1.5
    tao = 1000

    phi_in2 = phi(in2, d_basis_functions)  # numpy matrix 100 * 13
    sigma_w = inv(pow(tao, -2) * np.identity(d_basis_functions) + pow(sigma, -2) * t(phi_in2) * phi_in2)  # numpy matrix 13 * 13
    miu_w = pow(sigma, -2) * sigma_w * t(phi_in2) * ou2  # numpy matrix 13 * 1

    boundary = 1.1
    step = 0.001
    x_input = np.arange(-boundary, boundary, step).reshape(int(2 * boundary / step), 1)
    y_est = []
    y_plus = []
    y_minus = []
    phi_x_rnd = phi(x_input, d_basis_functions)

    for i in range(x_input.size):
        miu_tmp = (phi_x_rnd[i] * miu_w).item()  # scalar
        sigma2_tmp = sigma ** 2 + (phi_x_rnd[i] * sigma_w * t(phi_x_rnd[i])).item()  # scalar
        sigma_tmp = sigma2_tmp ** 0.5  # scalar
        y_est.append(miu_tmp)
        y_plus.append(miu_tmp + sigma_tmp)
        y_minus.append(miu_tmp - sigma_tmp)

    plt.plot(in2, ou2, 'g+')
    plt.plot(x_input, y_est, 'b')
    plt.plot(x_input, y_plus, 'g--')
    plt.plot(x_input, y_minus, 'g--')
    plt.axes().set_title("D = 13, x [-1.1, 1.1]")
    plt.show()
    plt.close()
