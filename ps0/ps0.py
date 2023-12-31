import copy
import numpy as np
from matplotlib import pyplot as plt

# Task 1.1
def mult_scalar(A, c):
    """
    Returns a new matrix created by multiplying elements of matrix A by a scalar c.
    Note
    ----
    Do not use numpy for this question.
    """
    # TODO: add your solution here and remove `raise NotImplementedError`
    B = copy.deepcopy(A)
    for i in range(len(B)):
        for j in range(len(B[0])): # Assuming that it is non empty
            B[i][j] = c * B[i][j]
    return B

# Test case for Task 1.1
def test_11():
    A = [[5, 7, 9], [1, 4, 3]]
    A_copy = copy.deepcopy(A)

    actual = mult_scalar(A_copy, 2)
    expected = [[10, 14, 18], [2, 8, 6]]
    assert(A == A_copy) # check for aliasing
    assert(actual == expected)

    A2 = [[6, 5, 5], [8, 6, 0], [1, 5, 8]]
    A2_copy = copy.deepcopy(A2)

    actual2 = mult_scalar(A2_copy, 5)
    expected2 = [[30, 25, 25], [40, 30, 0], [5, 25, 40]]
    assert(A2 == A2_copy) # check for aliasing
    assert(actual2 == expected2)

#test_11()

# Task 1.2
def add_matrices(A, B):
    """
    Returns a new matrix that is the result of adding matrix B to matrix A.
    Note
    ----
    Do not use numpy for this question.
    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise Exception('A and B cannot be added as they have incompatible dimensions!')
    C = copy.deepcopy(A)
    # TODO: add your solution here and remove `raise NotImplementedError`
    for i in range(len(C)):
        for j in range(len(C[0])):
            C[i][j] = A[i][j] + B[i][j]
    return C

# Test case for Task 1.2
def test_12():
    A = [[5, 7, 9], [1, 4, 3]]
    B = [[2, 3, 4], [5, 6, 7]]
    A_copy = copy.deepcopy(A)
    B_copy = copy.deepcopy(B)

    actual = add_matrices(A_copy, B_copy)
    expected = [[7, 10, 13], [6, 10, 10]]
    assert(A == A_copy) # check for aliasing
    assert(B == B_copy) # check for aliasing
    assert(actual == expected)

#test_12()


# Task 1.3
def transpose_matrix(A):
    """
    Returns a new matrix that is the transpose of matrix A.
    Note
    ----
    Do not use numpy for this question.
    """
    # TODO: add your solution here and remove `raise NotImplementedError`
    B = []
    for j in range(len(A[0])):
        B.append([])
        for i in range(len(A)):
            B[j].append(A[i][j])
    return B


# Test case for Task 1.3
def test_13():
    A = [[5, 7, 9], [1, 4, 3]]
    A_copy = copy.deepcopy(A)

    actual = transpose_matrix(A_copy)
    expected = [[5, 1], [7, 4], [9, 3]]
    assert(A == A_copy)
    assert(actual == expected)

#test_13()


# Task 1.4
def mult_matrices(A, B):
    """
    Multiplies matrix A by matrix B, giving AB.
    Note
    ----
    Do not use numpy for this question.
    """
    if len(A[0]) != len(B):
        raise Exception('Incompatible dimensions for matrix multiplication of A and B')
    
    # TODO: add your solution here and remove `raise NotImplementedError`
    C = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            row_A = A[i]
            col_B = [x[j] for x in B]
            product = list(map(lambda a, b : a * b, row_A, col_B))
            C[i][j] = sum(product)
    return C
            

# Test Case for Task 1.4
def test_14():
    A = [[5, 7, 9], [1, 4, 3]]
    B = [[2, 5], [3, 6], [4, 7]]
    A_copy = copy.deepcopy(A)
    B_copy = copy.deepcopy(B)

    actual = mult_matrices(A, B)
    expected = [[67, 130], [26, 50]]
    assert(A == A_copy and B == B_copy)
    assert(actual == expected)

    A2 = [[-13, -10], [-24, 14]]
    B2 = [[1, 0], [0, 1]]
    A2_copy = copy.deepcopy(A2)
    B2_copy = copy.deepcopy(B2)

    actual2 = mult_matrices(A2, B2)
    expected2 = [[-13, -10], [-24, 14]]
    assert(A2 == A2_copy and B2 == B2_copy)
    assert(actual2 == expected2)

#test_14()


# Task 1.5
import copy

def invert_matrix(A):
    """
    Returns the inverse of matrix A, if it exists; otherwise, returns False
    """
    if len(A[0]) != len(A):
        return False

    B = copy.deepcopy(A)
    length = len(A)
    
    mat = [[0 for i in range(length)] for j in range(length)]

    def swap(i, j):
        B[i], B[j] = B[j], B[i]
        mat[i], mat[j] = mat[j], mat[i]

    def scale(i):
        multiple = B[i][i]
        for k in range(length):
            B[i][k] /= multiple
            mat[i][k] /= multiple

    def add_multiples(i):
        for k in range(length):
            if k == i:
                continue
            if B[k][i] != 0:
                c = -B[k][i]
                for l in range(length):
                    B[k][l] += c * B[i][l]
                    mat[k][l] += c * mat[i][l]
                
    # Initialize identity matrix    
    for x in range(length):
        mat[x][x] = 1
        
    for i in range(length):  # For each pivot position
        found_pivot = False
        for j in range(i, length):  # Search for a non-zero pivot
            if B[j][i] != 0:
                swap(i, j)  
                scale(i)   
                add_multiples(i) 
                found_pivot = True
                break

        if not found_pivot: 
            return False
 
    return mat

                
    

# Test case for Task 1.5
def test_15():
    A = [[1, 0 ,0], [0, 1, 0], [0, -4, 1]]
    A_copy = copy.deepcopy(A)

    actual = invert_matrix(A)
    expected = [[1, 0 ,0], [0, 1, 0], [0, 4, 1]]
    assert(A == A_copy)
    for i in range(len(A)):
        for j in range(len(A[0])):
            assert(round(actual[i][j], 11) == round(expected[i][j], 11))
            
            
    A2 = [[0, 3, 2], [0, 0, 1], [1, 5, 3]]
    A2_copy = copy.deepcopy(A2)

    actual2 = invert_matrix(A2)
    expected2 = [[-5/3, 1/3 ,1], [1/3, -2/3, 0], [0, 1, 0]]
    assert(A2 == A2_copy)
    for i in range(len(A2)):
        for j in range(len(A2[0])):
            assert(round(actual2[i][j], 11) == round(expected2[i][j], 11))
            
            
    A3 = [[1, 0, 0], [0, 1, 0], [0, 0, 0]] # non-invertible matrix
    actual3 = invert_matrix(A3)
    expected3 = False
    assert actual3 == expected3

#test_15()


from prepare_data import *

# Example on loading the data for Task 2
from prepare_data import * # loads the `get_...` helper funtions

df = get_data()
cases_cumulative = get_n_cases_cumulative(df)
deaths_cumulative = get_n_deaths_cumulative(df)
healthcare_spending = get_healthcare_spending(df)
mask_prices = get_mask_prices(healthcare_spending.shape[1])
stringency_values = get_stringency_values(df)
cases_top_cumulative = get_n_cases_top_cumulative(df)

# Task 2.1
def compute_death_rate_first_n_days(n, cases_cumulative, deaths_cumulative):
    '''
    Computes the average number of deaths recorded for every confirmed case
    that is recorded from the first day to the nth day (inclusive).
    Parameters
    ----------
    n: int
        How many days of data to return in the final array.
    cases_cumulative: np.ndarray
        2D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the cumulative number of
        confirmed cases in that country, i.e. the ith row of `cases_cumulative`
        contains the data of the ith country, and the (i, j) entry of
        `cases_cumulative` is the cumulative number of confirmed cases on the
        (j + 1)th day in the ith country.
    deaths_cumulative: np.ndarray
        2D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the cumulative number of
        confirmed deaths (as a result of COVID-19) in that country, i.e. the ith
        row of `deaths_cumulative` contains the data of the ith country, and
        the (i, j) entry of `deaths_cumulative` is the cumulative number of
        confirmed deaths on the (j + 1)th day in the ith country.
    
    Returns
    -------
    Average number of deaths recorded for every confirmed case from the first day
    to the nth day (inclusive) for each country as a 1D `ndarray` such that the
    entry in the ith row corresponds to the death rate in the ith country as
    represented in `cases_cumulative` and `deaths_cumulative`.
    Note
    ----
    `cases_cumulative` and `deaths_cumulative` are such that the ith row in the 
    former and that in the latter contain data of the same country. In addition,
    if there are no confirmed cases for a particular country, the expected death
    rate for that country should be zero. (Hint: to deal with NaN look at
    `np.nan_to_num`)
    '''
    
    # TODO: add your solution here and remove `raise NotImplementedError`
    cases = cases_cumulative[:, n-1:n]
    deaths = deaths_cumulative[:, n-1:n]
    cleaned_rates = np.nan_to_num((deaths / cases), nan=0)
    return np.mean(cleaned_rates, axis=1)

# Test case for Task 2.1
def test_21():
    n_cases_cumulative = cases_cumulative[:3, :] #Using data from CSV. Make sure to run relevant cell above
    n_deaths_cumulative = deaths_cumulative[:3, :]
    expected = np.array([0.0337837838, 0.0562347188, 0.1410564226])
    np.testing.assert_allclose(compute_death_rate_first_n_days(100, n_cases_cumulative, n_deaths_cumulative), expected)

    sample_cumulative = np.array([[1,2,3,4,8,8,10,10,10,10], [1,2,3,4,8,8,10,10,10,10]])
    sample_death = np.array([[0,0,0,1,2,2,2,2,5,5], [0,0,0,1,2,2,2,2,5,5]])

    expected2 = np.array([0.5, 0.5])
    assert(np.all(compute_death_rate_first_n_days(10, sample_cumulative, sample_death) == expected2))

    sample_cumulative2 = np.array([[1,2,3,4,8,8,10,10,10,10]])
    sample_death2 = np.array([[0,0,0,1,2,2,2,2,5,5]])

    expected3 = np.array([0.5])
    assert(compute_death_rate_first_n_days(10, sample_cumulative2, sample_death2) == expected3)
    expected4 = np.array([0.25])
    assert(compute_death_rate_first_n_days(5, sample_cumulative2, sample_death2) == expected4)

#test_21()

# Task 2.2
def compute_increase_in_cases(n, cases_cumulative):
    '''
    Computes the daily increase in confirmed cases for each country for the first n days, starting
    from the first day.
    Parameters
    ----------    
    n: int
        How many days of data to return in the final array. If the input data has fewer
        than n days of data then we just return whatever we have for each country up to n. 
    cases_cumulative: np.ndarray
        2D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the cumulative number of
        confirmed cases in that country, i.e. the ith row of `cases_cumulative`
        contains the data of the ith country, and the (i, j) entry of
        `cases_cumulative` is the cumulative number of confirmed cases on the
        (j + 1)th day in the ith country.
    
    Returns
    -------
    Daily increase in cases for each country as a 2D `ndarray` such that the (i, j)
    entry corresponds to the increase in confirmed cases in the ith country on
    the (j + 1)th day, where j is non-negative.
    Note
    ----
    The number of cases on the zeroth day is assumed to be 0, and we want to
    compute the daily increase in cases starting from the first day.
    '''
    
    # TODO: add your solution here and remove `raise NotImplementedError`
    increase = np.diff(cases_cumulative)
    first_days = cases_cumulative[:, 0]
    return np.insert(increase, 0, first_days, axis=1)[:,:n]
    

# Test case for Task 2.2
def test_22():#  
    cases_cumulative = np.zeros((100, 20))
    cases_cumulative[:, :] = np.arange(1, 21)
    actual = compute_increase_in_cases(100, cases_cumulative)
    assert(np.all(actual == np.ones((100, 20))))

    sample_cumulative = np.array([[1,2,3,4,8,8,10,10,10,10],[1,1,3,5,8,10,15,20,25,30]])
    expected = np.array([[1, 1, 1, 1, 4.], [1, 0, 2, 2, 3]])
    assert(np.all(compute_increase_in_cases(5,sample_cumulative) == expected))

    expected2 = np.array([[1, 1, 1, 1, 4, 0, 2, 0, 0, 0],[1, 0, 2, 2, 3, 2, 5, 5, 5, 5]])
    assert(np.all(compute_increase_in_cases(10,sample_cumulative) == expected2))
    assert(np.all(compute_increase_in_cases(20,sample_cumulative) == expected2))

    sample_cumulative2 = np.array([[51764, 51848, 52007, 52147, 52330, 52330],\
                                [55755, 56254, 56572, 57146, 57727, 58316],\
                                [97857, 98249, 98631, 98988, 99311, 99610]])
    expected3 = np.array([\
                [51764, 84, 159, 140, 183, 0],\
                [55755, 499, 318, 574, 581, 589],\
                [97857, 392, 382, 357, 323, 299]])
    assert(np.all(compute_increase_in_cases(6,sample_cumulative2) == expected3))

#test_22()



# Task 2.3
def find_max_increase_in_cases(n_cases_increase):
    '''
    Finds the maximum daily increase in confirmed cases for each country.
    Parameters
    ----------
    n_cases_increase: np.ndarray
        2D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the daily increase in the
        number of confirmed cases in that country, i.e. the ith row of 
        `n_cases_increase` contains the data of the ith country, and the (i, j) entry of
        `n_cases_increase` is the daily increase in the number of confirmed cases on the
        (j + 1)th day in the ith country.
    
    Returns
    -------
    Maximum daily increase in cases for each country as a 1D `ndarray` such that the
    ith entry corresponds to the increase in confirmed cases in the ith country as
    represented in `n_cases_increase`.
    '''
    
    # TODO: add your solution here and remove `raise NotImplementedError`
    return np.amax(n_cases_increase, axis=1)


# Test case for Task 2.3
def test_23():
    n_cases_increase = np.ones((100, 20))
    actual = find_max_increase_in_cases(n_cases_increase)
    expected = np.ones(100)
    assert(np.all(actual == expected))

    sample_increase = np.array([[1,2,3,4,8,8,10,10,10,10],[1,1,3,5,8,10,15,20,25,30]])
    expected2 = np.array([10, 30]) # max of [1,2,3,4,8,8,10,10,10,10] => 10, max of [1,1,3,5,8,10,15,20,25,30] => 30
    assert(np.all(find_max_increase_in_cases(sample_increase) == expected2))

    sample_increase2 = np.array([\
                [51764, 84, 159, 140, 183, 0],\
                [55755, 499, 318, 574, 581, 589],\
                [97857, 392, 382, 357, 323, 299]])
    expected3 = np.array([51764, 55755, 97857])
    assert(np.all(find_max_increase_in_cases(sample_increase2) == expected3))

    n_cases_increase2 = compute_increase_in_cases(cases_top_cumulative.shape[1], cases_top_cumulative)
    expected4 = np.array([ 68699.,  97894., 258110.])
    assert(np.all(find_max_increase_in_cases(n_cases_increase2) == expected4))

#test_23()


# Task 2.4
def compute_n_masks_purchaseable(healthcare_spending, mask_prices):
    '''
    Computes the total number of masks that each country can purchase if she
    spends all her emergency healthcare spending on masks.
    Parameters
    ----------
    healthcare_spending: np.ndarray
        2D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the emergency healthcare
        spending made by that country, i.e. the ith row of `healthcare_spending`
        contains the data of the ith country, and the (i, j) entry of
        `healthcare_spending` is the amount which the ith country spent on healthcare
        on (j + 1)th day.
    mask_prices: np.ndarray
        1D `ndarray` such that the jth entry represents the cost of 100 masks on the
        (j + 1)th day.
    
    Returns
    -------
    Total number of masks which each country can purchase as a 1D `ndarray` such
    that the ith entry corresponds to the total number of masks purchaseable by the
    ith country as represented in `healthcare_spending`.
    Note
    ----
    The masks can only be bought in batches of 100s.
    '''
    
    # TODO: add your solution here and remove `raise NotImplementedError`
    masks_each_day = np.floor(healthcare_spending / mask_prices) * 100
    masks = masks_each_day.sum(axis=1)
    return masks

# Test case for Task 2.4
def test_24():
    prices_constant = np.ones(5)
    healthcare_spending_constant = np.ones((7, 5))
    actual = compute_n_masks_purchaseable(healthcare_spending_constant, prices_constant)
    expected = np.ones(7) * 500
    assert(np.all(actual == expected))

    healthcare_spending1 = healthcare_spending[:3, :]  #Using data from CSV
    expected2 = [3068779300, 378333500, 6208321700]
    assert(np.all(compute_n_masks_purchaseable(healthcare_spending1, mask_prices)==expected2))

    healthcare_spending2 = np.array([[0, 100, 0], [100, 0, 200]])
    mask_prices2 = np.array([4, 3, 20])
    expected3 = np.array([3300, 3500])
    assert(np.all(compute_n_masks_purchaseable(healthcare_spending2, mask_prices2)==expected3))

#test_24()

# Task 2.5
def compute_stringency_index(stringency_values):
    '''
    Computes the daily stringency index for each country.
    Parameters
    ----------
    stringency_values: np.ndarray
        3D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the stringency values as a
        vector. To be specific, on each day, there are four different stringency
        values for 'school closing', 'workplace closing', 'stay at home requirements'
        and 'international travel controls', respectively. For instance, the (i, j, 0)
        entry represents the `school closing` stringency value for the ith country
        on the (j + 1)th day.
    
    Returns
    -------
    Daily stringency index for each country as a 2D `ndarray` such that the (i, j)
    entry corresponds to the stringency index in the ith country on the (j + 1)th
    day.
    In this case, we shall assume that 'stay at home requirements' is the most
    restrictive regulation among the other regulations, 'international travel
    controls' is more restrictive than 'school closing' and 'workplace closing',
    and 'school closing' and 'workplace closing' are equally restrictive. Thus,
    to compute the stringency index, we shall weigh each stringency value by 1,
    1, 3 and 2 for 'school closing', 'workplace closing', 'stay at home
    requirements' and 'international travel controls', respectively. Then, the 
    index for the ith country on the (j + 1)th day is given by
    `stringency_values[i, j, 0] + stringency_values[i, j, 1] +
    3 * stringency_values[i, j, 2] + 2 * stringency_values[i, j, 3]`.
    Note
    ----
    Use matrix operations and broadcasting to complete this question. Please do
    not use iterative approaches like for-loops.
    '''
    
    # TODO: add your solution here and remove `raise NotImplementedError`
    weights = [1, 1, 3, 2]
    print(stringency_values)
    stringency_index = (stringency_values @ weights)
    print(stringency_index)
    return stringency_index

# Test case for Task 2.5
def test_25():
    stringency_values = np.ones((10, 20, 4))
    stringency_values[:, 10:, :] *= 2
    actual = compute_stringency_index(stringency_values)
    expected = np.ones((10, 20)) * (1 + 1 + 3 + 2)
    expected[:, 10:] *= 2
    assert(np.all(actual == expected))

    stringency_values2 = np.array([[[0, 0, 0, 0], [1, 0, 0, 0]], [[0, 0, 0, 0], [0, 1, 2, 0]]])
    actual2 = compute_stringency_index(stringency_values2)
    expected2 = np.array([[0, 1], [0, 7]])
    assert(np.all(actual2 == expected2))



#test_25()


# Task 2.6
def average_increase_in_cases(n_cases_increase, n_adj_entries_avg=7):
    '''
    Averages the increase in cases for each day using data from the previous
    `n_adj_entries_avg` number of days and the next `n_adj_entries_avg` number
    of days.
    Parameters
    ----------
    n_cases_increase: np.ndarray
        2D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the daily increase in the
        number of confirmed cases in that country, i.e. the ith row of 
        `n_cases_increase` contains the data of the ith country, and the (i, j) entry of
        `n_cases_increase` is the daily increase in the number of confirmed cases on the
        (j + 1)th day in the ith country.
    n_adj_entries_avg: int
        Number of days from which data will be used to compute the average increase
        in cases. This should be a positive integer.
    
    Returns
    -------
    Mean increase in cases for each day, using data from the previous
    `n_adj_entries_avg` number of days and the next `n_adj_entries_avg` number
    of days, as a 2D `ndarray` such that the (i, j) entry represents the
    average increase in daily cases on the (j + 1)th day in the ith country,
    rounded down to the smallest integer.
    
    The average increase in cases for a particular country on the (j + 1)th day
    is given by the mean of the daily increase in cases over the interval
    [-`n_adj_entries_avg` + j, `n_adj_entries_avg` + j]. (Note: this interval
    includes the endpoints).
    Note
    ----
    Since this computation requires data from the previous `n_adj_entries_avg`
    number of days and the next `n_adj_entries_avg` number of days, it is not
    possible to compute the average for the first and last `n_adj_entries_avg`
    number of days. Therefore, set the average increase in cases for these days
    to `np.nan` for all countries.
    '''
    
    # TODO: add your solution here and remove `raise NotImplementedError`
    window_size = 2 * n_adj_entries_avg + 1
    windows = np.lib.stride_tricks.sliding_window_view(n_cases_increase, (1, window_size))
    # Compute the mean over the windows and round down to the smallest integer
    avg_increase = np.floor(np.mean(windows, axis=-1)).squeeze()
    print(avg_increase)
    
    # Create a result array filled with nan values
    result = np.full(n_cases_increase.shape, np.nan)
    
    # Update the result array with the computed average increase, avoiding the edges
    result[:, n_adj_entries_avg:-n_adj_entries_avg] = avg_increase
    return result
    

# Test case for Task 2.6
def test_26():
    n_cases_increase = np.array([[0, 5, 10, 15, 20, 25, 30]])
    actual = average_increase_in_cases(n_cases_increase, n_adj_entries_avg=2)
    expected = np.array([[np.nan, np.nan, 10, 15, 20, np.nan, np.nan]])
    assert(np.array_equal(actual, expected, equal_nan=True))

#test_26()

# Task 2.7
def is_peak(n_cases_increase_avg, n_adj_entries_peak=7):
    
    window_size = 2 * n_adj_entries_peak + 1
    non_nan = np.ma.masked_invalid(n_cases_increase_avg)
    national_avg = np.nanmean(n_cases_increase_avg, axis=1)
    significance_threshold = 0.1 * national_avg
    windows = np.lib.stride_tricks.sliding_window_view(non_nan, (1, window_size)).squeeze()
    center_idx = n_adj_entries_peak

    # Calculate 'is_center_max' without using 'apply_along_axis'
    left_windows = windows[:, :, :center_idx]
    right_windows = windows[:, :, center_idx+1:]
    center_vals = windows[:, :, center_idx][:, :, np.newaxis]
    is_max_left = np.all(left_windows < center_vals, axis=-1)
    is_max_right = np.all(right_windows <= center_vals, axis=-1)
    is_max_result = np.logical_and(is_max_left, is_max_right)
    
    # Calculate 'is_significant'
    centers = windows[:, :, center_idx]
    is_significant = centers > significance_threshold[:, np.newaxis]

    result = np.logical_and(is_max_result, is_significant)

    padding_amount = (n_cases_increase_avg.shape[1] - result.shape[1]) // 2
    padded_result = np.pad(result, ((0, 0), (padding_amount, padding_amount)), mode='constant', constant_values=False)
    
    return padded_result


def test_27():
    n_cases_increase_avg = np.array([[np.nan, np.nan, 10, 10, 5, 20, 7, np.nan, np.nan], [np.nan, np.nan, 15, 5, 16, 17, 17, np.nan, np.nan]])
    n_adj_entries_peak = 1

    actual = is_peak(n_cases_increase_avg, n_adj_entries_peak=n_adj_entries_peak)
    expected = np.array([[False, False, False, False, False, True, False, False, False],
                        [False, False, False, False, False, True, False, False, False]])
    print(np.all(actual == expected))
    assert np.all(actual == expected)

    n_cases_increase_avg2 = np.array([[np.nan, np.nan, 10, 20, 20, 20, 20, np.nan, np.nan], [np.nan, np.nan, 20, 20, 20, 20, 10, np.nan, np.nan]])
    n_adj_entries_peak2 = 1

    actual2 = is_peak(n_cases_increase_avg2, n_adj_entries_peak=n_adj_entries_peak2)
    expected2 = np.array([[False, False, False, True, False, False, False, False, False],
                        [False, False, False, False, False, False, False, False, False]])
    print(np.all(actual2 == expected2))
    assert np.all(actual2 == expected2)

#test_27()

def visualise_increase(n_cases_increase, n_cases_increase_avg=None):
    '''
    Visualises the increase in cases for each country that is represented in
    `n_cases_increase`. If `n_cases_increase_avg` is passed into the
    function as well, visualisation will also be done for the average increase in
    cases for each country.

    NOTE: If more than 5 countries are represented, only the plots for the first 5
    countries will be shown.
    '''
    days = np.arange(1, n_cases_increase.shape[1] + 1)
    plt.figure()
    for i in range(min(5, n_cases_increase.shape[0])):
        plt.plot(days, n_cases_increase[i, :], label='country {}'.format(i))
    plt.legend()
    plt.title('Increase in Cases')

    if n_cases_increase_avg is None:
        plt.show()
        return
    
    plt.figure()
    for i in range(min(5, n_cases_increase_avg.shape[0])):
        plt.plot(days, n_cases_increase_avg[i, :], label='country {}'.format(i))
    plt.legend()
    plt.title('Average Increase in Cases')
    plt.show()


def visualise_peaks(n_cases_increase_avg, peaks):
    '''
    Visualises peaks for each of the country that is represented in
    `n_cases_increase_avg` according to variable `peaks`.
    
    NOTE: If there are more than 5 countries, only the plots for the first 5
    countries will be shown.
    '''
    days = np.arange(1, n_cases_increase_avg.shape[1] + 1)

    plt.figure()
    
    for i in range(min(5, n_cases_increase_avg.shape[0])):
        plt.plot(days, n_cases_increase_avg[i, :], label='country {}'.format(i))
        peak = (np.nonzero(peaks[i, :]))[0]
        peak_days = peak + 1 # since data starts from day 1, not 0
        plt.scatter(peak_days, n_cases_increase_avg[i, peak])
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    df = get_data()
    n_cases_cumulative = get_n_cases_cumulative(df)
    n_deaths_cumulative = get_n_deaths_cumulative(df)
    healthcare_spending = get_healthcare_spending(df)
    mask_prices = get_mask_prices(healthcare_spending.shape[1])
    stringency_values = get_stringency_values(df)
    n_cases_top_cumulative = get_n_cases_top_cumulative(df)

