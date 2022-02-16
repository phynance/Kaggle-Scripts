############################################################################################################################
def BlockingTimeSeriesSplit(n_samples, n_splits=5, margin_TrainValid = 16):
    idx = np.arange(n_samples)
    k_fold_size = n_samples // n_splits

    test_size = 60*24*30*3                                                     # 3 months test size, this is fixed
    smallest_train_size = k_fold_size - test_size - margin_TrainValid          # this is varying , depending on test size
    print(f"The size of smallest train set is {smallest_train_size/(60*24*30)} months") # better to be 12 months
    cv = []
    
    start = 0 # expanding window
    if smallest_train_size < test_size:     # if number of samples are not enough
        for i in range(n_splits):
            stop = start + (i+1)*k_fold_size
            mid = int(0.5 * stop ) - margin_TrainValid
            print(mid)
            train_idx = idx[start: mid]
            val_idx = idx[mid+margin_TrainValid : stop]    
            cv.append((
                train_idx,
                val_idx,
            ))
        return cv
    else:
        for i in range(n_splits):
            stop =  (i+1)*k_fold_size
            mid =  stop - test_size - margin_TrainValid

            train_idx = idx[start: mid]
            val_idx = idx[mid+margin_TrainValid : stop]    
            cv.append((
                train_idx,
                val_idx,
            ))
        return cv
############################################################################################################################
