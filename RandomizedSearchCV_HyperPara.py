param_dist = {
        #'n_estimators': np.linspace(500,2000,4, dtype="int"),
        'max_depth': range(2,20,2),
        #'learning_rate':np.linspace(0.01,0.1,4),
        'num_leaves': range(5,50,5),
        #'min_child_samples': range(60,120,20) ,
        #'min_data_in_leaf': np.linspace(100,1000,10, dtype="int"),

        }
new_models = {} # collecting the learned parameters
new_models_score = {}

#####################    optimize on Bitcoin first   #####################

print("RandomizedSearchCV for: " + asset_name)
grid_search = RandomizedSearchCV(
    estimator = models[asset_id],
    param_distributions = param_dist,
    n_jobs = -1,
    cv = BlockingTimeSeriesSplit(n_samples=int(Xs[asset_id].shape[0]), n_splits=3),
    verbose = 10,
    scoring = score,
    n_iter= 30,
    random_state = 0
)
grid_search.fit(Xs[asset_id], ys[asset_id])


new_models[asset_id] = grid_search.best_estimator_
new_models_score[asset_id] = grid_search.best_score_
