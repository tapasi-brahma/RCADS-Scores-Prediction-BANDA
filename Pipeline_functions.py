#3 functions: 1. Modeling, 2. Permutation Test, 3. Plots


def Modeling(dataset):
    
    import pandas as pd
    pd.options.mode.chained_assignment = None  # default='warn'
    import numpy as np
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    predicting = int(input("Enter 1 to predict 'Major Depression Total Score for T2' or \n  Enter 2 to predict 'Total Anxiety Score for T2'."))
    print("\n")
    #Defining target
    
    if predicting == 1:
        
        df_target = np.array(dataset['Major Depression Total Score for T2'])
        
    else:
        
        df_target = np.array(dataset['Total Anxiety Score for T2'])
        
    #Collating Features (Predictors)
    
    input_str = input("Enter a comma-separated list of predictors:\n Enter 2 for 'Sex'.\n Enter 11 for 'Interview Age for T2'.\n Enter 3 for 'Major Depression Total Score for T1'.\n Enter 4 for 'Social Phobia Total Score for T1'.\n Enter 9 for 'Total Anxiety Score for T1'.\n Enter 20 for 'Estimated Total Intra-Cranial Volume'.\n Enter 21 for 'Left-Thalamus-Proper Volume'.\n Enter 22 for 'Right-Thalamus-Proper Volume'.\n Enter 23 for 'Left-Putamen Volume'.\n Enter 24 for 'Right-Putamen Volume'.\n Enter 25 for 'Left-Caudate Volume'.\n Enter 26 for 'Right-Caudate Volume'.\n Enter 27 for 'Left-Accumbens-area Volume'.\n Enter 28 for 'Right-Accumbens-area Volume'.\n Enter 29 for 'Left-Amygdala Volume'.\n Enter 30 for 'Right-Amygdala Volume'.\n")
    print("\n")
    input_list = input_str.split(",")
    input_columns = [int(x) for x in input_list]

    input_columns = np.array(input_columns)
    
    if predicting == 1:
        
        Model_Label = 'Predicting Major Depression Total Score for T2 using:'
        
    else:
        
        Model_Label = 'Predicting Total Anxiety Score for T2 using'
        
    for i in input_columns:
        
        Model_Label = Model_Label + ',' + dataset.columns[i]
        
    print("\033[1;30;48m\033[2J \033[1;32;48m",   Model_Label   ,"\033[0m")
    
    print("\n")
        
    df_data_demographic_clinical = np.array(dataset.iloc[:,input_columns[input_columns<12]].values) #demographic features
    df_data_structural = np.array(dataset.iloc[:,input_columns[input_columns>19]].values) #structural features
    
    train_data_demographic_clinical, test_data_demographic_clinical, train_lbl, test_lbl = train_test_split(df_data_demographic_clinical, df_target, test_size=1/5, random_state=0)
    train_data_structural, test_data_structural, train_lbl, test_lbl = train_test_split(df_data_structural, df_target, test_size=1/5, random_state=0)

    #Feature Scaling
    train_data_demographic_clinical = StandardScaler().fit_transform(train_data_demographic_clinical)
    test_data_demographic_clinical = StandardScaler().fit_transform(test_data_demographic_clinical)
    
    if any(i > 19 for i in input_columns): #if structural predictors are entered
        
        train_data_structural = StandardScaler().fit_transform(train_data_structural)
        test_data_structural = StandardScaler().fit_transform(test_data_structural)
    
    #PCA of structural training and test data
    
    pca = PCA(0.95)
    
    if any(i > 19 for i in input_columns): #if structural predictors are entered
        
        
        pca.fit(train_data_structural)
        train_data_structural = pca.transform(train_data_structural)
        test_data_structural = pca.transform(test_data_structural)
        
    else:
        
        pca.n_components_ = 0
    
    print("The number of princial components with structural features is ", pca.n_components_)
    print("The total number of predictor components is ", len(input_columns[input_columns<12])+ pca.n_components_)
    
    print("\n")
    
    
    #Merging Clinical, Demographic, and Structural data to get final train and test data
    train_data = np.concatenate((train_data_demographic_clinical , train_data_structural), axis = 1)
    test_data = np.concatenate((test_data_demographic_clinical , test_data_structural), axis = 1)
    

    #Ridge Regression: Determining best alpha
    rr_test = Ridge(alpha=0.00)
    
    #Cross-validation to determine best alpha
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_index, val_index in kf.split(train_data):
        
        # Split the training data into training and validation sets for this fold
        X_fold_train, y_fold_train = train_data[train_index], train_lbl[train_index]
   
        X_fold_val, y_fold_val = train_data[val_index], train_lbl[val_index]
    
        # Train ridge regression model on the training set
    
        rr_test.fit(X_fold_train, y_fold_train)
    
    grid = dict()
    grid['alpha'] = np.arange(0, 10, 0.01)

    search = GridSearchCV(rr_test, grid, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)

    results = search.fit(train_data, train_lbl)
    # summarize
    best_alpha = results.best_params_['alpha']
    print('MAE: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    print(best_alpha)
    
    #Performing Ridge Regression with the best alpha obtained above
    
    rr = Ridge(alpha=best_alpha)
    
    #Cross-validation with best alpha
    mse_scores = []

    for train_index, val_index in kf.split(train_data):
        
        # Split the training data into training and validation sets for this fold
        X_fold_train, y_fold_train = train_data[train_index], train_lbl[train_index]
   
        X_fold_val, y_fold_val = train_data[val_index], train_lbl[val_index]
    
        # Train ridge regression model on the training set
    
        rr.fit(X_fold_train, y_fold_train)
    
        # Predict on the validation set and compute the mean squared error
        y_pred_val = rr.predict(X_fold_val)
        mse = mean_squared_error(y_fold_val, y_pred_val)
        mse_scores.append(mse)

    # Compute the mean and standard deviation of the mean squared errors across all folds
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    print("From cross-validation, Mean squared error: {:.2f} +/- {:.2f}".format(mean_mse, std_mse))
    
    #Final Validation: R squared
    rr_fit = rr.fit(train_data, train_lbl)
    rr_fit.score(test_data, test_lbl)

    R2 = rr_fit.score(test_data, test_lbl)
    print("\n")
    print("R squared of model: ", R2)
    
    #Final Validation: MSE
    pred = rr_fit.predict(test_data)
    mse = mean_squared_error(test_lbl, pred) 
    rmse = np.sqrt(mean_squared_error(test_lbl,pred))

    print("Root mean Squared Error of model: ", rmse)
    
    if pca.n_components_ > 0:
        
        #Scree Plot: Variance expalined by principal components
    
        plt.rcParams["figure.figsize"] = [10, 5]
        PC_values = np.arange(pca.n_components_) + 1
        plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
        plt.title('Scree Plot')
        plt.xlabel('Principal Components')
        plt.ylabel('Variance Explained')
        plt.show()

        for i in range(0, len(pca.explained_variance_ratio_ )):

            print("The variance explained by PCA", i+1,"=",pca.explained_variance_ratio_ [i])

        print("\n")

        eigenvectors = pca.components_

        # get the weightage of each variable in each principal component

        col = dataset.columns[input_columns[input_columns>19]]

        for i in range(len(eigenvectors)):

            print("Weightage of structural variables in Principal Component",i+1,":")

            for j in range(len(col)):

                print(col[j],":",eigenvectors[i][j])
            print("\n")


        #Visualization of Weightage of Structural Variables in Principal Components:
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        s_eigen = sns.heatmap(eigenvectors, cmap=cmap, vmin=-1, vmax=1, center=0, cbar_kws={'extend':'both'})   
        s_eigen.set(xlabel = "Predictors", ylabel = "Principal Components",  title = "Weightage of Variables in Principal Components")
        s_eigen.set_xticklabels(col, rotation=90)
        s_eigen.set_yticklabels(PC_values)

    return dataset, predicting, input_columns, best_alpha, R2, rmse
    
    
#################################################################################################################################
  
  
def Permutation_Test(dataset, predicting, input_columns, best_alpha, R2, rmse):
    
    import pandas as pd
    pd.options.mode.chained_assignment = None  # default='warn'
    import numpy as np
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import statsmodels.api as sm
    from scipy import stats
    
    RMSE_both_array = []
    R2_permutations = []
    r2_from_mse= []
    
    df2 = dataset.copy()
    
    #Defining target
    if predicting == 1:
        
        df_target = np.array(df2['Major Depression Total Score for T2'])
        
    else:
        
        df_target = np.array(df2['Total Anxiety Score for T2'])
    
    for i in range(1000): #1000 permutations   
        
        df_data_demographic_clinical = np.array(df2.iloc[:,input_columns[input_columns<12]].values) #demographic features
        df_data_structural = np.array(df2.iloc[:,input_columns[input_columns>19]].values) #structural features

        train_data_demographic_clinical, test_data_demographic_clinical, train_lbl, test_lbl = train_test_split(df_data_demographic_clinical, df_target, test_size=1/5, random_state=0)
        train_data_structural, test_data_structural, train_lbl, test_lbl = train_test_split(df_data_structural, df_target, test_size=1/5, random_state=0)

        #Randomly shuffling the feature to be predicted in each permutation while keeping predictor columns the same
        shuffled_idx = np.random.permutation(np.arange(len(train_lbl)))
        train_lbl = train_lbl[shuffled_idx]

        pca_permutations = PCA(0.95)

        #Feature Scaling
        train_data_demographic_clinical = StandardScaler().fit_transform(train_data_demographic_clinical)
        test_data_demographic_clinical = StandardScaler().fit_transform(test_data_demographic_clinical)
        
        if any(i > 19 for i in input_columns): #if structural predictors are entered
            
            train_data_structural = StandardScaler().fit_transform(train_data_structural)
            test_data_structural = StandardScaler().fit_transform(test_data_structural)
    
            #PCA of structural training and test data
        
            pca_permutations.fit(train_data_structural)
            train_data_structural = pca_permutations.transform(train_data_structural)
            test_data_structural = pca_permutations.transform(test_data_structural)

        #Merging Clinical, Demographic, and Structural data to get final train and test data
        train_data = np.concatenate((train_data_demographic_clinical , train_data_structural), axis = 1)
        test_data = np.concatenate((test_data_demographic_clinical , test_data_structural), axis = 1)

        rr = Ridge(alpha=best_alpha)

        #k_folds = KFold(n_splits = 5)

        k_folds = KFold(n_splits=5, shuffle=True, random_state=42)

        rr_fit_permutations = rr.fit(train_data, train_lbl)

        r2_permutations = rr_fit_permutations.score(test_data, test_lbl)

        pred = rr_fit_permutations.predict(test_data)
        mse_permutations = mean_squared_error(test_lbl, pred) 
        rmse_permutations = np.sqrt(mse_permutations)
        R2_permutations.append(r2_permutations)
        RMSE_both_array.append(rmse_permutations)

    n_permutations = 1000
    
    p_value_R2 = (np.sum(np.abs(R2_permutations) >= np.abs(R2) + 1)) / (n_permutations + 1)
    if p_value_R2 == 0:
        p_value_R2 = 1 / (n_permutations + 1)
    print("p-value for R2:", p_value_R2)

    p_value_RMSE = (np.sum(RMSE_both_array <= rmse) + 1) / (n_permutations + 1)
    if p_value_RMSE == 0:
        p_value_RMSE = 1 / (n_permutations + 1)
    print("p-value for RMSE:", p_value_RMSE)
    
    #Relationship between RMSE and R2 scores of permutations
    plt.plot(RMSE_both_array,R2_permutations , 'o')

    plt.xlabel("RMSE")
        
    plt.ylabel("R2")
        
    plt.title("R2 vs. RMSE of 1000 permutations")

    plt.show()

    print("The correlation coeficient between R2 and RMSE of 1000 permutations is = ",np.corrcoef(RMSE_both_array, R2_permutations)[0,1],"\n")
    
    RMSE_both_array = np.array(RMSE_both_array)
    
    print("Permutations RMSE:",stats.describe(RMSE_both_array),"\n")

    print("Permutations R2:",stats.describe(R2_permutations),"\n")
    
    #RMSE Histogram
    
    col = dataset.columns[input_columns]

    fig, axs = plt.subplots(2, 1, figsize=(15, 15))
    
    n, bins, patches = axs[0].hist(x=RMSE_both_array, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    axs[0].grid(axis='y', alpha=0.75)
    axs[0].set_xlabel('Root Mean Square Error',fontsize=16)
    axs[0].set_ylabel('Frequency',fontsize=16)
    
    if predicting == 1:
        
        Permutation_Histogram_Title = 'T2 MDD prediction using:'
        
        for i in input_columns:
            
            if i < 12:
                
                Permutation_Histogram_Title = Permutation_Histogram_Title + dataset.columns[i] + ','
            
        if 20 in input_columns:
                
                Permutation_Histogram_Title = Permutation_Histogram_Title + 'ROI Volumes (with Total Cranial Volume)'
                
        elif any(item > 19 for item in input_columns):
                
                Permutation_Histogram_Title = Permutation_Histogram_Title + 'ROI Volumes (without Total Cranial Volume)'
                
    else:
        
        Permutaion_Histogram_Title = 'T2 Total Anxiety prediction using:'
        
        for i in input_columns:
            
            if i < 12:
                
                Permutation_Histogram_Title = Permutation_Histogram_Title + dataset.columns[i] + ','
            
        if 20 in input_columns:
                
                Permutation_Histogram_Title = Permutation_Histogram_Title + 'ROI Volumes (with Total Cranial Volume)'
                
        elif any(item > 19 for item in input_columns):
                
                Permutation_Histogram_Title = Permutation_Histogram_Title + 'ROI Volumes (without Total Cranial Volume)'
          
    Permutation_Histogram_Title_RMSE = Permutation_Histogram_Title + ': RMSE for different permutations'
            
    axs[0].set_title(Permutation_Histogram_Title_RMSE)
    axs[0].axvline(rmse, ls="--", color="r")
    
    #Percentage pf permutations with RMSE less than actual RMSE

    count_less_RMSE = 0
    for i in range(0,len(RMSE_both_array)):
    
        if RMSE_both_array[i] < rmse:
        
            count_less_RMSE = count_less_RMSE + 1
        
    print("Percentage of permutations with RMSE less than the actual RMSE is =", count_less_RMSE /len(RMSE_both_array)* 100,"\n")

    n, bins, patches = axs[1].hist(x=R2_permutations, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    axs[1].grid(axis='y', alpha=0.75)
    axs[1].set_xlabel('R-squared',fontsize=16)
    axs[1].set_ylabel('Frequency',fontsize=16)
    Permutation_Histogram_Title_R_Squared = Permutation_Histogram_Title + ': R-squared for different permutations'
    axs[1].set_title(Permutation_Histogram_Title_R_Squared)
    axs[1].axvline(R2, ls="--", color="r")
    
    #Percentage pf permutations with R2 greater than actual RMSE
    count_greater_R2 = 0
    for i in range(0,len(R2_permutations)):
    
        if R2_permutations[i] > R2:
        
            count_greater_R2 = count_greater_R2 + 1
        
    print("Percentage of permutations with R-squared greater than the actual R-squared is =", count_greater_R2 /len(R2_permutations) * 100,"\n")


################################################################################################################################


def plots(dataset, predicting): 
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pointbiserialr
    import seaborn as sns
    
    plot_str = input("Enter a comma-separated list of predictors to plot:\n Enter 11 for 'Interview Age for T2'.\n Enter 3 for 'Major Depression Total Score for T1'.\n Enter 4 for 'Social Phobia Total Score for T1'.\n Enter 9 for 'Total Anxiety Score for T1'.\n Enter 20 for 'Estimated Total Intra-Cranial Volume'.\n Enter 21 for 'Left-Thalamus-Proper Volume'.\n Enter 22 for 'Right-Thalamus-Proper Volume'.\n Enter 23 for 'Left-Putamen Volume'.\n Enter 24 for 'Right-Putamen Volume'.\n Enter 25 for 'Left-Caudate Volume'.\n Enter 26 for 'Right-Caudate Volume'.\n Enter 27 for 'Left-Accumbens-area Volume'.\n Enter 28 for 'Right-Accumbens-area Volume'.\n Enter 29 for 'Left-Amygdala Volume'.\n Enter 30 for 'Right-Amygdala Volume'.\n")
    print("\n")
    plot_list = plot_str.split(",")
    plot_columns = [int(x) for x in plot_list]

    if predicting == 1:
        
        dependent_variable = dataset['Major Depression Total Score for T2']
        
        Scatterplot_title = 'Major Depressive Disorder Total Score for T2'
        
        data1 = dataset[dataset['Female'] ==1]['Major Depression Total Score for T2']
        data2 = dataset[dataset['Female'] ==0]['Major Depression Total Score for T2']
        
        # Calculate the point-biserial correlation coefficient #r_pb = (M1 - M0) / (SD * sqrt(p * (1 - p)))
        r_pb, p_value = pointbiserialr(dataset['Female'], dataset['Major Depression Total Score for T2'])

        
    else:
        
        dependent_variable = dataset['Total Anxiety Score for T2']
        
        Scatterplot_title = 'Total Anxiety Score for T2'
        
        data1 = dataset[dataset['Female'] ==1]['Total Anxiety Score for T2']
        data2 = dataset[dataset['Female'] ==0]['Total Anxiety Score for T2']
        
        # Calculate the point-biserial correlation coefficient #r_pb = (M1 - M0) / (SD * sqrt(p * (1 - p)))
        r_pb, p_value = pointbiserialr(dataset['Female'], dataset['Total Anxiety Score for T2'])

        
    #dataset_for_plots = dataset[dataset.columns[plot_indices]]

    for i in plot_columns:
    
        corr = np.corrcoef(dataset.iloc[:,i], dependent_variable)
        
        print("The Pearson's Correlation Co-efficient between",dataset.columns[i] , "and " + Scatterplot_title+ "=", corr[0,1], ".")
    
        fig, ax = plt.subplots(1,1)
            
        plt.plot(dataset.iloc[:,i],dependent_variable ,'o')
        
        ####
        
        #obtain m (slope) and b(intercept) of linear regression line
        m, b = np.polyfit(dataset.iloc[:,i],dependent_variable, 1)
        
        #add linear regression line to scatterplot 
        plt.plot(dataset.iloc[:,i], m*dataset.iloc[:,i]+b, label = 'Linear Regression Line')
        
        plt.xlabel(dataset.columns[i])
        
        plt.ylabel(Scatterplot_title)
        
        plt.title(dataset.columns[i]+" Vs." + Scatterplot_title)
            
        plt.show()  
        
    #Box plot for Sex
    data=[data1, data2]

    s_box = sns.boxplot(data, showmeans=True, meanline=True)
    s_box.set(xlabel = "Sex", ylabel = Scatterplot_title, title = "Box plots of " + Scatterplot_title+ " for different genders")
    s_box.set_xticklabels(['Male', 'Female'])

    means = [np.mean(d) for d in data]
    stds = [np.std(d) for d in data]

    print("The point-biserial correlation coefficient between Sex and "+ Scatterplot_title + "=",r_pb,".")
    print("The mean and standard deviation of " + Scatterplot_title +" for female participants are",means[1], "and", stds[1],"respectively.")
    print("The mean and standard deviation of "+ Scatterplot_title + " for male participants are",means[0], "and", stds[0],"respectively.")

 
    
    
    