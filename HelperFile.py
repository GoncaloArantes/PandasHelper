# Functions for various tasks

import os
import functools
import operator
import numpy as np
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

class PandasHelper():

    r'''
        Library to automate pandas Dataframe handling.
    '''
    
    def __init__(self, dataframe: pd.DataFrame):
        
        if isinstance(dataframe, pd.DataFrame):
            self.df = dataframe
        else:
            raise TypeError(f'Argument is not a valid Dataframe. Argument type => {type(dataframe)}')
        
    
    # Private helper function for type checking
    def __check_type(self, var):
        # Strings and lists are the types we want to check
        if not isinstance(var, Union[str, list]):
            raise TypeError(f'Argument must be a string or list. Argument type => {type(var)}')
        

    # Private helper function for splitting the label from the features
    def __data_label_splitting(self, label_name: str):
        # Check for valid argument type (It is highly unlikely someone would enter a list as the name of the label)
        self.__check_type(label_name)
        # Split data and return X (features) and y (label)
        return self.df.drop(columns=label_name, axis=1), self.df[label_name]

    
    # Private helper function to check if a column has all its values respecting a rule and treat invalid ones 
    def __rule_checking(self, col: str, rules: list, methods: list, args: list):
        # Empty values (Will be updated later with rule checking)
        indexes_drop = None
        # Condition value (Can be a list of values if needed)
        condition_values = None
        # Counter to keep track of extra arguments
        arg_i = 0
        # Keep column type for later
        col_type = self.df[col].dtype

        # Methods and way to implement them
        methods_way = {
                        'drop': self.df.drop(indexes_drop, inplace=True),
                        'null': self.df.replace({col: condition_values}, np.nan, inplace=True),
                        'mean': self.df.replace({col: condition_values}, self.df[col].mean(), inplace=True),
                        'median': self.df.replace({col: condition_values}, self.df[col].median(), inplace=True),
                        'most_frequent': self.df.replace({col: condition_values}, self.df[col].mode(), inplace=True),
                        'max': self.df.replace({col: condition_values}, self.df[col].max(), inplace=True),
                        'min': self.df.replace({col: condition_values}, self.df[col].min(), inplace=True),
                        'interpolate': self.df[col].interpolate(inplace=True),
        }
        r'''Interpolation method -> Linear: Selected values have to be converted into np.nan before interpolation.
            After replacement, if the initial column type was a variation of integers, type conversion is performed,
                since each replacement method converts the column type to "float".
        '''
                
        # Iterate over the rules for the column
        for index, rule in enumerate(rules):
            
            # Check rule
            match rule:
                case 'non_negative': 
                    indexes_drop = self.df[self.df[col] < 0].index
                    condition_values = self.df[self.df[col] < 0].values
                case 'non_positive':
                    indexes_drop = self.df[self.df[col] > 0].index
                    condition_values = self.df[self.df[col] > 0].values
                case 'not_zero':
                    indexes_drop = self.df[self.df[col] == 0].index
                    condition_values = self.df[self.df[col] == 0].values
                case 'higher_than':
                    indexes_drop = self.df[self.df[col] < args[arg_i]].index
                    condition_values = self.df[self.df[col] < args[arg_i]].values
                    arg_i += 1
                case 'lower_than':
                    indexes_drop = self.df[self.df[col] > args[arg_i]].index
                    condition_values = self.df[self.df[col] > args[arg_i]].values
                    arg_i += 1

            # If the rule is not followed, indexes_drop won't be empty
            if not any(indexes_drop):
                print(f'Rule {rule} not respected, method {methods[index]} will be used!')

                # Interpolation requires null values
                if methods[index] == 'interpolate':
                    methods_way.get('null')
                
                # Perform value changing
                methods_way.get(methods[index])
                print('Method completed!')

                # Change column type (It will be from "float" to "int" if needed)
                if col_type in ['int', 'int32', 'int64']:
                    self.df[col] = self.df[col].astype('int64')
            
            # If the rule is followed, nothing happens
            else:
                print(f'Rule {rule} is respected!')


    # Check all callable functions defined explicitly in this class
    def all_functions(self):
        # Filter explicit methods
        return [func for func in dir(PandasHelper) if (callable(getattr(PandasHelper, func)) and not func.startswith('_'))]


    def dummy_creation(self, non_encoded_cols: list):
        r'''Create dummies/one-hot encoded columns.

            Arg:
                non_encoded_cols (list): List of columns to perform the get_dummies
                    operation from pandas.
        '''
        # Check columns' types
        col_types = {col: self.df[col].dtype for col in self.df.columns}
        
        # Check for valid argument type
        self.__check_type(non_encoded_cols)
        
        # Check if the number of columns in the dataframe is higher than the number of columns to encode
        if len(non_encoded_cols) > len(self.df.columns): raise AttributeError('Number of columns exceed the number of categories in the dataframe')

        # Remove columns from the dictionary (Columns we don't want to one-hot encode)
        for column in col_types.copy().keys():
            if column not in non_encoded_cols:
                col_types.pop(column)
        
        # Dummy creation for object columns (They must be objects or integers to be converted)
        eligible_dummies = [col for col in col_types.keys() if col_types[col] in ['O','int', 'int32', 'int64']]

        #Check eligible dummies (Must have at least 1) and perform dummy operation
        if len(eligible_dummies) <= 0: raise AttributeError('No eligible dummies')
        dummy_cols = pd.get_dummies(data=self.df[eligible_dummies], drop_first=True, dtype=int)

        # Concatenate the dummy columns with the original dataframe and drop the old unmodified ones
        self.df = pd.concat([self.df.drop(eligible_dummies, axis=1), dummy_cols], axis=1)
        
        # End statement
        print('Dummy(ies) created!')


    def datetime_handling(self, date_cols: list):
        r'''Create year and month columns, while dropping the timestamp.

            Arg:
                date_cols (list): List of columns to perform the to_datetime
                    operation from pandas.
        '''
        # Check for valid argument type
        self.__check_type(date_cols)

        # Convert to datetime format
        for column in date_cols:
            try:
                self.df[column] = pd.to_datetime(self.df[column])
            except ValueError:
                raise AttributeError(f'{column} cannot be converted to datetime!')

        # Insert year and month from the datetime columns
        for column in date_cols:
            self.df[[f'year_{column}', f'month_{column}']] = self.df[column].apply(lambda date: pd.Series([date.year, date.month]))

        # Drop datetime columns
        self.df.drop(date_cols, axis=1, inplace=True)

        # End statement
        print('Timestamp(s) modified!')

    
    def data_compacting(self, csv_path: str, file_type: str = 'parquet'):
        r'''Convert csv file to another format for data compacting purposes.
            Some column data types are also going to be changed for the same reason.
        
            Args:
                csv_path (str): Path where the dataframe is located
                file_type (str): Type wanted for data compacting (default: 'parquet')

            Options for file_type:
                -parquet: Binary Parquet Format;
                -feather: Binary Feather Format;
                -pickle: Pickle (serialize) object to file;
                -hdf5: Hierarchical Data Format. Can hold a mix of related objects 
                    which can be accessed as a group or as individual objects.
        '''
        # Check both the csv_path and file_type type
        self.__check_type(csv_path), self.__check_type(file_type)

        # Check if the file_type is valid (Dictionary with values to append to the final path)
        file_types = {
                        'parquet': ['parquet', lambda path: self.df.to_parquet(path, engine='pyarrow')], 
                        'feather': ['feather', lambda path: self.df.to_feather(path)], 
                        'pickle': ['pkl', lambda path: self.df.to_pickle(path)], 
                        'hdf5': ['h5', lambda path: self.df.to_hdf(path, key='df')]
        }
        
        if file_type not in file_types.keys(): raise SyntaxError(f'Unknown file type. Available types => {list(file_types.keys())}')

        # Check if there is a dataframe in the csv_path and if the path is valid
        try:
            pd.read_csv(csv_path)
            print(f'CSV path {csv_path} contains a dataframe!')
        # If the path is not valid
        except FileNotFoundError:
            raise NotADirectoryError(f'CSV path {csv_path} does not exist!')
        # If the file is empty
        except pd.errors.EmptyDataError:
            raise ValueError('CSV file is empty!')
        # If there is any error during parsing
        except pd.errors.ParserError:
            raise IOError('Error parsing the CSV file!')

        # Iterate over all columns to check their type
        for col in self.df.columns:

            # Integer type changing
            if self.df[col].dtype in ['int32', 'int64']:
                max_value = self.df[col].max()
                min_value = self.df[col].min()
                if min_value > -128 and max_value < 127:
                    self.df[col] = self.df[col].astype('int8')
                elif min_value > -32768 and max_value < 32767:
                    self.df[col] = self.df[col].astype('int16')
                else:
                    print("Column's data type not changed!")

            # Float type changing
            if self.df[col].dtype == 'float64':
                self.df[col] = self.df[col].astype('float32')

        # Change file type
        base_path = os.path.split(csv_path)
        file_name = base_path[1].split('.')[0]

        # Final format for the file
        f_type = file_types.get(file_type)[0]
        final_path = f'{base_path[0]}/{file_name}.{f_type}'
        
        # Perform file type changing
        file_types.get(file_type)[1](final_path)
        
        # End statement
        print(f'CSV file converted to the {file_type} format')


    def null_handling(
                    self, 
                    row_ratio_removal: float = 0.1,
                    col_ratio_removal: float = 0.5,
                    imputation: str = 'univariate',
                    method: str = 'median',
                    label_name: str = 'label',
                    test_size: float = 0.33,
                    random_state: int = 101,
                    scaler: str = 'standard',
                    **kwargs
        ):
        r'''Function to handle null entries.
            
            If a dataframe is passed through this function, the following will happen:
                - Rows with more than 50% of the corresponding column values missing are dropped;
                - Rows with columns having less missing values than the row_ratio_removal are dropped;
                - Columns having more missing values than the col_ratio_removal are dropped;
                - Null values are filled by 'univariate' or 'multivariate' imputations, with an associated method;
                - Data is split before imputation and is returned divided into training and testing;
                - If a multivariate imputation method is selected, data needs to be scaled;
                - If a univariate imputation method is selected, data is not scaled and will need to be scaled before
                    model training and testing.

            Args:
                row_ratio_removal (float): Threshold ratio of null values in a column 
                    for the rows to be dropped (default: 0.1)
                col_ratio_removal (float): Threshold ratio of null values in a column 
                    for the column to be dropped (default: 0.5)
                imputation (str): Selection of imputation type (default: 'univariate')
                method (str): Imputation method (default: 'median')
                label_name (str): Name of the label for data splitting
                    (default: 'label')
                test_size (float): Proportion of the dataset to include in the test split
                    (default: 0.33)
                random_state (int): Controls the shuffling applied to the data
                    (default: 101)
                scaler (str): Scales the data before null value imputation (default: 'standard')
                **kwargs (optional): Additional key-value arguments for functions
                    (example: if imputation == 'multivariate' and method == 'knn',
                    the n_neighbors will need to be passed here {'n_neighbors': 2})

            Examples of methods for Univariate imputation:
                -mean: Replace null values with the mean of the column;
                -median: Replace null values with the median of the column;
                -most_frequent: Replace null values with the most frequent value in the column;
                -constant: Replace null values with the same constant value
                    (if method == 'constant', fill_value should be in the kwargs)
                
            For Multivariate imputation only the KNN and Iterative Imputers are available.

            Returns:
                X_train, X_test, y_train, y_test -> Ready for model training and evaluation
        '''
        # Drop rows with more than 50% of the corresponding column values missing
        self.df = self.df[self.df.isna().mean(axis=1) < 0.5]
        print('Rows with more than half of the corresponding column values missing dropped!')

        # Check columns with null values
        null_op_count = lambda col: self.df[col].isna().sum()
        cols_null_values = [col for col in self.df.columns if null_op_count(col) > 0]

        # Check ratio of null values for each column to the dataframe's size
        null_values_ratio_per_col = {col: null_op_count(col)/len(df) for col in cols_null_values}

        # Drop rows if the ratio is smaller than 10% (row_ratio_removal)
        nans_drop_per_col = [col for col in null_values_ratio_per_col.keys() if null_values_ratio_per_col[col] <= row_ratio_removal]
        self.df.dropna(subset=nans_drop_per_col, inplace=True)
        print(f'Rows with a ratio of missing values per column smaller than {row_ratio_removal:.0%} dropped!')

        # Drop columns if the ratio is higher than 50% (When dealing with columns with high importance, imputation methods should be used instead)
        cols_to_drop = [col for col in null_values_ratio_per_col.keys() if null_values_ratio_per_col[col] >= col_ratio_removal]
        self.df.drop(columns=cols_to_drop, axis=1, inplace=True)
        print(f'Columns with a ratio of missing values higher or equal than {col_ratio_removal:.0%} dropped!')

        # Data imputation (Repeat the null count).
        null_cols_to_fill = [col for col in self.df.columns if null_op_count(col) > 0]
        
        # By this point, if there are null values left to handle, data splitting will be performed due to the need for imputation
        # Instead of just applying changes to the dataframe, after this point, X_train, X_test, y_train and y_test will be returned
        
        # Make sure all null values are np.nan to ensure integrity when using imputers
        self.df.replace({'': np.nan}, inplace=True)

        # Imputation should be performed after data splitting to avoid data leakage
        if label_name not in self.df.columns: raise AttributeError(f'Label name {label_name} is not represented in the dataframe')
        if type(test_size) != 'float': raise TypeError(f'Test size argument must be a float. Argument type => {type(test_size)}')
        if not 0.0 < test_size < 1.0: raise ValueError('Test size out of bounds. Must be between 0.0 and 1.0')
        if type(random_state) != 'int': raise TypeError(f'Random state argument must be an integer. Argument type => {type(random_state)}')
        
        # Split the label from the features and then, performing the training and testing split 
        X, y = self.__data_label_splitting(label_name)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)
        print('Data is split into training and testing!')

        # Check if there are any null values left to fill (If no columns are represented, the divided partitions are returned)
        if not null_cols_to_fill:
            print('No null values left to fill')
            return X_train, X_test, y_train, y_test

        # Imputation methods
        match imputation:
            case 'univariate':
                if method in ['mean', 'median', 'most_frequent']:
                    imputer = SimpleImputer(missing_values=np.nan, strategy=method)
                elif method == 'constant':
                    if 'fill_value' in kwargs:
                        imputer = SimpleImputer(missing_values=np.nan, strategy=method, fill_value=kwargs.get('fill_value'))
                    else:
                        raise TypeError('Value to fill not provided')
                else:
                    raise SyntaxError('Unknown method')

            case 'multivariate':
                # KNN Imputer
                if method.lower() in ['knn', 'knnimputer', 'knn_imputer']: 
                    # Number of neighbors argument
                    if 'n_neighbors' in kwargs:
                        n_neighbors = kwargs.get('n_neighbors')
                    else:
                        raise TypeError('Number of neighbors not provided')
                    # Rules for number of neighbors argument
                    if type(n_neighbors) != 'int': 
                        raise TypeError(f'Number of neighbors must be an integer. Inputted type => {type(n_neighbors)}')
                    if n_neighbors >= len(self.df.columns):
                        raise AttributeError(f'Number of neighbors {n_neighbors} higher than the number of columns {len(self.df.columns)}')
                    imputer = KNNImputer(n_neighbors=n_neighbors)

                # Iterative Imputer
                elif method.lower() in ['iter', 'iterative', 'iterativeimputer', 'iterative_imputer']:
                    if 'max_iter' in kwargs:
                        max_iter = kwargs.get('max_iter')
                    else:
                        raise AttributeError('Number of maximum iterations not provided')    
                    imputer = IterativeImputer(max_iter=max_iter, random_state=101)
            
                # If multivariate imputation is selected, data needs to be scaled (Univariate imputation does not require data scaling)
                scalers = {
                            # Assumes the dataset follows a Gaussian distribution, sensitive to outliers, but fast
                            'standard': StandardScaler(), 
                            # Data compressed in [0, 1], might result in loss of significant information, sensitive to outliers, but fast
                            'minmax': MinMaxScaler(),
                            # Only applicable to positive data, sensitive to outliers, but sparsity is preserved
                            'maxabs': MaxAbsScaler(),
                            # Does not take into account mean and median, but handles outliers and skewness
                            'robust': RobustScaler(),
                }
                if scaler in scalers.keys():
                    scaler = scalers.get(scaler)
                else:
                    raise AttributeError('No scaler/Wrong scaler provided. Available scalers => ("standard", "minmax", "maxabs", "robust")')
                
                # Scale the data
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                print('Data is scaled!')

            case _:
                raise SyntaxError('Unknown imputation type. Imputations should be either univariate or multivariate')    
        
        print('Imputation method defined!')

        # Perform fit and transform operations
        X_train = imputer.fit_transform(X_train)
        print('Missing values from the training set imputed!')
        
        X_test = imputer.transform(X_test)
        print('Missing values from the testing set imputted!')

        # Return the variables ready for model training and evaluation
        return X_train, X_test, y_train, y_test
    

    def duplicate_handling(self, cols: list, methods: list = 'drop'):
        r'''Function to handle duplicate data in the provided columns.
            This function helps handling invalid duplicate data, such as customer identifiers in a dataframe 
                where only the customer data is being tracked.
            

            Args:
                cols (list): Column(s) to check for duplicate values
                methods (list): Method to implement on duplicate values
                    (default: 'drop')

            Examples of methods for handling duplicates:
                -drop: Drop all the column's duplicate values;
                -mean: Replace all duplicate values with the mean of the column;
                -median: Replace all duplicate values with the median of the column;
                -most_frequent: Replace all duplicate values with the most frequent value in the column;
                -null: Replace all duplicate values with np.nan (Handled later with null_handling function)
        '''
        # Check for valid argument types (Type and size)
        self.__check_type(cols), self.__check_type(methods)
        if len(cols) != len(methods): raise SyntaxError('Number of columns and methods must be the same!')

        # Variable to store duplicate values to handle
        condition_values = None

        # Check if inputted methods are known
        possible_methods = {
                            'drop': lambda col: self.df.drop_duplicates(subset=col, keep='first', inplace=True),
                            'mean': lambda col: self.df[col].mask(condition_values, self.df[col].mean(), inplace=True),
                            'median': lambda col: self.df[col].mask(condition_values, self.df[col].median(), inplace=True),
                            'most_frequent': lambda col: self.df[col].mask(condition_values, self.df[col].mode(), inplace=True),
                            'null': lambda col: self.df[col].mask(condition_values, np.nan, inplace=True),
        }
        all_methods = list(set(methods))
        if not set(all_methods).issubset(set(possible_methods.keys())): 
            raise SyntaxError(f'Unknown methods. Available methods => {possible_methods.keys()}')

        # Iterate over the columns and correct, if needed, the duplicates
        for index, col in enumerate(cols):
            condition_values = df.duplicated(subset=col, keep='first')
            
            # Check if there are duplicates
            if self.df[condition_values].empty:
                print(f'{col} does not have duplicates to handle!')
            else:
                print(f'{col} has duplicate values!')
                # Apply correction method to the duplicates
                possible_methods.get(methods[index])(col)
                print(f"{col}'s duplicate values are treated!")

        # End statement
        print(f'Duplicate values from columns {cols} handled!')


    # Check data integrity (e.g. No negative ages)
    def validate_date(self, cols_rules: dict, correction_methods: list, *args):
        r'''Function to validate data in the provided columns.
            More functionalities will be added to handle proficiently more data types (Handles numeric data better).
            
            Args:
                cols_rules (dict): Dictionary with columns and the rules the respective
                    column should follow (e.g. {'age': ['non_negative']})
                    Any column may have more than one rule. In that case => ['rule_1', 'rule_2']
                    Even if only one rule is inputted, it has to be inside of a list
                correction_methods (list): Methods to handle invalid values. They have to be in order
                    relatively to cols_rules. They are passed as a list of lists.
                    Example: [['drop', 'mean'], ['interpolate']]
                *args (optional): Additional arguments for rules
                    (example: if one of the rules is higher_than or lower_than, arguments need to be inputted)
                
            Rows not respecting the rules can be:
                -Dropped;
                -Replaced with np.nan - handled later with null_handling function;
                -Corrected - given the mean/median/most frquent/max/min value of the column;
                -Interpolated.

            If the dataset is large, removing invalid rows might be negligible.
            If the dataset is small, imputation might be preferred.
            Either way, the declared rule must make sense and take into account domain knowledge.
        '''
        # Possible rules and methods to apply to the columns
        possible_rules = ('non_negative', 'non_positive', 'not_zero', 'higher_than', 'lower_than')
        possible_methods = ('drop', 'null', 'mean', 'median', 'most_frequent', 'max', 'min', 'interpolate')

        # Arguments must have the required type
        if type(cols_rules) != 'dict': raise TypeError(f'Argument cols_rules must be a dictionary. Argument type => {type(cols_rules)}')
        self.__check_type(correction_methods)
        if len(cols_rules) != len(correction_methods): raise SyntaxError('Number of columns and methods must be the same!')

        # Both the rules and correction methods need to have valid values
        all_rules = functools.reduce(operator.iconcat, list(cols_rules.values()), [])
        all_methods = functools.reduce(operator.iconcat, list(correction_methods), [])
        if not set(all_rules).issubset(set(possible_rules)): raise SyntaxError(f'Unknown rules. Available rules => {possible_rules}')
        if not set(all_methods).issubset(set(possible_methods)): raise SyntaxError(f'Unknown methods. Available methods => {possible_methods}')

        # Rules 'higher_than' and 'lower_than' need additional arguments
        col_i = 0
        arg_i = 0
        arguments = []
        
        # Iterate over the columns
        for col, rules in cols_rules.items():
            # Iterate over the rules
            for rule_i in range(len(rules)):
                # Check if the current rule is either 'higher_than' or 'lower_than'
                if rules[rule_i] in ['higher_than', 'lower_than']:
                    if args[arg_i]:
                        arguments.append(args[arg_i])
                        arg_i += 1
                    else: 
                        raise AttributeError(f'Argument not inputted for rule {rules[rule_i]}')
            
            print(f'{col} is being checked!')
            self.__rule_checking(col, rules, correction_methods[col_i], arguments)
            args.clear()
            col_i += 1

        # End statement
        print('Data is validated!')


    def highest_lowest_correlation(self, columns: list, method: str = 'pearson'):
        r'''Function to get the highest and lowest correlation values with the inputted columns

            Examples of correlation methods:
                -pearson: Drop all the column's duplicate values;
                -kendall: Replace all duplicate values with the mean of the column;
                -spearman: Replace all duplicate values with the median of the column
        '''
        # Check for valid argument type
        self.__check_type(columns)

        # Check for valid method
        methods = ('pearson', 'kendall', 'spearman')
        if type(method) != 'str': raise TypeError(f'Method must be a string. Argument type => {type(method)}')
        if method not in methods: raise AttributeError(f'{method} must be one of {methods}')

        # Store the correlation matrix
        corr_matrix = self.df.corr(method=method, numeric_only=True)

        # Function to get the column regarding a specific correlation value
        corr_comp = lambda col, corr: list(corr_matrix[corr_matrix[col] == corr].index)[0]
        
        # Get the highest and lowest correlation values from the required columns
        for column in columns:
            min_corr, max_corr = corr_matrix[column].sort_values()[0:-1].values
            min_corr_col, max_corr_col = corr_comp(column, min_corr), corr_comp(column, max_corr)
            print(f'{column} has the lowest correlation ({min_corr}) with {min_corr_col}!')
            print(f'{column} has the highest correlation ({max_corr}) with {max_corr_col}!')


    #def summary(self):

        # Additional functions I'd like to implement:

        #         summarization
        # outlier detection
        #






dataframe = pd.DataFrame(np.random.randn(5, 5), columns=list('ABCDE'))
d = pd.Series([1,2,3,4,2], name='integers')
df = pd.concat([dataframe, d], axis=1)
print(PandasHelper(dataframe=df).all_functions())
exit()
PandasHelper(dataframe=df).dummy_creation(['A','B','integers'])
PandasHelper(dataframe=dataframe).correlation(['A','B'])