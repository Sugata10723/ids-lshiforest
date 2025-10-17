import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataLoader:
    @staticmethod
    def load_nsl_kdd():
        # Load the NSL-KDD dataset
        train_df = pd.read_csv('data/nsl-kdd/KDDTrain+.txt', header=None)
        test_df = pd.read_csv('data/nsl-kdd/KDDTest+.txt', header=None)
        
        # Add column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack',
            'difficulty'
        ]
        train_df.columns = columns
        test_df.columns = columns
        
        # delete uneccesary features
        train_df.drop(['difficulty', 'protocol_type', 'service', 'flag'], axis=1, inplace=True)
        test_df.drop(['difficulty', 'protocol_type', 'service', 'flag'], axis=1, inplace=True)
            
        # encode the attack column to 0/1
        train_df['attack'] = train_df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
        test_df['attack'] = test_df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # Separate features and labels
        X_train = train_df.drop('attack', axis=1)
        y_train = train_df['attack']
        X_test = test_df.drop('attack', axis=1)
        y_test = test_df['attack']

        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, y_train, X_test, y_test

    @staticmethod
    def load_unsw_nb15(n_data=10000, n_test=3000):
        # Load the UNSW-NB15 dataset
        train_df = pd.read_csv('data/unsw_nb15/UNSW_NB15_testing-set.csv') # traversal
        test_df = pd.read_csv('data/unsw_nb15/UNSW_NB15_training-set.csv')
        
        # Preprocess the data
        train_df.drop(['id', 'attack_cat', 'rate'], axis=1, inplace=True) # rate
        test_df.drop(['id', 'attack_cat', 'rate'], axis=1, inplace=True)
        
        # eliminate categorical columns
        train_df.drop(['proto', 'service', 'state', 'is_ftp_login', 'is_sm_ips_ports'], axis=1, inplace=True)
        test_df.drop(['proto', 'service', 'state', 'is_ftp_login', 'is_sm_ips_ports'], axis=1, inplace=True)
            
        # Separate features and labels
        X_train = train_df.drop('label', axis=1)
        y_train = train_df['label']
        X_test = test_df.drop('label', axis=1)
        y_test = test_df['label']
        
        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, y_train, X_test, y_test
