def subset_x_y(target, features, start_index:int, end_index:int):
      
    return features[start_index:end_index], target[start_index:end_index]
   
def split_sets_by_time(df, target_col, test_ratio=0.2):
        
    df_copy = df.copy()
    target = df_copy.pop(target_col)
    cutoff = int(len(target) / 5)
    
    X_train, y_train = subset_x_y(target=target, features=df_copy, start_index=0, end_index=-cutoff*2)
    X_val, y_val     = subset_x_y(target=target, features=df_copy, start_index=-cutoff*2, end_index=-cutoff)
    X_test, y_test   = subset_x_y(target=target, features=df_copy, start_index=-cutoff, end_index=len(target))

    return X_train, y_train, X_val, y_val, X_test, y_test
    

def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='C:/Users/Yatindra/Documents/ADSI/ADSI_AT2-master/data_files/processed/'):
    
    import numpy as np

    if X_train is not None:
      np.save(f'{path}X_train', X_train)
    if X_val is not None:
      np.save(f'{path}X_val',   X_val)
    if X_test is not None:
      np.save(f'{path}X_test',  X_test)
    if y_train is not None:
      np.save(f'{path}y_train', y_train)
    if y_val is not None:
      np.save(f'{path}y_val',   y_val)
    if y_test is not None:
      np.save(f'{path}y_test',  y_test)


def load_sets(path='C:/Users/Yatindra/Documents/ADSI/ADSI_AT2-master/data_files/processed/', val=False):
    
    import numpy as np
    import os.path

    X_train = np.load(f'{path}X_train.npy') if os.path.isfile(f'{path}X_train.npy') else None
    X_val   = np.load(f'{path}X_val.npy'  ) if os.path.isfile(f'{path}X_val.npy')   else None
    X_test  = np.load(f'{path}X_test.npy' ) if os.path.isfile(f'{path}X_test.npy')  else None
    y_train = np.load(f'{path}y_train.npy') if os.path.isfile(f'{path}y_train.npy') else None
    y_val   = np.load(f'{path}y_val.npy'  ) if os.path.isfile(f'{path}y_val.npy')   else None
    y_test  = np.load(f'{path}y_test.npy' ) if os.path.isfile(f'{path}y_test.npy')  else None
    
    return X_train, y_train, X_val, y_val, X_test, y_test
    

def pop_target(df, target_col, to_numpy=False):
    
    df_copy = df.copy()
    target = df_copy.pop(target_col)
    
    if to_numpy:
        df_copy = df_copy.to_numpy()
        target = target.to_numpy()
    
    return df_copy, target
       
    
def split_sets_random(df, target_col, test_ratio=0.2, to_numpy=False):
    
    from sklearn.model_selection import train_test_split
    
    features, target = pop_target(df=df, target_col=target_col, to_numpy=to_numpy)
    
    X_data, X_test, y_data, y_test = train_test_split(features, target, test_size=test_ratio, random_state=8)
    
    val_ratio = test_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=val_ratio, random_state=8)

    return X_train, y_train, X_val, y_val, X_test, y_test
