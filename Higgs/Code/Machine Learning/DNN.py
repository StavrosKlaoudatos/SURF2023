import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
import tensorflow as tf
from sklearn.preprocessing import normalize


with tf.device("/gpu:0"):


    bkg1 = pd.read_parquet('/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/data/BKG/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_2017/BKG1.parquet').dropna()
    bkg2 = pd.read_parquet('/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/data/BKG/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_2017/BKG2.parquet').dropna()
    signal = pd.read_parquet('/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/data/Signals.parquet').dropna()
    #signal = pd.read_parquet('/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/data/Parquets/XHYPrivate_MX1000_MY90_2017.parquet').dropna()
    # Labeling
    bkg1['label'] = 0
    bkg2['label'] = 0
    signal['label'] = 1





    # Calculate the weights
    n_signal = len(signal)
    n_bkg1 = len(bkg1)
    n_bkg2 = len(bkg2)
    n_total_bkg = n_bkg1 + n_bkg2

    weight_signal = n_total_bkg / n_signal
    weight_bkg = 1

    signal['weight'] = weight_signal
    bkg1['weight'] = weight_bkg
    bkg2['weight'] = weight_bkg




    flatten_weights = 1




    variables = ["Diphoton_pt", "LeadPhoton_pt", "SubleadPhoton_pt", "LeadPhoton_eta", "SubleadPhoton_eta", "LeadPhoton_hoe", "SubleadPhoton_hoe","LeadPhoton_sieie","SubleadPhoton_sieie"]

   


    # Combine the datasets
    data_combined = pd.concat([bkg1.dropna(), bkg2.dropna(), signal.dropna()])


    
    from sklearn.model_selection import train_test_split
    X = data_combined[variables]
    y = data_combined['label']
    weights = data_combined['weight']

    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.4, random_state=42)



    #Flattening of the Diphoton Pt
    z_train = X_train["Diphoton_pt"].values
    bins = np.arange(0, np.max(z_train) + 5, 5)
    hist, _ = np.histogram(z_train[y_train == 0], bins=bins)
    hist2, _ = np.histogram(z_train[y_train == 1], bins=bins)
    weight_index = np.digitize(z_train, bins=bins) - 1

    # Compute weights based on histogram bins
    flatten_weights = np.array([1 / hist[weight_index[i]] if y_train.iloc[i] == 0 else 1 / hist2[weight_index[i]] for i in range(len(y_train))])
    flatten_weights[np.isinf(flatten_weights)] = 0
    flatten_weights[np.isnan(flatten_weights)] = 0




    X_mean = np.mean(X_train[variables],axis=0)
    X_std = np.std(X_train[variables],axis=0)
    print(X_mean.shape,X_std.shape)


    X_train[variables] = (X_train[variables]-X_mean)/X_std
    X_test[variables] = (X_test[variables]-X_mean)/X_std


    

    
    from keras.models import Sequential
    from keras.layers import Dense, Flatten, Dropout
    from keras.regularizers import l1


    assert not np.any(np.isnan(X_train))
    assert not np.any(np.isnan(y_train))
    assert not np.any(np.isnan(X_test))
    assert not np.any(np.isnan(y_test))



    

    combined_weights = weights_train.values * flatten_weights



    plt.hist(combined_weights, bins=50, alpha=0.6, color='g', label='Weights')
    plt.title("Distribution of Weights")
    plt.xlabel("Weight")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.show()






    model = Sequential()

    model.add(Flatten(input_shape=(9,)))
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])



    print(model.summary())
    # Setting up early stopping
    from keras.callbacks import EarlyStopping

    # Define early stopping
    early_stopping = EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=10000, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping], sample_weight=combined_weights)






    # Evaluating the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test, sample_weight=weights_test, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")




    # Save the model
    model.save('particle_collider_cnn_model_2.h5')
