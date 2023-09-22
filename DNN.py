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









    variables = ["Diphoton_pt", "LeadPhoton_pt", "SubleadPhoton_pt", "LeadPhoton_eta", "SubleadPhoton_eta", "LeadPhoton_hoe", "SubleadPhoton_hoe","LeadPhoton_sieie","SubleadPhoton_sieie"]

    # Combine the data for standardization
    #combined_data = pd.concat([bkg1[variables], bkg2[variables], signal[variables]])

    # Standardize the data
    #scaler = StandardScaler().fit(combined_data)
    #bkg1[variables] = scaler.transform(bkg1[variables])
   # bkg2[variables] = scaler.transform(bkg2[variables])
   # signal[variables] = scaler.transform(signal[variables])




    # Combine the datasets
    data_combined = pd.concat([bkg1.dropna(), bkg2.dropna(), signal.dropna()])


    
    from sklearn.model_selection import train_test_split
    X = data_combined[variables]
    y = data_combined['label']
    weights = data_combined['weight']

    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.4, random_state=42)


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


    print(bkg1.isna().sum())
    print(bkg2.isna().sum())
   



    print((X_train.shape[1], 1))



    
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



    #ROC Curve

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # Predict probabilities
    y_pred = model.predict(X_test)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_pred, sample_weight=weights_test)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/Plots/DNN_Plots/"+"ROC.png")
    plt.show()




    #Accuracy of the Model vs Epoch

    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy vs. Epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/Plots/DNN_Plots/"+"AccvEpoch.png")
    plt.show()





    #Accuracy for Training vs Testing Set

    plt.figure()
    plt.bar(['Training', 'Testing'], [max(history.history['accuracy']), accuracy])
    plt.title('Accuracy for Training vs Testing Set')
    plt.ylabel('Accuracy')
    plt.savefig("/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/Plots/DNN_Plots/"+"acc_tr_v_ts.png")
    plt.show()






    #Signal vs Background Score on Train and Test Set over Epochs

    # Predicting for training set
    train_pred_probs = model.predict(X_train)

    # Averaging scores over epochs
    avg_train_signal_score = np.mean(train_pred_probs[y_train == 1])
    avg_train_background_score = np.mean(train_pred_probs[y_train == 0])

    avg_test_signal_score = np.mean(y_pred[y_test == 1])
    avg_test_background_score = np.mean(y_pred[y_test == 0])

    plt.figure()
    plt.bar(['Train Signal', 'Train Background', 'Test Signal', 'Test Background'], 
            [avg_train_signal_score, avg_train_background_score, avg_test_signal_score, avg_test_background_score])
    plt.title('Signal vs Background Score on Train and Test Set')
    plt.ylabel('Average Score')
    plt.savefig("/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/Plots/DNN_Plots/"+"Signal_vs_bkg.png")
    plt.show()



    def get_activations(model, layer_idx, input_data):
        intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[layer_idx].output)
        activations = intermediate_model.predict(input_data)
        return activations

    # Choose a random sample from the test set
    sample_input = X_test.sample(1)

    # Get activations for all hidden layers
    activations = []
    for i in range(4):  # We have 4 hidden layers
        activations.append(get_activations(model, i, sample_input))

    # Plotting
    plt.figure(figsize=(15, 5))
    for i, activation in enumerate(activations):
        plt.subplot(1, 4, i+1)
        plt.hist(activation[0])
        plt.title(f'Layer {i+1} Activations')
    plt.tight_layout()
    plt.savefig("/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/Plots/DNN_Plots/"+"activation_layers.png")
    plt.show()




    # Extracting weights and biases
    weights = []
    biases = []
    for layer in model.layers:
        if isinstance(layer, Dense):  # Check if the layer is a Dense layer
            w, b = layer.get_weights()
            weights.append(w)
            biases.append(b)

    # Plotting
    # Plotting
    plt.figure(figsize=(15, 5))
    for i, (w, b) in enumerate(zip(weights, biases)):
        # Plot weights
        plt.subplot(2, 4, i+1)
        plt.hist(w.ravel())
        plt.title(f'Layer {i+1} Weights')
        
        # Plot biases
        plt.subplot(2, 4, i+5)
        plt.hist(b.ravel())
        plt.title(f'Layer {i+1} Biases')
    plt.tight_layout()
    plt.savefig("/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/Plots/DNN_Plots/"+"w_and_b.png")
    plt.show()





    binary_predictions = np.where(y_pred > 0.5, 1, 0).ravel()

    # Plotting
    plt.figure(figsize=(15, 5))
    plt.scatter(range(len(y_test)), y_test, alpha=0.5, label='True Label', s=5)
    plt.scatter(range(len(y_test)), binary_predictions, alpha=0.5, label='Predicted Label', s=5)
    plt.title('True vs. Predicted Labels')
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.legend()
    plt.show()



    # Get the raw predictions for the test set

    def score_classifier(y_test, y_pred, caption):
        ##########################################################
        # make histogram of discriminator value for signal and bkg
        ##########################################################
        yNN_frame = pd.DataFrame({'truth': y_test, 'discriminator': y_pred.ravel()})
        disc_bkg = yNN_frame[yNN_frame['truth'] == 0]['discriminator'].values
        disc_signal = yNN_frame[yNN_frame['truth'] == 1]['discriminator'].values
        
        plt.figure(figsize=(10, 6))
        plt.hist(disc_bkg, label='Background', density=True, bins=50, alpha=0.3)
        plt.hist(disc_signal, label='Signal', density=True, bins=50, alpha=0.3)
        plt.xlabel(caption)
        plt.yscale("log")
        plt.legend(prop={'size': 10})
        plt.legend(loc='best')
        plt.savefig("/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/Plots/DNN_Plots/"+"Net_Discrim.png")
        plt.show()

    # Call the function
    score_classifier(y_test, y_pred, 'Neural Network Discriminator')







    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Predict classes
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("/Users/stavrosklaoudatos/Desktop/SURF2023/Higgs/Plots/DNN_Plots/"+"confusion.png")
    plt.show()



    


    



    
