from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

def train_bilstm_model(model, X_train, y_train_enc, X_val, y_val_enc):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint('./model_bilstm_augmented_commit.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    with tf.device('/GPU:0'):
        history_embedding = model.fit(X_train, y_train_enc, 
                                      epochs=40, batch_size=16, 
                                      validation_data=(X_val, y_val_enc),
                                      verbose=1, callbacks=[es, mc])
    return history_embedding
