import numpy as np


from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping





# mm_ann = 'sty' # MLP512 Acc: 0.8110184669494629 SOFTMAX Acc: 0.7777806720469078


# clf = MLPClassifier(hidden_layer_sizes=(512,), activation='relu', solver='adam', max_iter=200, verbose=True, random_state=42)
# clf = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', max_iter=200, verbose=True, random_state=42)
# clf = LogisticRegression(random_state=42, multi_class='multinomial', solver='sag', max_iter=200, n_jobs=4, verbose=True)

n_classes = len(set(y_train)) + 1  # UNK
# model = Sequential([
#     Dense(512, activation='relu', input_shape=(768,)),
#     Dense(n_classes, activation='softmax'),
# ])
# model = Sequential([
#     Dense(n_classes, activation='softmax', input_shape=(768,)),
# ])
#
#
# es = EarlyStopping(monitor='acc', mode='max', verbose=1, min_delta=0.01, patience=10)

# print('Training ...')
# clf.fit(X_train, y_train)

print("Loading Model...")
model = Sequential([
        Dense(18426, activation='softmax', input_shape=(768,)),
    ])
model.load_weights("models/Classifiers/softmax.cui.h5")
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# model.fit(
#     X_train,
#     to_categorical(y_train),
#     epochs=100,
#     batch_size=64,
#     callbacks=[es],
# )

print('Evaluating ...')
# y_dev_preds = mlp.predict_proba(X_dev)
# y_dev_preds = clf.predict(X_dev)

# acc = accuracy_score(y_dev, y_dev_preds)

loss, acc = model.evaluate(X_dev[:1], to_categorical(y_dev[:1], num_classes=18426))
print('Acc:', acc)

# print('Saving model ...')
# joblib.dump(clf, 'lr_multi.%s.model.joblib' % mm_ann)
# model.save('softmax.%s.model.h5' % mm_ann)
# joblib.dump(train_label_mapping, 'softmax.%s.mapping.joblib' % mm_ann)
