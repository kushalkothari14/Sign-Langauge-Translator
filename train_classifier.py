import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except (FileNotFoundError, pickle.UnpicklingError):
    print("Error: Unable to load 'data.pickle'. Ensure the file exists and is not corrupted.")
    exit()

if 'data' not in data_dict or 'labels' not in data_dict:
    print("Error: 'data.pickle' must contain 'data' and 'labels' keys.")
    exit()




target_size = 42
data = [np.asarray(sample, dtype=np.float32)[:target_size] if len(sample) > target_size
        else np.pad(np.asarray(sample, dtype=np.float32), (0, target_size - len(sample)), 'constant')
        for sample in data_dict['data']]

data = np.array(data)  # Convert to NumPy array
labels = np.array(data_dict['labels'], dtype=np.int64)




x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)


y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score*100))

f = open('model.p', 'wb')
pickle.dump({'model':model}, f)
f.close()



