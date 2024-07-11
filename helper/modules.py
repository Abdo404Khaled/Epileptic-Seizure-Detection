import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from scipy.interpolate import interp1d
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

class EpilepticSeizure():
    def __init__(self, config):
        self.config = config
        self.base_folder = self.config['data_path']
        self.X = None
        self.y = None
    
    def process(self):
        self.__read_data()

        if self.config.get("preprocess", True):
            self.__preprocess()

        self.__augment_data()

        return self.X, self.y

    def __read_data(self):
        data = []
        
        for folder in sorted(os.listdir(self.base_folder)):
            if '.' in folder:
                continue
                
            folder_path = os.path.join(self.base_folder, folder)
            
            for filename in tqdm(sorted(os.listdir(folder_path)), desc=f"Processing files in {folder}"):
                file_path = os.path.join(folder_path, filename)
                
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    
                    for i in range(23):
                        start = i * 178
                        end = start + 178
                        eeg_readings = [float(value.strip()) for value in lines[start:end]]
                        
                        data.append(eeg_readings + [folder])
        
        column_names = [f'Channel_{i+1}' for i in range(178)] + ['Label']
        data = pd.DataFrame(data, columns=column_names)

        self.X = data.iloc[:,:-1]
        self.y = data.iloc[:,-1:]
    
    def __preprocess(self):
        if self.config.get('labels', 5) == 2:
            self.y['Label'] = self.y['Label'].map({'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 1})
        elif self.config.get('labels', 5) == 3:
            self.y['Label'] = self.y['Label'].map({'A': 0, 'B': 0, 'C': 1, 'D': 1, 'E': 2})
        else:
             self.y['Label'] = self.y['Label'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4})

    def __augment_data(self):
        X_list = [self.X]
        y_list = [self.y]

        x_columns = self.X.columns
        y_columns = self.y.columns

        if self.config.get('add_noise', False):
            X_augmented_noise, y_augmented_noise = self.__add_noise()
            X_list.append(X_augmented_noise)
            y_list.append(y_augmented_noise)

        if self.config.get('undersampling', False):
            X_under_resampled, y_under_resampled = self.__under_sampling()
            X_list.append(X_under_resampled)
            y_list.append(y_under_resampled)
        
        if self.config.get('oversampling', False):
            X_over_resampled, y_over_resampled = self.__over_sampling()
            X_list.append(X_over_resampled)
            y_list.append(y_over_resampled)

        self.X = np.vstack(X_list)
        self.y = np.concatenate(y_list)

        self.X = pd.DataFrame(self.X, columns=x_columns)
        self.y = pd.DataFrame(self.y, columns=y_columns)

    
    def __add_noise(self):
        augmented_data = []

        for idx, row in self.X.iterrows():
            data = row.values
            label = self.y.iloc[idx]

            noise_level = np.random.uniform(0, 0.1)
            noisy_data = data + noise_level * np.random.normal(size=len(data))

            factor = np.random.uniform(0.9, 1.1)
            x = np.arange(len(noisy_data))
            f = interp1d(x, noisy_data)
            new_x = np.linspace(0, len(noisy_data) - 1, int(len(noisy_data) * factor))
            interpolated_data = f(new_x)

            resampled_data = resample(interpolated_data, len(data))
            augmented_data.append(np.append(resampled_data, label))

        augmented_data = np.array(augmented_data)

        np.random.shuffle(augmented_data)

        X_augmented_noise = pd.DataFrame(augmented_data[:, :-1], columns=self.X.columns)
        y_augmented_noise = pd.DataFrame(augmented_data[:, -1], columns=['Label'])
    
        return X_augmented_noise, y_augmented_noise
        
    def __over_sampling(self):
        return RandomOverSampler(random_state=42).fit_resample(self.X, self.y)

    def __under_sampling(self):
        return RandomUnderSampler(random_state=42).fit_resample(self.X, self.y)

#####################################################################################################################
#TODO: Modify for this dataset

class LeaveOneOut():
    def __init__(self, dataset, model, subsets_no, prediction_label):
        self.dataset = data
        self.patient_subjects = []
        self.model = model
        self.patient_no = subsets_no
        self.prediction_label = prediction_label
        self.data_len = len(self.dataset)
        self.num_rows_per_df = self.data_len // self.patient_no
        
        for i in range(self.patient_no):
            start_index = i * self.num_rows_per_df
            end_index = (i + 1) * self.num_rows_per_df
            subject = dataset.iloc[start_index:end_index]
            self.patient_subjects.append(subject)
        
    def __split(self, selected_index):
        test_data = self.patient_subjects[selected_index]
        train_data = [self.patient_subjects[i] for i in range(self.patient_no) if i != selected_index]
        train_labels = [i for i in range(self.patient_no) if i != selected_index]
        train_data = pd.concat(train_data, ignore_index=True)
        print(f"Leaving patient {selected_index} out, training on patients {', '.join(map(str, train_labels))}")
        X_train = train_data.drop(self.prediction_label, axis = 1)
        y_train = train_data[self.prediction_label]
        X_test = test_data.drop(self.prediction_label, axis = 1)
        y_test = test_data[self.prediction_label]
        return X_train, y_train, X_test, y_test
        
    def train(self, epoch, batch_size, validation_split = 0, callbacks=[]):
        results = []
        for selected_index in range(self.patient_no):
            X_train, y_train, X_test, y_test = self.__split(selected_index)
            model = self.model()
            print('Training....')
            train_history = model.fit(X_train, y_train, epochs = epoch, batch_size = batch_size, callbacks=callbacks, validation_split = validation_split)
            print('Test....')
            test_accuracy = model.evaluate(X_test, y_test, batch_size = batch_size)
            print()
            results.append((train_history, test_accuracy))
        return results