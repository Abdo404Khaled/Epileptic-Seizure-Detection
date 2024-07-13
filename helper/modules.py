import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.signal import spectrogram
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
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def process(self):
        self.__read_data()

        self.__split_data()

        if self.config.get("preprocess", True):
            self.__preprocess()

        self.__augment_data()

        if self.config.get("generate_images", True):
            self.__generate_images(self.X_train, "Train")
            self.__generate_images(self.X_test, "Test")

        return self.X_train, self.y_train, self.X_test, self.y_test

    def __read_data(self):
        data = []
        
        for folder in sorted(os.listdir(self.base_folder)):
            if '.' in folder or len(folder) > 1:
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

        num_people = len(data) // self.config.get("rows_per_person", 23)

        grouped_records = [data.iloc[i*self.config.get("rows_per_person", 23):(i+1)*self.config.get("rows_per_person", 23)] for i in range(num_people)]
        np.random.shuffle(grouped_records)

        data = pd.concat(grouped_records, ignore_index=True)

        self.X = data.iloc[:,:-1]
        self.y = data.iloc[:,-1:]
    
    def __split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.config.get("test_size", 0.2), shuffle=False)
    
    def __preprocess(self):
        if self.config.get('labels', 5) == 2:
            label_map = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 1}
        elif self.config.get('labels', 5) == 3:
            label_map = {'A': 0, 'B': 0, 'C': 1, 'D': 1, 'E': 2}
        else:
            label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

        self.y_train['Label'] = self.y_train['Label'].map(label_map)
        self.y_test['Label'] = self.y_test['Label'].map(label_map)
    
    def __generate_images(self, X, mode):
        output_folder = os.path.join(self.base_folder, f"Patient_images_{mode}")

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
            
        os.makedirs(output_folder, exist_ok=True)

        image_paths = []

        for person_id in tqdm(range(0, len(X), self.config.get("rows_per_person", 23)), desc=f"Processing Patients {mode}"):
            person_data = X.iloc[person_id:person_id + self.config.get("rows_per_person", 23), :178].values.flatten()
            image_path = self.__compute_stft_and_save(person_data, person_id, output_folder)

            image_paths.extend([image_path] * self.config.get("rows_per_person", 23))
        
        X['Image_Path'] = pd.Series(image_paths, index=X.index)

    def __augment_data(self):
        X_list = [self.X_train]
        y_list = [self.y_train]

        x_columns = self.X_train.columns
        y_columns = self.y_train.columns

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

        self.X_train = np.vstack(X_list)
        self.y_train = np.concatenate(y_list)

        self.X_train = pd.DataFrame(self.X_train, columns=x_columns)
        self.y_train = pd.DataFrame(self.y_train, columns=y_columns)

    
    def __add_noise(self):
        augmented_data = []

        for idx, row in self.X_train.iterrows():
            data = row.values
            label = self.y_train.iloc[idx]

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

        X_augmented_noise = pd.DataFrame(augmented_data[:, :-1], columns=self.X_train.columns)
        y_augmented_noise = pd.DataFrame(augmented_data[:, -1], columns=['Label'])
    
        return X_augmented_noise, y_augmented_noise
    
    def __compute_stft_and_save(self, person_data, person_id, output_folder):
        f, t, Sxx = spectrogram(person_data, self.config.get('frequency', 178))

        plt.figure(figsize=(8, 4))
        plt.pcolormesh(t, f, np.log(Sxx + 1e-10), cmap='viridis')
        plt.axis('off')  

        image_path = os.path.join(output_folder, f'person_{person_id // self.config.get("rows_per_person", 23) + 1}_spectrogram.png')
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return image_path
        
    def __over_sampling(self):
        return RandomOverSampler(random_state=42).fit_resample(self.X_train, self.y_train)

    def __under_sampling(self):
        return RandomUnderSampler(random_state=42).fit_resample(self.X_train, self.y_train)