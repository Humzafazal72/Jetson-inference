import torch
import boto3
import os
from datetime import datetime
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from bnt import BrainNetworkTransformer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="size_average and reduce args will be deprecated.*")



s3 = boto3.client('s3')
BUCKET_NAME = os.environ['BUCKET_NAME']

subprocess.run(["./api_call.sh", "online"], cwd="/home/nvidia/Documents/inference")

def calculate_sigmoid(prediction):
    denom = 1 + (np.exp(-1 * prediction))
    return 1 / denom

'''
def calculate_safe_correlation(data):  # Removed `self` since it's a standalone function
    eps = 1e-8

    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    std = np.where(std < eps, eps, std)

    normalized_data = (data - mean) / std
    corr = np.dot(normalized_data, normalized_data.T) / (data.shape[1] - 1)

    corr = np.nan_to_num(corr)
    corr = np.clip(corr, -1.0, 1.0)

    return corr
'''

patient_id="6gx2vgpSvj4CmUeQtqg5"

channel_labels = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9",
    "FT9-FT10", "FT10-T8"
]


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BrainNetworkTransformer()
checkpoint = torch.load('model_chb01.pt', map_location=DEVICE)
model.load_state_dict(checkpoint)
model = model.to(DEVICE)
model.eval()

if __name__ == "__main__":
    files_path = '/home/nvidia/Documents/inference/dummy_data'
    for file in sorted(os.listdir(files_path)):
        file_path = os.path.join(files_path, file)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        print(file)
        
        shutil.copyfile(file_path, "/home/nvidia/Documents/inference/correlation_calc/input.npz")
        subprocess.run(["./exec"], cwd="/home/nvidia/Documents/inference/correlation_calc", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.copy("/home/nvidia/Documents/inference/correlation_calc/correlation.npz", "/home/nvidia/Documents/inference/BNT/corr.npz")		
        data = np.load("/home/nvidia/Documents/inference/BNT/corr.npz")['correlation'][0:22, 0:22]
        '''
        timeseires = np.load(file_path, allow_pickle=True)['data'][0:22]
        pearson = calculate_safe_correlation(timeseires)
        '''
        data_tensor = torch.from_numpy(data).float().to(DEVICE).unsqueeze(0)
        
        np.fill_diagonal(data, 1.0)
        binary_data = np.where(data > 0.5, 1, 0)
        np.savez_compressed('/home/nvidia/Documents/inference/BNT/binary_data.npz', corr=binary_data) 

        with torch.no_grad():
            prediction = np.array(model(data_tensor).detach().to('cpu')).reshape(2,)
        print(prediction)

        final_pred = np.argmax(calculate_sigmoid(prediction))

        if final_pred == 1:
            subprocess.run(["./api_call.sh", "notification"], cwd="/home/nvidia/Documents/inference")
            
            fig, ax = plt.subplots(figsize=(8, 8))
            cax = ax.matshow(binary_data, cmap='Reds', interpolation='nearest')

		# Set axis ticks
            ax.set_xticks(np.arange(len(channel_labels)))
            ax.set_yticks(np.arange(len(channel_labels)))

		# Set axis tick labels
            ax.set_xticklabels(channel_labels, rotation=90)
            ax.set_yticklabels(channel_labels)
            ax.tick_params(axis='both', which='both', length=0)

                # Save the image
            plt.tight_layout()
            plt.savefig('/home/nvidia/Documents/inference/BNT/binary_image.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close()
            subprocess.run(["./api_call.sh", "image"], cwd="/home/nvidia/Documents/inference")
            s3.upload_file('/home/nvidia/Documents/inference/BNT/binary_data.npz', BUCKET_NAME, f'Correlation_npz/{patient_id}/Preictal/{timestamp}.npz')

        else:
            s3.upload_file('/home/nvidia/Documents/inference/BNT/binary_data.npz', BUCKET_NAME, f'Correlation_npz/{patient_id}/Interictal/{timestamp}.npz')
            print('no')

subprocess.run(["./api_call.sh", "offline"], cwd="/home/nvidia/Documents/inference")

