import scipy.io as sio
import json
import gzip
import numpy as np

def convert_json_gz_to_mat(json_gz_path, mat_output_path):
    """
    Reads the hermes JSON.GZ stream and saves it as a .mat file
    compatible with Matlab structs.
    """
    frames_data = []
    
    # 1. Read Stream
    with gzip.open(json_gz_path, 'rt', encoding='utf-8') as f:
        for line in f:
            frame_dict = json.loads(line)
            # Matlab prefers specific structures. 
            # We must ensure lists are converted to something Matlab understands.
            
            # Reformat detections for Matlab struct array compatibility
            clean_dets = []
            for det in frame_dict.get('det', []):
                clean_dets.append({
                    'track_id': float(det['track_id']), # Matlab numbers are doubles
                    'conf': float(det['conf']),
                    'box': [det['box']['x1'], det['box']['y1'], det['box']['x2'], det['box']['y2']],
                    'keypoints': np.array(det['keypoints']) # Convert list to numpy array
                })
            
            frames_data.append({
                'f_idx': frame_dict['f_idx'],
                'ts': frame_dict['ts'],
                'detections': clean_dets
            })

    # 2. Save to .mat
    # 'yolo_data' will be the variable name inside Matlab
    sio.savemat(mat_output_path, {'yolo_data': frames_data}, do_compression=True)