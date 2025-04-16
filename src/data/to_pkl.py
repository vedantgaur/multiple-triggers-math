import pickle
import json

def txt_to_pkl(file_path):
    with open(file_path, 'r') as file:
        data = eval(file.read())

    pkl_file_path = file_path.replace('.txt', '.pkl')
    with open(pkl_file_path, 'wb') as file:
        pickle.dump(data, file)

    print(f"Text file '{file_path}' converted and saved as '{pkl_file_path}'.")

def json_to_pkl(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    pkl_file_path = file_path.replace('.json', '.pkl')
    
    with open(pkl_file_path, 'wb') as file:
        pickle.dump(data, file)
    
    print(f"JSON file '{file_path}' converted and saved as '{pkl_file_path}'.")

json_to_pkl('datasets/test_math_50.json')
json_to_pkl('datasets/math_300.json')