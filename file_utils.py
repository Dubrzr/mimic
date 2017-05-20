import os
from pathlib import Path
import requests
import hashlib
import pickle
import json


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def file_exists(path):
    return os.path.isfile(path)

def folder_exists(path):
    return os.path.isdir(path)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clean_folder(path):
    if not folder_exists(path):
        return
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)

def delete_folder(path):
    if not folder_exists(path):
        return
    rmtree(path)

def create_file(path, contents=None):
    if file_exists(path):
        return
    Path(path).touch()
    if contents is not None:
        with open(path, 'w') as f:
            f.write(contents)

def delete_file(path, response=None):
    if not file_exists(path):
        return
    os.remove(path)

def file_size(path):
    stat_info = os.stat(path)
    return stat_info.st_size

def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()

def download_file(url, path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

def read_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def write_json(contents, filepath):
    with open(filepath, 'w+') as f:
        json.dump(contents, f)

        
def save_model(out_file, model, params):
    net = {'model': model, 'params': params}
    pickle.dump(net, open(out_file, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)

    
def load_model(filepath):
    net = pickle.load(open(filepath,'rb'))
    model = net['model']
    params = net['params']
    return model, params


def load_fnpz(f):
    xy = numpy.load(f, mmap_mode='r', allow_pickle=True)['arr_0']
    return xy[0,:], xy[1,:]


def load_example(data_dir, db, i, fs_target):
    f = data_dir + db + '/' + 'tr_{}_{}hz-normalized.npy'.format(i, fs_target)
    if not os.path.isfile(f):
        transform_example(db, i, fs_target)
    return numpy.load(f, mmap_mode='r', allow_pickle=True)


def load_steps(data_dir, db, i, params):
    f = data_dir + db + '/' + 'tr_{}_{}hz_{}ssi_{}sst'.format(i, params['fs_target'], params['segment_size'], params['segment_step'])
    if params['normalized_steps']:
        f += '_normalized'
    if params['correct_peaks']:
        f += '_corrected'
    f += '.npy'
    return numpy.load(f, mmap_mode='r', allow_pickle=True)

