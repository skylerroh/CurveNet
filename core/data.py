"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@Time: 2021/1/21 3:10 PM
"""


import glob
import gzip
import h5py
import itertools
import numpy as np
import os
import sys
import torch
import pandas as pd
import pickle5 as pickle
from dataclasses import dataclass
from torch.utils.data import Dataset

sys.path.append(f"{os.path.dirname(__file__)}/../../..")
from utils.proteins import *

# change this to your data root
DATA_DIR = f'{os.path.dirname(__file__)}/../../../structure_files/atom_sites'
BASE_POINT_CLOUD_DIR = os.path.join(DATA_DIR, "point_clouds")
POINT_CLOUD_HDF5 = lambda x, y: f"{POINT_CLOUD_DIR(x)}/{y}_protein_point_clouds.hdf5"

EVAL_PCT = 0.2

PAD = 'PAD'

SHAPE_TO_INDEX = {
 None: 0,
 'BEND': 1,
 'HELX_LH_PP_P': 2,
 'HELX_RH_3T_P': 3,
 'HELX_RH_AL_P': 4,
 'HELX_RH_PI_P': 5,
 'STRN': 6,
 'TURN_TY1_P': 7,
 PAD: 8,
}

AA_TO_INDEX = {
 'ALA': 0,
 'ARG': 1,
 'ASN': 2,
 'ASP': 3,
 'CYS': 4,
 'GLN': 5,
 'GLU': 6,
 'GLY': 7,
 'HIS': 8,
 'ILE': 9,
 'LEU': 10,
 'LYS': 11,
 'MET': 12,
 'PHE': 13,
 'PRO': 14,
 'SER': 15,
 'THR': 16,
 'TRP': 17,
 'TYR': 18,
 'VAL': 19,
 PAD: 20,
}

def get_point_cloud_dir(name):
    return os.path.join(BASE_POINT_CLOUD_DIR, name)


def get_point_cloud_hdf5(name, train_test):
    return f"{get_point_cloud_dir(name)}/{train_test}_protein_point_clouds.hdf5"


@dataclass
class ProteinPointCloud:
    protein_id: str
    atom_sites: np.ndarray
    shapes: np.ndarray
    aas: np.ndarray # AminoAcids
    num_atoms: int
    
    def labels_to_multihot(self, labels, N):
        arr = np.zeros((N,))
        if labels: 
            idx = np.array(labels, dtype=int)
            arr[idx] = 1.0
        return arr

    def get_w_label(self, label_lookup_dict, n_categories:int=18):
        return ProteinPointCloudWLabel(
            protein_id=self.protein_id,
            atom_sites=self.atom_sites,
            shapes=self.shapes,
            aas=self.aas,
            num_atoms=self.num_atoms,
            labels=self.labels_to_multihot(label_lookup_dict.get(self.protein_id), n_categories))


@dataclass
class ProteinPointCloudWLabel:
    protein_id: str
    atom_sites: np.ndarray
    shapes: np.ndarray
    aas: np.ndarray # AminoAcids
    num_atoms: int
    labels: np.ndarray


def store_point_clouds_w_labels_as_hd5f(name, point_clouds_w_labels, partition, num_points, n_categories):
    POINT_CLOUD_DIR = get_point_cloud_dir(name)
    
    if not os.path.exists(POINT_CLOUD_DIR): os.mkdir(POINT_CLOUD_DIR)
    N = len(point_clouds_w_labels)
    with h5py.File(get_point_cloud_hdf5(name, partition), "w") as f:
        points = np.stack([p.atom_sites for p in point_clouds_w_labels])
        print(points.shape)
        
        f.create_dataset("atom_sites", shape=(N,num_points,3), dtype=np.dtype(float),
                         data=points)
            
        f.create_dataset("protein_id", shape=(N,), dtype=h5py.string_dtype(),
                         data=[p.protein_id for p in point_clouds_w_labels])
        
        f.create_dataset("shapes", shape=(N,num_points), dtype='i8',
                         data=[p.shapes for p in point_clouds_w_labels])
        
        f.create_dataset("amino_acids", shape=(N,num_points), dtype='i8',
                         data=[p.aas for p in point_clouds_w_labels])
        
        f.create_dataset("labels", shape=(N, n_categories), dtype=np.dtype(float),
                         data=[p.labels for p in point_clouds_w_labels])
        
        f.create_dataset("num_atoms", shape=(N,), dtype='i8',
                         data=[p.num_atoms for p in point_clouds_w_labels])
    


def read_point_clouds_w_labels_as_hd5f(name, partition):
    with h5py.File(get_point_cloud_hdf5(name, partition), "r+") as f:
        return f["protein_id"][:], f["atom_sites"][:], f["shapes"][:], f["amino_acids"][:], f["labels"][:]


def one_per_amino_acid(atom_sites: pd.DataFrame):
    return atom_sites[atom_sites["label_atom_id"] == "CA"].reset_index()


def protein_pandas_to_numpy(group):
    return group[['Cartn_x', 'Cartn_y', 'Cartn_z']].apply(pd.to_numeric).to_numpy()

    
def mask_by_confidence(atom_group):
    return atom_group[atom_group.confidence_pLDDT.astype(float) > 50].sort_values("id")
    
    
def protein_to_sampled_point_cloud(atom_sites: pd.DataFrame, num_points: int):
    shapes = atom_sites["shape.conf_type_id"].apply(lambda x: SHAPE_TO_INDEX[x]).to_numpy()
    aas = atom_sites["label_comp_id"].apply(lambda x: AA_TO_INDEX[x]).to_numpy()
    atom_sites = protein_pandas_to_numpy(atom_sites).astype(float)
    
    def assertAllOfSize(arrays, shape):
        for a in arrays:
            assert(a.shape == shape, f"got {a.shape}, expected {shape}")
    if atom_sites.shape[0] == 0: 
        return None, None, None
    elif atom_sites.shape[0] <= num_points:
        num_pad = num_points - atom_sites.shape[0]
        padded = np.concatenate([atom_sites, np.zeros((num_pad, 3))])
        padded_shapes = np.concatenate([shapes, np.repeat(SHAPE_TO_INDEX[PAD], num_pad)])
        padded_AA = np.concatenate([aas, np.repeat(AA_TO_INDEX[PAD], num_pad)])
        assertAllOfSize([padded, padded_shapes, padded_AA], (num_points, 3))
        return padded, padded_shapes, padded_AA

    else:
        idx = np.sort(np.random.choice(np.arange(0, atom_sites.shape[0]), size=num_points, replace=False))
        sampled = atom_sites[idx,:]
        sampled_shapes = shapes[idx]
        sampled_AA = aas[idx]
        assertAllOfSize([sampled, sampled_shapes, sampled_AA], (num_points, 3))
        return sampled, sampled_shapes, sampled_AA
    

def protein_to_masked_point_cloud(atom_sites: pd.DataFrame, num_points: int):
    atom_sites = mask_by_confidence(atom_sites)
    return protein_to_sampled_point_cloud(atom_sites, num_points)
    

def train_test_split(all_point_clouds_w_labels):
    train_point_clouds, test_point_clouds = [], []
    for p in all_point_clouds_w_labels:
        if hash(p.protein_id) % 100 >= (EVAL_PCT*100):
            train_point_clouds.append(p)
        else:
            test_point_clouds.append(p)
    return train_point_clouds, test_point_clouds

    
def create_protein_point_clouds(name, num_points=2048, overwrite=False):
    if not os.path.exists(BASE_POINT_CLOUD_DIR): os.mkdir(BASE_POINT_CLOUD_DIR)
    POINT_CLOUD_DIR = get_point_cloud_dir(name)
    
    dir_exists = os.path.exists(POINT_CLOUD_DIR)
    if dir_exists and overwrite:
        os.rmdir(POINT_CLOUD_DIR)
    if not dir_exists:
        labels = load_labels()
        n_categories = labels.num_unique # int(max([max(k) for k in label_lookup_dict.values() if len(k) > 0]) + 1)

        all_point_clouds = []

        atom_files = glob.glob(os.path.join(DATA_DIR, "atom_sites_part_*.parquet"))
        for filename in atom_files:
            print(filename.split("/")[-1])
            atom_sites = pd.read_parquet(filename)
            for id, group in atom_sites.groupby("protein_id"):
                group = one_per_amino_acid(group)
                atoms, shapes, aas = point_cloud_method_by_name[name](group, num_points=num_points)
                if atoms is None:
                    print("no points, skipping")
                    continue
                all_point_clouds.append(ProteinPointCloud(id, atoms, shapes, aas, group.shape[0]))

        all_point_clouds = [p.get_w_label(labels.protein_to_labels, n_categories) for p in all_point_clouds]
        all_point_clouds_w_labels = [p for p in all_point_clouds if p.labels.sum() > 0]
        train_point_clouds, test_point_clouds = train_test_split(all_point_clouds_w_labels)
        
        print(f"Number of protein point clouds generated: {len(all_point_clouds)}")
        print(f"Number of protein point clouds generated with labels: {len(all_point_clouds_w_labels)}")
        print(f"Number of training proteins: {len(train_point_clouds)}")
        print(f"Number of test proteins: {len(test_point_clouds)}")
        print(f"Storing protein point clouds in hd5f format: {POINT_CLOUD_HDF5}")
        store_point_clouds_w_labels_as_hd5f(name, train_point_clouds, "train", num_points, n_categories)
        store_point_clouds_w_labels_as_hd5f(name, test_point_clouds, "test", num_points, n_categories)
        store_point_clouds_w_labels_as_hd5f(name, all_point_clouds, "all", num_points, n_categories)
        store_point_clouds_w_labels_as_hd5f(name, all_point_clouds_w_labels, "all_with_labels", num_points, n_categories)
        print("Done created point clouds")


@dataclass
class Labels:
    protein_to_labels: dict
    num_unique: int
    ic_vec: np.ndarray
    pos_weights: np.ndarray

        
def load_labels():
    def get_index_of_function_label(function, function_arr):
        return function_arr.get(function)
    min_proteins_for_label = 20
    
    functions = pd.read_parquet(f"{DATA_DIR}/protein_goa_mf_parent_only_lessthan500.parquet")
    functions = functions[~pd.isna(functions.IC_t) & (functions.num_proteins_per_parent_mf > 20)]
    unique_functions = sorted(functions.parent_mf.dropna().unique())
    print("num unique function labels: ", len(unique_functions))

    functions_to_index = {function: int(i) for i, function in enumerate(unique_functions)}
    functions['function_idx'] = functions.parent_mf.apply(
        lambda x: get_index_of_function_label(x, functions_to_index)
    )
    protein_to_function_labels = functions.groupby("protein_id").function_idx.agg(lambda x: sorted(list(set(x.dropna()))))
    protein_to_labels_dict = protein_to_function_labels.to_dict()
    
    ic_vec = functions[["parent_mf", "IC_t"]].drop_duplicates().set_index("parent_mf").IC_t.sort_index().to_numpy()
    pos_weight = get_pos_weight(protein_to_labels_dict)
    
    return Labels(
        protein_to_labels = protein_to_labels_dict,
        num_unique = len(unique_functions),
        ic_vec = ic_vec,
        pos_weights = pos_weight
    )
        
        


# def get_ic_vec(labels_dict):
#     N = len(labels_dict)
#     m = pd.Series(list(itertools.chain(*list(labels_dict.values())))).value_counts()
#     ic = -np.log(m.sort_index()/N)
#     return ic

def get_pos_weight(labels_dict):
    N = len(labels_dict)
    m = pd.Series(list(itertools.chain(*list(labels_dict.values())))).value_counts().sort_index()
    pos_weight = (N-m) / m
    return np.sqrt(pos_weight).to_numpy()


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class ProteinsSampled(Dataset):
    def __init__(self, num_points, partition='train'):
        self.num_label_categories = 18
        self.num_points = num_points
        self.partition = partition
        self.max_points = 2048
        self.data, self.label = self.load_data_cls(partition)

        
    def load_data_cls(self, partition, overwrite=False):
        if not os.path.exists(DATA_DIR):
            raise Exception("first download structure_files into project root")
        create_protein_point_clouds(name="sampled", num_points=self.max_points, overwrite=overwrite)
        all_data, all_label = read_point_clouds_w_labels_as_hd5f(name="sampled", partition=partition)
        print(all_data.shape)
        print(all_label.shape)
        return all_data, all_label


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            # pointcloud = jitter_pointcloud(pointcloud)
            # np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    
    
class ProteinsExtended(Dataset):
    def __init__(self, num_points, partition='train'):
        self.num_label_categories = 18
        self.num_points = num_points
        self.partition = partition
        self.max_points = 2700
        self.data, self.label = self.load_data_cls(partition)

    def load_data_cls(self, partition, overwrite=False):
        if not os.path.exists(DATA_DIR):
            raise Exception("first download structure_files into project root")
        create_protein_point_clouds(name="sequence_head", num_points=self.max_points, overwrite=overwrite)
        all_data, all_label = read_point_clouds_w_labels_as_hd5f(name="sequence_head", partition=partition)
        print(all_data.shape)
        print(all_label.shape)
        return all_data, all_label


    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = rotate_pointcloud(pointcloud)
            # pointcloud = translate_pointcloud(pointcloud)
            # pointcloud = jitter_pointcloud(pointcloud)
            # np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

        
class ProteinsExtendedWithMask(Dataset):
    def __init__(self, num_points, partition='train'):
        self.name = "confidence_mask_new_child_labels_shape_and_aminos"
        self.num_label_categories = 312
        self.num_points = num_points
        self.partition = partition
        self.max_points = 4000
        self.id, self.data, self.shapes, self.amino_acids, self.seqvec, self.label = self.load_data_cls(partition)
        self.augment_data = self.partition in ('train', 'all_with_labels')
        print(f"partition `{partition}`, augment_data with rotation when get item called: {self.augment_data}")
        
    def load_data_cls(self, partition, overwrite=False):
        if not os.path.exists(DATA_DIR):
            raise Exception("first download structure_files into project root")
        create_protein_point_clouds(name=self.name, num_points=self.max_points, overwrite=overwrite)
        all_id, all_data, all_shapes, all_amino_acids, all_label = read_point_clouds_w_labels_as_hd5f(name=self.name, partition=partition)
        
        # add seqvec to data 
        with open(f"{DATA_DIR}/seqvec_vectors.pkl", "rb") as fh:
            seqvec = pickle.load(fh)[["protein_id", "seqvec"]].set_index("protein_id").seqvec.to_dict()
            all_seqvec = [seqvec[p.decode("utf-8")] for p in all_id]
        
        print(all_id.shape)
        print(all_data.shape)
        print(all_label.shape)
        return all_id, all_data,  all_shapes, all_amino_acids, all_seqvec, all_label


    def __getitem__(self, item):
        _id = self.id[item]
        amino_acids = self.amino_acids[item]
        shapes = self.shapes[item]
        seqvec = self.seqvec[item]
        pointcloud = self.data[item]
        label = self.label[item]
        if self.augment_data:
            pointcloud = rotate_pointcloud(pointcloud)
        return _id, pointcloud, shapes, amino_acids, seqvec, label

    def __len__(self):
        return self.data.shape[0]
    
    
point_cloud_method_by_name = {
    "sampled": protein_to_sampled_point_cloud,
    "sequence_head": protein_to_sampled_point_cloud,
    "confidence_mask_new_labels": protein_to_masked_point_cloud,
    "confidence_mask_new_child_labels": protein_to_masked_point_cloud,
    'confidence_mask_new_child_labels_shape_and_aminos': protein_to_masked_point_cloud,
}