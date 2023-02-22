import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from os.path import isfile, join
import h5py
import subprocess
import shlex
import json
import glob
from .. ops import transform_functions, se3
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
#from pycuda import gpuarray
#from pycuda.compiler import SourceModule
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from open3d import io
from .. data_utils.utils_data_utils import generate_pointcloud, better_random_corner_sensor_imitation


def download_modelnet40():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)
	if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
		www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
		zipfile = os.path.basename(www)
		os.system('wget %s; unzip %s' % (www, zipfile))
		os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
		os.system('rm %s' % (zipfile))


def load_data(train, use_normals):
	if train: partition = 'train'
	else: partition = 'test'
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	all_data = []
	all_label = []
	print(DATA_DIR)
	for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
		f = h5py.File(h5_name)
		if use_normals: data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1).astype('float32')
		else: data = f['data'][:].astype('float32')
		label = f['label'][:].astype('int64')
		f.close()
		all_data.append(data)
		all_label.append(label)
	all_data = np.concatenate(all_data, axis=0)
	all_label = np.concatenate(all_label, axis=0)
	return all_data, all_label


def deg_to_rad(deg):
	return np.pi / 180 * deg


def create_random_transform(dtype, max_rotation_deg, max_translation):
	max_rotation = deg_to_rad(max_rotation_deg)
	rot = np.random.uniform(-max_rotation, max_rotation, [1, 3])
	trans = np.random.uniform(-max_translation, max_translation, [1, 3])
	quat = transform_functions.euler_to_quaternion(rot, "xyz")

	vec = np.concatenate([quat, trans], axis=1)
	vec = torch.tensor(vec, dtype=dtype)
	return vec


def jitter_pointcloud(pointcloud, sigma=0.04, clip=0.05):
	# N, C = pointcloud.shape
	sigma = 0.04*np.random.random_sample()
	pointcloud += torch.empty(pointcloud.shape).normal_(mean=0, std=sigma).clamp(-clip, clip)
	return pointcloud


def farthest_subsample_points(pointcloud1, num_subsampled_points=768):
	pointcloud1 = pointcloud1
	num_points = pointcloud1.shape[0]
	nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
							 metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
	random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
	#random_p1 = better_random_corner_sensor_imitation()
	idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
	gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)
	return pointcloud1[idx1, :], gt_mask

def knn_idx(pts, k):
    kdt = cKDTree(pts) 
    _, idx = kdt.query(pts, k=k+1)
    return idx[:, 1:]

def get_rri(pts, k):
    # pts: N x 3, original points
    # q: N x K x 3, nearest neighbors
    q = pts[knn_idx(pts, k)]
    p = np.repeat(pts[:, None], k, axis=1)
    # rp, rq: N x K x 1, norms
    rp = np.linalg.norm(p, axis=-1, keepdims=True)
    rq = np.linalg.norm(q, axis=-1, keepdims=True)
    pn = p / rp
    qn = q / rq
    dot = np.sum(pn * qn, -1, keepdims=True)
    # theta: N x K x 1, angles
    theta = np.arccos(np.clip(dot, -1, 1))
    T_q = q - dot * p
    sin_psi = np.sum(np.cross(T_q[:, None], T_q[:, :, None]) * pn[:, None], -1)
    cos_psi = np.sum(T_q[:, None] * T_q[:, :, None], -1)
    psi = np.arctan2(sin_psi, cos_psi) % (2*np.pi)
    idx = np.argpartition(psi, 1)[:, :, 1:2]
    # phi: N x K x 1, projection angles
    phi = np.take_along_axis(psi, idx, axis=-1)
    feat = np.concatenate([rp, rq, theta, phi], axis=-1)
    return feat.reshape(-1, k * 4)

def get_rri_cuda(pts, k, npts_per_block=1):
	raise NotImplementedError
	"""
    import pycuda.autoinit
    mod_rri = SourceModule(open('rri.cu').read() % (k, npts_per_block))
    rri_cuda = mod_rri.get_function('get_rri_feature')

    N = len(pts)
    pts_gpu = gpuarray.to_gpu(pts.astype(np.float32).ravel())
    k_idx = knn_idx(pts, k)
    k_idx_gpu = gpuarray.to_gpu(k_idx.astype(np.int32).ravel())
    feat_gpu = gpuarray.GPUArray((N * k * 4,), np.float32)

    rri_cuda(pts_gpu, np.int32(N), k_idx_gpu, feat_gpu,
             grid=(((N-1) // npts_per_block)+1, 1),
             block=(npts_per_block, k, 1))
    
    feat = feat_gpu.get().reshape(N, k * 4).astype(np.float32)
    return feat
    """


class UnknownDataTypeError(Exception):
	def __init__(self, *args):
		if args: self.message = args[0]
		else: self.message = 'Datatype not understood for dataset.'

	def __str__(self):
		return self.message


class ModelNet40Data(Dataset):
	def __init__(
		self,
		train=True,
		num_points=1024,
		download=True,
		randomize_data=False,
		use_normals=False
	):
		super(ModelNet40Data, self).__init__()
		if download: download_modelnet40()
		self.data, self.labels = load_data(train, use_normals)
		if not train: self.shapes = self.read_classes_ModelNet40()
		self.num_points = num_points
		self.randomize_data = randomize_data

	def __getitem__(self, idx):
		if self.randomize_data: current_points = self.randomize(idx)
		else: current_points = self.data[idx].copy()

		current_points = torch.from_numpy(current_points[:self.num_points, :]).float()
		label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

		return current_points, label

	def __len__(self):
		return self.data.shape[0]

	def randomize(self, idx):
		pt_idxs = np.arange(0, self.num_points)
		np.random.shuffle(pt_idxs)
		return self.data[idx, pt_idxs].copy()

	def get_shape(self, label):
		return self.shapes[label]

	def read_classes_ModelNet40(self):
		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
		DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
		file = open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r')
		shape_names = file.read()
		shape_names = np.array(shape_names.split('\n')[:-1])
		return shape_names


class ClassificationData(Dataset):
	def __init__(self, data_class=ModelNet40Data()):
		super(ClassificationData, self).__init__()
		self.set_class(data_class)

	def __len__(self):
		return len(self.data_class)

	def set_class(self, data_class):
		self.data_class = data_class

	def get_shape(self, label):
		try:
			return self.data_class.get_shape(label)
		except:
			return -1

	def __getitem__(self, index):
		return self.data_class[index]


class RegistrationData(Dataset):
	def __init__(self, algorithm, data_class=ModelNet40Data(), partial_source=False, partial_template=False, noise=False, half_fragments=False, additional_params={}):
		super(RegistrationData, self).__init__()
		available_algorithms = ['PCRNet', 'PointNetLK', 'DCP', 'PRNet', 'iPCRNet', 'RPMNet', 'DeepGMR']
		if algorithm in available_algorithms: self.algorithm = algorithm
		else: raise Exception("Algorithm not available for registration.")
		
		self.set_class(data_class)
		self.partial_template = partial_template
		self.partial_source = partial_source
		self.noise = noise
		self.additional_params = additional_params
		self.use_rri = False
		self.half_fragments = half_fragments

		if self.algorithm == 'PCRNet' or self.algorithm == 'iPCRNet':
			from .. ops.transform_functions import PCRNetTransform
			self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=1)
		if self.algorithm == 'PointNetLK':
			from .. ops.transform_functions import PNLKTransform
			self.transforms = PNLKTransform(0.8, True)
		if self.algorithm == 'RPMNet':
			from .. ops.transform_functions import RPMNetTransform
			self.transforms = RPMNetTransform(0.8, True)
		if self.algorithm == 'DCP' or self.algorithm == 'PRNet':
			from .. ops.transform_functions import DCPTransform
			self.transforms = DCPTransform(angle_range=45, translation_range=1)
		if self.algorithm == 'DeepGMR':
			self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri
			from .. ops.transform_functions import DeepGMRTransform
			self.transforms = DeepGMRTransform(angle_range=90, translation_range=1)
			if 'nearest_neighbors' in self.additional_params.keys() and self.additional_params['nearest_neighbors'] > 0:
				self.use_rri = True
				self.nearest_neighbors = self.additional_params['nearest_neighbors']

	def __len__(self):
		return len(self.data_class)

	def set_class(self, data_class):
		self.data_class = data_class

	def __getitem__(self, index):
		template, label = self.data_class[index]
		self.transforms.index = index				# for fixed transformations in PCRNet.
		source = self.transforms(template)

		if self.half_fragments:
			number_of_points = source.shape[0]
			number_of_points = int(number_of_points * 0.5)
		else:
			number_of_points = 768

		# Check for Partial Data.
		if self.partial_source: source, self.source_mask = farthest_subsample_points(source, num_subsampled_points=number_of_points)
		if self.partial_template: template, self.template_mask = farthest_subsample_points(template, num_subsampled_points=number_of_points)

		# Check for Noise in Source Data.
		if self.noise: source = jitter_pointcloud(source)

		if self.use_rri:
			template, source = template.numpy(), source.numpy()
			template = np.concatenate([template, self.get_rri(template - template.mean(axis=0), self.nearest_neighbors)], axis=1)
			source = np.concatenate([source, self.get_rri(source - source.mean(axis=0), self.nearest_neighbors)], axis=1)
			template, source = torch.tensor(template).float(), torch.tensor(source).float()

		igt = self.transforms.igt
		
		if 'use_masknet' in self.additional_params:
			if self.partial_source and self.partial_template:
				return template, source, igt, self.template_mask, self.source_mask
			elif self.partial_source:
				return template, source, igt, self.source_mask
			elif self.partial_template:
				return template, source, igt, self.template_mask
		else:
			return template, source, igt


class RegistrationDataFragments(Dataset):
	def __init__(self, algorithm, number_of_points=1024, additional_params={}):
		super(RegistrationDataFragments, self).__init__()
		available_algorithms = ['PRNet']
		if algorithm in available_algorithms:
			self.algorithm = algorithm
		else:
			raise Exception("Algorithm not available for registration.")

		dataset_path = "/home/cracee/Documents/Optonic_Project/OptonicsObjectReconstruction/registration/data/7_RAMP_order"
		self.data_path = dataset_path
		self._data = self._read_folder(dataset_path)
		self.additional_params = additional_params
		self.use_rri = False
		self.number_of_points = number_of_points
		if self.algorithm == 'PRNet':
			from ..ops.transform_functions import DCPTransform
			self.transforms = DCPTransform(angle_range=45, translation_range=1)

	def __len__(self):
		return len(self._data)

	def _read_folder(self, folder_name):
		onlyfiles = [f for f in os.listdir(folder_name) if isfile(join(folder_name, f))]

		return onlyfiles

	def _read_file(self, item):
		data_path = self._data[item]
		meta_path = self.data_path + "/" + data_path
		point_cloud = io.read_point_cloud(meta_path)
		return np.asarray(point_cloud.points)

	def normalise_pcd_with_respect(self, points_a, points_b):
		centroid = np.mean(points_a, axis=0)
		points_a -= centroid
		furthest_distance_a = np.max(np.sqrt(np.sum(abs(points_a) ** 2, axis=-1)))

		centroid = np.mean(points_b, axis=0)
		points_b -= centroid
		furthest_distance_b = np.max(np.sqrt(np.sum(abs(points_b) ** 2, axis=-1)))

		if furthest_distance_a > furthest_distance_b:
			furthest_distance = furthest_distance_a
		else:
			furthest_distance = furthest_distance_b
		points_a /= furthest_distance
		points_b /= furthest_distance

		return points_a, points_b

	def __getitem__(self, index):
		item_next = (index + 1) % len(self._data)

		# load the first item
		numpy_points_1 = self._read_file(index)
		# load the second item
		numpy_points_2 = self._read_file(item_next)

		idx1 = (np.random.choice(numpy_points_1.shape[0], self.number_of_points, replace=False),)
		numpy_points_1 = numpy_points_1[idx1]
		idx2 = (np.random.choice(numpy_points_2.shape[0], self.number_of_points, replace=False),)
		numpy_points_2 = numpy_points_2[idx2]

		# normalise with scaling both equally with respect to the environment
		numpy_points_1, numpy_points_2 = self.normalise_pcd_with_respect(numpy_points_1, numpy_points_2)

		template = torch.from_numpy(numpy_points_1).float()
		source = torch.from_numpy(numpy_points_2).float()
		igt = self.transforms.get_fake_transform()

		return template, source, igt


class SyntheticFragments(Dataset):
	def __init__(self, algorithm, number_of_points=768, partial=True, additional_params={}):
		super(SyntheticFragments, self).__init__()
		available_algorithms = ['PRNet']
		if algorithm in available_algorithms:
			self.algorithm = algorithm
		else:
			raise Exception("Algorithm not available for registration.")

		dataset_path = "/home/cracee/Documents/Optonic_Project/OptonicsObjectReconstruction/data/Ramp_sphere_upscale.stl"
		self.data_path = dataset_path
		self.additional_params = additional_params
		self.use_rri = False
		self.partial = partial
		self.number_of_points = number_of_points
		if self.algorithm == 'PRNet':
			from ..ops.transform_functions import DCPTransform
			self.transforms = DCPTransform(angle_range=45, translation_range=1)

	def __len__(self):
		return 50

	def normalise_pcd_with_respect(self, points_a, points_b):
		centroid = np.mean(points_a, axis=0)
		points_a -= centroid
		furthest_distance_a = np.max(np.sqrt(np.sum(abs(points_a) ** 2, axis=-1)))

		centroid = np.mean(points_b, axis=0)
		points_b -= centroid
		furthest_distance_b = np.max(np.sqrt(np.sum(abs(points_b) ** 2, axis=-1)))

		if furthest_distance_a > furthest_distance_b:
			furthest_distance = furthest_distance_a
		else:
			furthest_distance = furthest_distance_b
		points_a /= furthest_distance
		points_b /= furthest_distance

		return points_a, points_b

	def __getitem__(self, index):
		data_path = self.data_path

		numpy_pcd, numpy_pcd2 = generate_pointcloud(data_path, self.number_of_points * 2, resampled=True)
		self.transforms.index = index  # for fixed transformations in PCRNet.
		numpy_pcd2 = self.transforms(numpy_pcd2)

		# normalisation step
		# numpy_pcd, numpy_pcd2 = self.normalise_pcd_with_respect(numpy_pcd, numpy_pcd2)

		# downsampling step
		if self.partial:
			source, self.source_mask = farthest_subsample_points(numpy_pcd, num_subsampled_points=self.number_of_points)
			template, self.template_mask = farthest_subsample_points(numpy_pcd2, num_subsampled_points=self.number_of_points)

		igt = self.transforms.igt
		#template = torch.from_numpy(template).float()
		source = torch.from_numpy(source).float()

		return template, source, igt


class SegmentationData(Dataset):
	def __init__(self):
		super(SegmentationData, self).__init__()

	def __len__(self):
		pass

	def __getitem__(self, index):
		pass


class FlowData(Dataset):
	def __init__(self):
		super(FlowData, self).__init__()
		self.pc1, self.pc2, self.flow = self.read_data()

	def __len__(self):
		if isinstance(self.pc1, np.ndarray):
			return self.pc1.shape[0]
		elif isinstance(self.pc1, list):
			return len(self.pc1)
		else:
			raise UnknownDataTypeError

	def read_data(self):
		pass

	def __getitem__(self, index):
		return self.pc1[index], self.pc2[index], self.flow[index]


class SceneflowDataset(Dataset):
	def __init__(self, npoints=1024, root='', partition='train'):
		if root == '':
			BASE_DIR = os.path.dirname(os.path.abspath(__file__))
			DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
			root = os.path.join(DATA_DIR, 'data_processed_maxcut_35_20k_2k_8192')
			if not os.path.exists(root): 
				print("To download dataset, click here: https://drive.google.com/file/d/1CMaxdt-Tg1Wct8v8eGNwuT7qRSIyJPY-/view")
				exit()
			else:
				print("SceneflowDataset Found Successfully!")

		self.npoints = npoints
		self.partition = partition
		self.root = root
		if self.partition=='train':
			self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
		else:
			self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
		self.cache = {}
		self.cache_size = 30000

		###### deal with one bad datapoint with nan value
		self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
		######
		print(self.partition, ': ',len(self.datapath))

	def __getitem__(self, index):
		if index in self.cache:
			pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
		else:
			fn = self.datapath[index]
			with open(fn, 'rb') as fp:
				data = np.load(fp)
				pos1 = data['points1'].astype('float32')
				pos2 = data['points2'].astype('float32')
				color1 = data['color1'].astype('float32')
				color2 = data['color2'].astype('float32')
				flow = data['flow'].astype('float32')
				mask1 = data['valid_mask1']

			if len(self.cache) < self.cache_size:
				self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)

		if self.partition == 'train':
			n1 = pos1.shape[0]
			sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
			n2 = pos2.shape[0]
			sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

			pos1 = pos1[sample_idx1, :]
			pos2 = pos2[sample_idx2, :]
			color1 = color1[sample_idx1, :]
			color2 = color2[sample_idx2, :]
			flow = flow[sample_idx1, :]
			mask1 = mask1[sample_idx1]
		else:
			pos1 = pos1[:self.npoints, :]
			pos2 = pos2[:self.npoints, :]
			color1 = color1[:self.npoints, :]
			color2 = color2[:self.npoints, :]
			flow = flow[:self.npoints, :]
			mask1 = mask1[:self.npoints]

		pos1_center = np.mean(pos1, 0)
		pos1 -= pos1_center
		pos2 -= pos1_center

		return pos1, pos2, color1, color2, flow, mask1

	def __len__(self):
		return len(self.datapath)


if __name__ == '__main__':
	class Data():
		def __init__(self):
			super(Data, self).__init__()
			self.data, self.label = self.read_data()

		def read_data(self):
			return [4,5,6], [4,5,6]

		def __len__(self):
			return len(self.data)

		def __getitem__(self, idx):
			return self.data[idx], self.label[idx]

	cd = RegistrationData('abc')
	import ipdb; ipdb.set_trace()
