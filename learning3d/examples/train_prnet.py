import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pynvml import *

from torch.autograd import Variable

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))
	
from .. models import PRNet
from .. data_utils import RegistrationData, ModelNet40Data

def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.exp_name):
		os.makedirs('checkpoints/' + args.exp_name)
	if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
		os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
	os.system('cp train_dcp.py checkpoints' + '/' + args.exp_name + '/' + 'train.py.backup')

class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()

def get_transformations(igt):
	R_ba = igt[:, 0:3, 0:3]								# Ps = R_ba * Pt
	translation_ba = igt[:, 0:3, 3].unsqueeze(2)		# Ps = Pt + t_ba
	R_ab = R_ba.permute(0, 2, 1)						# Pt = R_ab * Ps
	translation_ab = -torch.bmm(R_ab, translation_ba)	# Pt = Ps + t_ab
	return R_ab, translation_ab, R_ba, translation_ba

def test_one_epoch(device, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(test_loader)):
		template, source, igt = data
		transformations = get_transformations(igt)
		transformations = [t.to(device) for t in transformations]
		R_ab, translation_ab, R_ba, translation_ba = transformations

		template = template.to(device)
		source = source.to(device)
		igt = igt.to(device)

		output = model(template, source, R_ab, translation_ab.squeeze(2))
		loss_val = output['loss']

		test_loss += loss_val.item()
		count += 1

	test_loss = float(test_loss)/count
	return test_loss

def test(args, model, test_loader, textio):
	test_loss = test_one_epoch(args.device, model, test_loader)
	textio.cprint('Validation Loss: %f & Validation Accuracy: %f'%(test_loss, test_accuracy))

def train_one_epoch(device, model, train_loader, optimizer):
	model.train()
	train_loss = 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(train_loader)):
		optimizer.zero_grad()
		get_gpu_status("first thing of the batch loop")
		template, source, igt = data
		get_gpu_status("Now is the data available")
		transformations = get_transformations(igt)
		transformations = [t.to(device) for t in transformations]
		get_gpu_status("Now is the transformation on the device")
		R_ab, translation_ab, R_ba, translation_ba = transformations

		template = template.to(device)
		source = source.to(device)
		igt = igt.to(device)
		get_gpu_status("Now is template and source on gpu")

		output = model(template, source, R_ab, translation_ab.squeeze(2))
		get_gpu_status("now is the output calculated")
		loss_val = output['loss']

		# forward + backward + optimize

		loss_val.backward()
		optimizer.step()
		get_gpu_status("after optimizer step")

		train_loss += loss_val.item()
		count += 1

	train_loss = float(train_loss)/count
	return train_loss

def train(args, model, train_loader, test_loader, boardio, textio, checkpoint):
	learnable_params = filter(lambda p: p.requires_grad, model.parameters())
	if args.optimizer == 'Adam':
		#optimizer = torch.optim.Adam(learnable_params, lr=0.000125)
		optimizer = torch.optim.Adam(learnable_params)
	else:
		optimizer = torch.optim.SGD(learnable_params, lr=0.1)

	if checkpoint is not None:
		min_loss = checkpoint['min_loss']
		optimizer.load_state_dict(checkpoint['optimizer'])

	best_test_loss = np.inf

	get_gpu_status("starting the epoch loop")

	for epoch in range(args.start_epoch, args.epochs):
		train_loss = train_one_epoch(args.device, model, train_loader, optimizer)
		test_loss = test_one_epoch(args.device, model, test_loader)

		if test_loss<best_test_loss:
			best_test_loss = test_loss
			snap = {'epoch': epoch + 1,
					'model': model.state_dict(),
					'min_loss': best_test_loss,
					'optimizer' : optimizer.state_dict(),}
			torch.save(snap, 'checkpoints/%s/models/best_model_snap.t7' % (args.exp_name))
			torch.save(model.state_dict(), 'checkpoints/%s/models/best_model.t7' % (args.exp_name))
			#torch.save(model.feature_model.state_dict(), 'checkpoints/%s/models/best_ptnet_model.t7' % (args.exp_name))

		torch.save(snap, 'checkpoints/%s/models/model_snap.t7' % (args.exp_name))
		torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % (args.exp_name))
		#torch.save(model.feature_model.state_dict(), 'checkpoints/%s/models/ptnet_model.t7' % (args.exp_name))
		
		boardio.add_scalar('Train Loss', train_loss, epoch+1)
		boardio.add_scalar('Test Loss', test_loss, epoch+1)
		boardio.add_scalar('Best Test Loss', best_test_loss, epoch+1)

		textio.cprint('EPOCH:: %d, Traininig Loss: %f, Testing Loss: %f, Best Loss: %f'%(epoch+1, train_loss, test_loss, best_test_loss))

def options():
	parser = argparse.ArgumentParser(description='Point Cloud Registration')
	parser.add_argument('--exp_name', type=str, default='current_exp_prnet', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--dataset_path', type=str, default='ModelNet40',
						metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

	# settings for input data
	parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
						metavar='DATASET', help='dataset type (default: modelnet)')
	parser.add_argument('--emb_dims', default=512, type=int,
						metavar='K', help='dim. of the feature vector (default: 1024)')
	parser.add_argument('--num_iterations', default=3, type=int,
						help='Number of Iterations')

	# settings for on training
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=4, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--batch_size_test', default=2, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--epochs', default=250, type=int,
						metavar='N', help='number of total epochs to run')
	parser.add_argument('--start_epoch', default=0, type=int,
						metavar='N', help='manual epoch number (useful on restarts)')
	parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
						metavar='METHOD', help='name of an optimizer (default: Adam)')
	parser.add_argument('--resume', default='', type=str,
						metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
	parser.add_argument('--pretrained', default='', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args


def get_gpu_status(message):
	return
	h = nvmlDeviceGetHandleByIndex(0)
	info = nvmlDeviceGetMemoryInfo(h)
	print(message)
	print(f'total    : {info.total / 1000000000} GB')
	print(f'free     : {info.free / 1000000000} GB')
	print(f'used     : {info.used / 1000000000} GB')


def main():
	args = options()

	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	nvmlInit()

	boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
	_init_(args)

	textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
	textio.cprint(str(args))

	
	trainset = RegistrationData('PRNet', ModelNet40Data(train=True), partial_source=True, partial_template=True)
	testset = RegistrationData('PRNet', ModelNet40Data(train=False), partial_source=True, partial_template=True)
	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
	test_loader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Create PointNet Model.
	model = PRNet(emb_dims=args.emb_dims, num_iters=args.num_iterations)

	nvmlInit()

	get_gpu_status("The GPU before the model")
	model.to(args.device)
	get_gpu_status("After the model is loaded")

	checkpoint = None
	if args.resume:
		assert os.path.isfile(args.resume)
		print("Resuming the Training!")
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model'])

	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		print("Using a pretrained model!")
		model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))

	model.to(args.device)

	if args.eval:
		test(args, model, test_loader, textio)
	else:
		get_gpu_status("Going into the train function")
		train(args, model, train_loader, test_loader, boardio, textio, checkpoint)


if __name__ == '__main__':
	main()