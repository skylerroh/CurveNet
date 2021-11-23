"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main_cls.py
@Time: 2018/10/13 10:39 PM

Modified by 
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@Time: 2021/01/21 3:10 PM
"""

from __future__ import print_function
import os
import pandas as pd
import argparse
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from data import ProteinsSampled, ProteinsExtended, ProteinsExtendedWithMask, load_labels
from models.curvenet_cls import CurveNet, LSTMWithMetadata
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, evaluate
import sklearn.metrics as metrics
from torchsummary import summary
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

def _init_():
    # fix random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(6279)

    # prepare file structures
    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')
    if not os.path.exists('../checkpoints/'+args.exp_name):
        os.makedirs('../checkpoints/'+args.exp_name)
    if not os.path.exists('../checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('../checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main_cls.py ../checkpoints/'+args.exp_name+'/main_cls.py.backup')
    os.system('cp models/curvenet_cls.py ../checkpoints/'+args.exp_name+'/curvenet_cls.py.backup')


def sort_length_indices(lengths):
    sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
    sorted_indices = sorted_indices.to(lengths.device)
    return sorted_lengths, sorted_indices


def reorder_batch_by_length(tensor_input, sorted_indices):
    return tensor_input.index_select(0, sorted_indices)


def to_packed_sequence(tensor_input, sorted_lengths, sorted_indices):
    return pack_padded_sequence(reorder_batch_by_length(tensor_input, sorted_indices), sorted_lengths, batch_first=True, enforce_sorted=True)
    
def to_input_tensors(_id, data, shapes, amino_acids, seqvec, lengths, multihot_label):
    _shapes = F.one_hot(shapes.long(), 9)
    _amino_acids = F.one_hot(amino_acids.long(), 21)

    sorted_lengths, length_indices_sorted = sort_length_indices(lengths)
    sorted_lengths = torch.clip(sorted_lengths, max=args.num_points)
    _id = np.array(_id)[length_indices_sorted.cpu().numpy()]
    struct_features = torch.cat((data, _shapes, _amino_acids), dim=2)
    struct_features = reorder_batch_by_length(struct_features, length_indices_sorted)

    struct_features_packed, seqvec, multihot_label = (
        struct_features.to(dtype=torch.float), 
        seqvec.to(device, dtype=torch.float), 
        multihot_label.to(device)
    )
    return _id, struct_features, seqvec, multihot_label
        

def train(args, io):
    train_loader = DataLoader(ProteinsExtendedWithMask(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ProteinsExtendedWithMask(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    io.cprint("Let's use" + str(torch.cuda.device_count()) + "GPUs!")

    # create model

    labelsData = load_labels()
    
    num_classes = labelsData.num_unique
    
    icvec_np = labelsData.ic_vec
    assert icvec_np.size == num_classes
    icvec = torch.from_numpy(icvec_np).to(device)
    pos_weights = torch.from_numpy(labelsData.pos_weights).to(device)
    
    model = LSTMWithMetadata(k=16, num_classes=num_classes, num_input_to_curvenet=args.num_points).to(device, dtype=torch.float)
    model = nn.DataParallel(model)
    # print(summary(model, [(32, 4000,3), (32,4000,9), (32, 4000,21), (32, 4000, 1024)]))

    if args.use_sgd:
        io.cprint("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        io.cprint("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = MultiStepLR(opt, [120, 160], gamma=0.1)
    
    criterion = lambda x, y: cal_loss(x, y, smoothing=0.0, label_weights=icvec, pos_weights=pos_weights)

    best_test_loss = float("inf")
    best_metrics = None
    for epoch in range(args.epochs):
        ####################
        # Train
        ########s############
        train_loss = 0.0
        count = 0.0
        model.train()
        train_prob = []
        train_true = []
        for _id, data, shapes, amino_acids, seqvec, lengths, multihot_label in tqdm(train_loader):

            _id, struct_features, seqvec, multihot_label = to_input_tensors(_id, data, shapes, amino_acids, seqvec, lengths, multihot_label)
        
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(struct_features, sorted_lengths, seqvec)[0]
            probs = torch.sigmoid(logits)
            loss = criterion(logits, multihot_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(multihot_label.cpu().numpy())
            train_prob.append(probs.detach().cpu().numpy())
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_prob = np.concatenate(train_prob)
        
        train_eval_metrics = evaluate(train_true, train_prob, icvec_np, nth=10)
        outstr = 'Train %d, loss: %.6f, ' % (epoch, train_loss*1.0/count) + "train metrics: {}".format({k: "%.6f" % v for k, v in train_eval_metrics.items()})
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_prob = []
        test_true = []
        with torch.no_grad():   # set all 'requires_grad' to False
            for _id, data, shapes, amino_acids, seqvec, lengths, multihot_label in tqdm(test_loader):
                
                _id, struct_features, seqvec, multihot_label = to_input_tensors(_id, data, shapes, amino_acids, seqvec, lengths, multihot_label)
                
                batch_size = data.size()[0]
                logits = model(data, shapes, amino_acids, seqvec)[0]
                probs = torch.sigmoid(logits)
                loss = criterion(logits, multihot_label)
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(multihot_label.cpu().numpy())
                test_prob.append(probs.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_prob = np.concatenate(test_prob)
        
        test_eval_metrics = evaluate(test_true, test_prob, icvec_np, nth=50)
        outstr = 'Test %d, loss: %.6f, ' % (epoch, test_loss*1.0/count) + "test metrics: {}".format({k: "%.6f" % v for k, v in test_eval_metrics.items()})
        io.cprint(outstr)
        if test_loss*1.0/count <= best_test_loss:
            best_test_loss = test_loss*1.0/count
            best_metrics = test_eval_metrics
            torch.save(model.state_dict(), '../checkpoints/%s/models/model.t7' % args.exp_name)
        io.cprint('best: %.4f, ' % best_test_loss + "test metrics: {}".format({k: "%.6f" % v for k, v in best_metrics.items()}))

        
def test(args, io):
    test_loader = DataLoader(Proteins(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = LSTMWithMetadata(k=16, num_classes=num_classes, num_input_to_curvenet=args.num_points).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))

    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)[0]
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f'%(test_acc)
    io.cprint(outstr)
    
    
def train_all(args, io):
    train_loader = DataLoader(ProteinsExtendedWithMask(partition='all_with_labels', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    device = torch.device("cuda" if args.cuda else "cpu")
    io.cprint("Let's use" + str(torch.cuda.device_count()) + "GPUs!")

    # create model
    labelsData = load_labels()
    
    num_classes = labelsData.num_unique
    
    icvec_np = labelsData.ic_vec
    assert icvec_np.size == num_classes
    icvec = torch.from_numpy(icvec_np).to(device)
    pos_weights = torch.from_numpy(labelsData.pos_weights).to(device)
    
    model = LSTMWithMetadata(k=16, num_classes=num_classes, num_input_to_curvenet=args.num_points, device=device).to(device)
    model = nn.DataParallel(model)

    if args.use_sgd:
        io.cprint("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        io.cprint("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = MultiStepLR(opt, [120, 160], gamma=0.1)
    
    criterion = lambda x, y: cal_loss(x, y, smoothing=0.0, label_weights=icvec, pos_weights=pos_weights)

    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_prob = []
        train_true = []
        for _id, data, seqvec, multihot_label in train_loader:
            data, seqvec, multihot_label = data.to(device, dtype=torch.float), seqvec.to(device, dtype=torch.float), multihot_label.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data, seqvec)[0]
            probs = torch.sigmoid(logits)
            loss = criterion(logits, multihot_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(multihot_label.cpu().numpy())
            train_prob.append(probs.detach().cpu().numpy())
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_prob = np.concatenate(train_prob)
        
        train_eval_metrics = evaluate(train_true, train_prob, icvec_np, nth=10)
        outstr = 'Train %d, loss: %.6f, ' % (epoch, train_loss*1.0/count) + "train metrics: {}".format({k: "%.6f" % v for k, v in train_eval_metrics.items()})
        io.cprint(outstr)
        if (epoch+1) % 25 == 0:
            torch.save(model.state_dict(), f'../checkpoints/{args.exp_name}/models/model_{epoch+1}.t7')
    
    

def embed(args, io):
    all_loader = DataLoader(ProteinsExtendedWithMask(partition='all', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    num_classes = all_loader.dataset.num_label_categories
    #Try to load models
    model = LSTMWithMetadata(k=16, num_classes=num_classes, num_input_to_curvenet=args.num_points, device=device).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    model = model.eval()
    count = 0.0
    _protein_ids = []
    _embeddings = []
    _probs = []
    _labels = []
    for protein_id, data, label in tqdm.tqdm(all_loader):
        data, label = data.float().to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits, embs = model(data)
        probs = torch.sigmoid(logits)
        
        _protein_ids.append(protein_id)
        _embeddings.append(embs.detach().cpu().numpy())
        _probs.append(probs.detach().cpu().numpy())
        _labels.append(label.detach().cpu().numpy())
    
    protein_ids = np.concatenate(_protein_ids)
    embeddings = np.concatenate(_embeddings)
    labels = np.concatenate(_labels)
    probs = np.concatenate(_probs)
    
    [print(arr[0]) for arr in (protein_ids, embeddings, labels, probs)]
    df = pd.DataFrame({"protein_id": protein_ids, "curvenet_embeddings": embeddings.tolist(), "goa_parent_labels": labels.tolist(), "pred_prob": probs.tolist()})
    df.to_parquet(f"../checkpoints/{args.exp_name}/curvenet_embedding.parquet")
    


if __name__ == "__main__":
    def str2bool(value):
        return value.lower() == 'true'
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=str2bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', action="store_true", default=False,
                        help='enables CUDA training')
    parser.add_argument('--eval', action="store_true", default=False,
                        help='evaluate the model')
    parser.add_argument('--embed', action="store_true", default=False,
                        help='embed from trained model')
    parser.add_argument('--train_all', action="store_true", default=False,
                        help='embed from trained model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    seed = np.random.randint(1, 10000)

    _init_()

    if args.eval:
        io = IOStream('../checkpoints/' + args.exp_name + '/eval.log')
    elif args.embed:
        io = IOStream('../checkpoints/' + args.exp_name + '/embed.log')
    elif args.train_all:
        io = IOStream('../checkpoints/' + args.exp_name + '/train_all.log')
    else:
        io = IOStream('../checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    io.cprint('random seed is: ' + str(seed))
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    else:
        io.cprint('Using CPU')

    if args.eval:
        with torch.no_grad():
            test(args, io)
    elif args.embed:
        with torch.no_grad():
            embed(args, io)
    elif args.train_all:
        train_all(args, io)
    else:
        train(args, io)
        
