#!/usr/bin/env python
# coding=utf-8

from .parameters import params2path, param2dataset, param2model, get_dataset_param
from .utils import getDevice, setup_logger
from .evaluate import evaluate

import numpy as np
from tqdm import tqdm
import torch, os, time, shutil, copy
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader, DataListLoader

def running_test(prefix, net, test_data_loader, device,
                 running_metric_name_list, test_loss_hist,
                 epoch, step, global_step,
                 summary_writer, logger):
    test_metric_list, _ = evaluate(net, test_data_loader, device,
                                parallel_flag=False,
                                # post_processing_flag=False,
                                # no_post_processing_flag=True,
                                metric_name_list=running_metric_name_list)
    for mn in running_metric_name_list:
        test_loss_hist[mn].append(np.mean(test_metric_list[mn]))
        summary_writer.add_scalar(prefix+'%s'%mn,test_loss_hist[mn][-1],
                                  global_step=global_step)
    logger.info('epoch {} step {}: {}'.format(epoch, step, ', '.join([prefix+'{}={:.2f}'.format(mn, ml[-1])
                                                                      for mn, ml in test_loss_hist.items()])))


def train_model(dataset_param, model_param, general_param):
    lr = general_param.learning_rate
    epoch_num = general_param.epoch_num
    batch_size = general_param.batch_size
    save_freq = general_param.save_freq
    log_freq = general_param.log_freq
    resume_flag = general_param.resume_flag
    running_metric_name_list = general_param.running_metric_name_list

    device = getDevice()
    dataset_param.device = torch.device('cpu')

    # setup logger and summary_writer
    log_dir_name = params2path(dataset_param, model_param)
    if resume_flag and len(os.listdir(log_dir_name)) > 0:
        log_dir_base = sorted(os.listdir(log_dir_name))[-1]
    else:
        log_dir_base = time.strftime('%b-%d-%H:%M:%S')
    log_dir_name = os.path.join(log_dir_name, log_dir_base)
    summary_writer = SummaryWriter(log_dir=log_dir_name, comment='')
    log_dir = summary_writer.log_dir
    shutil.copy(__file__, log_dir)
    logger = setup_logger('', log_dir)
    logger.info(str(dataset_param))
    logger.info(str(model_param))
    logger.info(str(general_param))
    logger.info(str(device))

    # dataset
    dataset_size = dataset_param.size

    start_time = time.time()
    train_dataset = param2dataset(dataset_param, train_flag=True)
    logger.info('train_dataset generated: %.2f seconds'%(time.time()-start_time))

    start_time = time.time()
    test_dataset_param = copy.deepcopy(dataset_param)
    test_dataset_param.size = 1000
    test_dataset = param2dataset(test_dataset_param, train_flag=False)
    logger.info('test_dataset generated: %.2f seconds'%(time.time()-start_time))

    start_time = time.time()

    data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size)

    # build network
    net = param2model(train_dataset, model_param)
    net = net.to(device)
    logger.info(str(net))

    # optimizer related
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    # Initialization
    epoch, global_step = 0, 0
    layer_num_hist, loss_hist = [], []
    test_loss_hist = {mn:[] for mn in running_metric_name_list}
    if resume_flag and len([fn for fn in os.listdir(log_dir_name) if 'model' in fn and '.pth' in fn]) > 0:
        model_filename = [fn for fn in os.listdir(log_dir_name) if 'model' in fn and '.pth' in fn]
        if len(model_filename):
            model_path = os.path.join(log_dir_name, sorted(model_filename)[-1])
            checkpoint = torch.load(model_path, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'], )
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss_hist, epoch =  checkpoint['loss_hist'], checkpoint['epoch']
            test_loss_hist = checkpoint['test_loss_hist']
            layer_num_hist = checkpoint['layer_num_hist']
            global_step = (int(dataset_size/batch_size)+(dataset_size%batch_size>0))*epoch

    for epoch in range(epoch+1, epoch_num+1):
        last_log_time = time.time()
        last_log_step = 0
        net.train()
        print(data_loader)
        for step,data in tqdm(enumerate(data_loader)):
            global_step += 1
            optimizer.zero_grad()
            data = data.to(device)
            pred, layer_num, residual_confidence = net(data, output_layer_num_flag=True, output_residual_confidence_flag=True)
            y = data.y
            loss = F.mse_loss(pred.reshape(y.size()), y) + net.tao*torch.sum(residual_confidence)
            loss.backward()
            optimizer.step()
            summary_writer.add_scalar('loss',loss, global_step=global_step)
            loss_hist.append(loss.item())
            layer_num_hist.append(layer_num)
            if (step+1) % log_freq == 0:
                logger.info('epoch {} step {}: {:.1f} step/s loss={:.2f}'.format(
                    epoch, step, (step - last_log_step) / (
                    time.time()-last_log_time), loss_hist[-1]))
                last_log_step = step
                last_log_time = time.time()
        running_test('test_', net, test_data_loader, device,
                     running_metric_name_list, test_loss_hist,
                     epoch, step, global_step,
                     summary_writer, logger)
        if epoch % save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_hist':loss_hist,
                'test_loss_hist':test_loss_hist,
                'layer_num_hist':layer_num_hist,
            }, os.path.join(log_dir_name, 'model_{:6d}.pth'.format(epoch)))

#======================Testing training=========================================

from .parameters import get_dataset_param, get_model_param, GeneralParam

def test_train_model():
    dataset_param = get_dataset_param(size=1000)
    model_param = get_model_param(architecture_name='ACT1IterGNN')
    general_param = GeneralParam()
    # general_param.resume_flag = True

    train_model(dataset_param, model_param, general_param)

if __name__ == '__main__':
    test_train_model()
