# coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import mmformer as models
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser, criterions
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
from predict import AverageMeter, test_softmax

from Meta_AMF import Adaptive_Modality_Fuser , MetaNetwork

from random_mask import random_mask

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default=None,
                    type=str)
parser.add_argument('--dataname', default='BRATS2020', type=str)
parser.add_argument('--savepath', default=None,
                    type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--TEMP', default=8, type=float)
parser.add_argument('--mask_ratio', default=0.2, type=float)
parser.add_argument('--alpha', default=5.0, type=float)
parser.add_argument('--lambda_ensemble', default=0.1, type=float)
parser.add_argument('--lambda_imputation_consistency', default=0.01, type=float)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
         [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True],
         [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair',
             't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
             'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
             'flairt1cet1t2']
print(masks_torch.int())


def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print('dataset is error')
        exit(0)
    model = models.Model(num_cls=num_cls)
    print(model)
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)

    meta_network = MetaNetwork(input_features_dim=num_cls * 4).cuda()

    train_params = [
        {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': meta_network.parameters(), 'lr': args.lr, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ##########Setting data
    train_file = 'train.txt'
    test_file = 'test.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls,
                                  train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Evaluate
    if args.resume is not None:
        checkpoint = torch.load(args.pretrain, weights_only=False)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        test_score = AverageMeter()
        with torch.no_grad():
            logging.info('###########test set wi post process###########')
            for i, mask in enumerate(masks[::-1]):
                logging.info('{}'.format(mask_name[::-1][i]))
                dice_score = test_softmax(
                    test_loader,
                    model,
                    dataname=args.dataname,
                    feature_mask=mask,
                    mask_name=mask_name[::-1][i])
                test_score.update(dice_score)
            logging.info('Avg scores: {}'.format(test_score.avg))
            exit(0)
    startepoch = 0
    ################ Pretrain
    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain, weights_only=False)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        startepoch = checkpoint['epoch']

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = args.iter_per_epoch
    train_iter = iter(train_loader)
    temp = args.TEMP
    mask_rato = args.mask_ratio
    alpha=args.alpha
    lambda_ensemble=args.lambda_ensemble
    lambda_imputation_consistency=args.lambda_imputation_consistency
    max_iter = args.num_epochs * iter_per_epoch
    # print('startepoch', startepoch)
    for epoch in range(args.num_epochs):
        # epoch = startepoch
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch + 1))
        b = time.time()
        for i in range(iter_per_epoch):

            step = (i + 1) + epoch * iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True
            fuse_pred, sep_preds, prm_preds, seg_logits = model(x, mask)

            ###Loss compute
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            all_logits = [seg_logits[0], seg_logits[1], seg_logits[2],
                          seg_logits[3]]  # [flair_logits, t1ce_logits, t1_logits, t2_logits]
            all_preds = sep_preds  # [flair_pred, t1ce_pred, t1_pred, t2_pred]

            dice_losses = []
            for i_modality in range(len(all_preds)):
                dice_losses.append(criterions.dice_loss(all_preds[i_modality], target, num_cls=num_cls))

            # ### ensemble_loss
            # dice_losses_tensor = torch.stack(dice_losses)  # shape: [num_modals]
            # confidence_weights = F.softmax(-alpha * dice_losses_tensor, dim=0)  # shape: [num_modals]
            # ensemble_logits = torch.zeros_like(all_logits[0]).cuda().float()
            # for i_modality in range(len(all_logits)):
            #
            #     ensemble_logits += confidence_weights[i_modality] * all_logits[i_modality]
            # ensemble_cross_loss = criterions.softmax_weighted_loss(ensemble_logits, target, num_cls=num_cls)
            # ensemble_dice_loss = criterions.dice_loss(ensemble_logits, target, num_cls=num_cls)
            # ensemble_loss = ensemble_cross_loss + ensemble_dice_loss
            # ### ensemble_loss

            ### meta
            meta_input_features = torch.cat([p.mean(dim=(-1, -2, -3)) for p in sep_preds], dim=1)
            T1_dynamic, T2_dynamic, fusion_balance_weight, beta_param, alpha_param = meta_network(meta_input_features)
            final_soft_target_logits = Adaptive_Modality_Fuser(
                all_logits[0], all_logits[1], all_logits[2], all_logits[3],
                T1_dynamic, T2_dynamic, fusion_balance_weight, beta_param, alpha_param
            )

            soft_target = final_soft_target_logits
            soft_sep_cross_loss = torch.zeros(1).cuda().float()

            soft_target_softmax = F.softmax(soft_target / temp, dim=1)

            soft_target_f, flair_logits_masked = random_mask(soft_target_softmax, all_logits[0], mask_rato)

            soft_sep_cross_loss += criterions.softmax_weighted_loss(flair_logits_masked, soft_target_f, num_cls=num_cls)

            soft_target_t1c, t1ce_logits_masked = random_mask(soft_target_softmax, all_logits[1], mask_rato)
            soft_sep_cross_loss += criterions.softmax_weighted_loss(t1ce_logits_masked, soft_target_t1c,
                                                                    num_cls=num_cls)

            soft_target_t1, t1_logits_masked = random_mask(soft_target_softmax, all_logits[2], mask_rato)
            soft_sep_cross_loss += criterions.softmax_weighted_loss(t1_logits_masked, soft_target_t1, num_cls=num_cls)

            soft_target_t2, t2_logits_masked = random_mask(soft_target_softmax, all_logits[3], mask_rato)
            soft_sep_cross_loss += criterions.softmax_weighted_loss(t2_logits_masked, soft_target_t2, num_cls=num_cls)

            soft_sep_loss = soft_sep_cross_loss
            ### meta


            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss

            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss

            ### Self-supervision
            consistency_loss_imputation = torch.zeros(1).cuda().float()
            full_modal_mask = masks_torch[-1].unsqueeze(0).expand(mask.size(0), -1)  # B x num_modals
            model.module.is_training = False
            with torch.no_grad():
                fuse_pred_aug = model(x, full_modal_mask)
            model.module.is_training = True
            student_logits = fuse_pred
            teacher_logits = fuse_pred_aug
            consistency_loss_imputation += F.kl_div(
                F.log_softmax(student_logits / temp, dim=1),
                F.softmax(teacher_logits / temp, dim=1),
                reduction='batchmean'
            )
            ### Self-supervision


            loss = fuse_loss + 0.7 * sep_loss + prm_loss + soft_sep_loss * 0.3 * temp * temp / 10 + consistency_loss_imputation*lambda_imputation_consistency

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)
            writer.add_scalar('soft_sep_cross_loss', soft_sep_cross_loss.item(), global_step=step)
            writer.add_scalar('fusion_balance_weight', fusion_balance_weight.mean().item(), global_step=step)
            writer.add_scalar('fusion_balance_weight', fusion_balance_weight.mean().item(), global_step=step)
            writer.add_scalar('beta_param', beta_param.mean().item(), global_step=step)
            writer.add_scalar('alpha_param', alpha_param.mean().item(), global_step=step)
            # writer.add_scalar('ensemble_cross_loss', ensemble_cross_loss.item(), global_step=step)
            # writer.add_scalar('ensemble_dice_loss', ensemble_dice_loss.item(), global_step=step)
            # writer.add_scalar('consistency_loss_imputation', consistency_loss_imputation.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch + 1), args.num_epochs, (i + 1), iter_per_epoch,
                                                                  loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            msg += 'softsepcross:{:.4f},'.format(soft_sep_cross_loss.item())
            # msg += 'ensemblecross:{:.4f}, ensembledice:{:.4f},'.format(ensemble_cross_loss.item(),
            #                                                           ensemble_dice_loss.item())
            # msg += 'consist_imput:{:.4f},'.format(consistency_loss_imputation.item())
            logging.info(msg)

        logging.info('train time per epoch: {}'.format(time.time() - b))

        ##########model save
        file_name = os.path.join(ckpts, 'model_300.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            file_name)

        if (epoch + 1) % 50 == 0 or (epoch >= (args.num_epochs - 10)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch + 1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

    msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
    logging.info(msg)

    ##########Evaluate the last epoch model
    test_score = AverageMeter()
    with torch.no_grad():
        logging.info('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            logging.info('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                test_loader,
                model,
                dataname=args.dataname,
                feature_mask=mask)
            test_score.update(dice_score)
        logging.info('Avg scores: {}'.format(test_score.avg))


if __name__ == '__main__':
    main()

