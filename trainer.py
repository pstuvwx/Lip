import argparse

import torch
import torch.nn as nn
import squib.updaters.updater
import squib.trainer.trainer

import dataset_jvs
import model


def Updater(net:nn.Module,
            opt=None,
            tag:str=''):
    loss_func = nn.CrossEntropyLoss()
    def _update(x, t):
        if opt is None:
            net.eval()
        else:
            net.train()

        y = net(x)
        l = loss_func(y, t)

        log = {
            'loss' :float(l),
        }

        return l, log
    
    upd = squib.updaters.updater.StanderdUpdater(loss_func=_update,
                                                 optimizer=opt,
                                                 tag      =tag)
    return upd



def main(args):
    device = torch.device(args.device)

    net = model.ConvModel(args.n_win//2, args.n_class, args.n_unit,
                                                      args.n_layer, args.ksize)
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), args.lr)

    pathes_tr, pathes_vl = dataset_jvs.listup_pathes(args.dir_jvs)

    ds_tr = dataset_jvs.AudioDataset(pathes_tr, args.n_win, args.n_hop,
                                          args.eps, args.n_frames, args.n_skip)
    ds_vl = dataset_jvs.AudioDataset(pathes_vl, args.n_win, args.n_hop,
                                          args.eps, args.n_frames, args.n_skip)

    loader_tr = torch.utils.data.DataLoader(ds_tr, args.batch_size,
                                                shuffle=True,  drop_last=True)
    loader_vl = torch.utils.data.DataLoader(ds_vl, args.batch_size,
                                                shuffle=False, drop_last=False)

    upd_tr = Updater(net, opt, 'train')
    upd_vl = Updater(net, tag='val')

    trainer = squib.trainer.trainer.Trainer(loader_tr, upd_tr, device,args.dst)
    trainer.add_evaluation(loader_vl, upd_vl)
    trainer.log_report(['train/loss', 'val/loss'],
                       plots={'loss.png':['train/loss', 'val/loss']})

    trainer.save_model(path ='models/models_{epoch}.pth',
                       model=net)
    trainer.save_trainer(path  ='trainer.pth',
                         models={'model':net, 'opt':opt})
    
    trainer.run((args.n_epoch, 'epoch'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_jvs',      type=str)
    parser.add_argument('--n_win',      type=int,   default=1024)
    parser.add_argument('--n_hop',      type=int,   default=128)
    parser.add_argument('--n_class',    type=int,   default=12)
    parser.add_argument('--n_unit',     type=int,   default=256)
    parser.add_argument('--n_layer',    type=int,   default=4)
    parser.add_argument('--ksize',      type=int,   default=3)
    parser.add_argument('--n_frames',   type=int,   default=128)
    parser.add_argument('--n_skip',     type=int,   default=3)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--eps',        type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int,   default=128)
    parser.add_argument('--n_epoch',    type=int,   default=100)
    parser.add_argument('--device',     type=str,   default='cpu')
    parser.add_argument('--dst',        type=str,   default='./result')

    args = parser.parse_args()

    main(args)
