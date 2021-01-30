import argparse
import os
import traceback

import numpy as np
import pyaudio
import pygame
import pygame.locals
import scipy.signal as sig
import torch

import dataset_jvs
import model


def play(path_wav, dir_imgs, n_class, predicted, n_win, n_hop):
    audio = dataset_jvs.load_wav(path_wav)

    imgs = [pygame.image.load(os.path.join(dir_imgs, '%d.png'%i))
                                                       for i in range(n_class)]

    audio  = audio[n_win//2:]
    frames = audio[:len(audio)//n_hop*n_hop].reshape(-1, n_hop)
    frames = (frames*20000).astype(np.int16)

    screen = pygame.display.set_mode((imgs[0].get_width(),
                                      imgs[0].get_height()))
    p      = pyaudio.PyAudio()

    # 録画用に、Rキーを押すまで一時停止
    # stop = True
    # while stop:
    #     for event in pygame.event.get():
    #         if event.type == pygame.locals.QUIT:
    #             pygame.quit()
    #             return
    #         if event.type == pygame.locals.KEYDOWN:
    #             if event.key == pygame.locals.K_r:
    #                 stop = False
    #     pygame.display.update()

    try:
        stream = p.open(format  =pyaudio.paInt16,
                        channels=1,
                        rate    =16000,
                        input   =False,
                        output  =True)

        last = -1
        for w, i in zip(frames, predicted):
            stream.write(w.tobytes())
            if last != i:
                last = i
                screen.blit(imgs[i], (0,0))
                pygame.display.update()
        
        pygame.quit()
        stream.stop_stream()
        stream.close()

    except Exception:
        traceback.print_exc()
    finally:
        p.terminate()


def main(args):
    device = torch.device(args.device)

    net = model.ConvModel(args.n_win//2, args.n_class, args.n_unit,
                                                      args.n_layer, args.ksize)
    net.load_state_dict(torch.load(args.model, map_location='cpu'))
    net = net.to(device)

    x = dataset_jvs.preprocess(args.wavfile, None,
                                        args.n_win, args.n_hop, args.eps, None)

    with torch.no_grad():
        x = torch.FloatTensor([x.T])
        y = net(x)
        y = torch.softmax(y, dim=1).cpu().numpy()[0].T
    
    amax = np.argmax(y, axis=1)
    med  = sig.medfilt(amax, args.k_median).astype(int)

    play(args.wavfile, args.dir_img, args.n_class, med, args.n_win, args.n_hop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('wavfile',     type=str)
    parser.add_argument('model',       type=str)
    parser.add_argument('--n_win',     type=int,   default=1024)
    parser.add_argument('--n_hop',     type=int,   default=128)
    parser.add_argument('--n_class',   type=int,   default=12)
    parser.add_argument('--n_unit',    type=int,   default=256)
    parser.add_argument('--n_layer',   type=int,   default=4)
    parser.add_argument('--ksize',     type=int,   default=3)
    parser.add_argument('--eps',       type=float, default=1e-5)
    parser.add_argument('--device',    type=str,   default='cpu')
    parser.add_argument('--k_median',  type=int,   default=9)
    parser.add_argument('--dir_img',   type=str,   default='./Lip/face')

    args = parser.parse_args()

    main(args)
