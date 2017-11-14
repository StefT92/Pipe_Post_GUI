import numpy as np
import librosa
import matplotlib.pyplot as plt

from IPython import embed


def normalize(sig):
    M = np.max(np.abs(sig))
    return sig * (1. / M)


def LSD(fft_target, fft_est):

    fft_target = np.abs(fft_target)
    fft_est = np.abs(fft_est)
    LSD = np.sqrt(np.mean((10*np.log10(np.abs(fft_target))-10*np.log10(np.abs(fft_est)))**2))

    return LSD


def RSD(spec_target, spec_est):
    
    spec_target=np.abs(spec_target)
    spec_est = np.abs(spec_est)
    RSD = np.mean(np.sqrt((np.sum((spec_target-spec_est)**2,axis=0))/((np.sum((spec_target)**2,axis=0))+1e-6)))

    # plt.subplot(221)
    # plt.imshow(np.abs(spec_target), aspect='auto', cmap='gray_r')
    # plt.subplot(222)
    # plt.plot((np.sum(np.abs(spec_target) ** 2, axis=0)),'k')
    # plt.plot(np.sum(np.abs(spec_target-spec_est)**2,axis=0),'gray')
    # plt.grid(True)
    # plt.subplot(223)
    # plt.imshow(np.abs(spec_est), aspect='auto', cmap='gray_r')
    # plt.subplot(224)
    # plt.plot(np.sqrt((np.sum((spec_target-spec_est)**2,axis=0))/((np.sum((spec_target)**2,axis=0))+1e-6)),'k')
    # plt.show()

    return RSD
# ###MAIN MAIN MAIN MAIN
# fs = 31250
# NFFT=8192
# hop_length=NFFT/8
# RSD_v=[]
# LSD_v=[]
#
# path_target_folder='/media/stefano/DATA/Flue_Pipe_Prove_Network_Output/REGISTRI_FLUE/8_Principale_Stentor_IT[FT2]_[WE80]_[AL1]_[ID3511]/'
# path_est_folder= '/media/stefano/DATA/Flue_Pipe_Prove_Network_Output/TR_CNN/TR_30_reg/TR_2_mini_80/8_Principale_Stentor_IT[FT2]_[WE80]_[AL1]_[ID3511]/'
# # path_est_folder = '/media/stefano/DATA/Flue_Pipe_Prove_Network_Output/ESTIMATED_REGISTERS/8_Principale_Stentor_IT[FT2]_[WE80]_[AL1]_[ID3511]/'
#
# for i in xrange(1,74):
#     if i<10:
#         n_str='0'+str(i)
#     else:
#         n_str=str(i)
#     path_target = path_target_folder+'8_Principale_Stentor_IT[FT2]_[WE80]_[AL1]_[ID3511]_'+n_str+'.wav'
#     path_est =path_est_folder+'8_Principale_Stentor_IT[FT2]_[WE80]_[AL1]_[ID3511]_'+n_str+'.wav'
#
#
#     audio_target = normalize(librosa.load(path_target,fs)[0])
#     audio_est = normalize(librosa.load(path_est,fs)[0])
#
#     spec_target=librosa.stft(audio_target,n_fft=NFFT,hop_length=hop_length,win_length=NFFT)
#     spec_est=librosa.stft(audio_est,n_fft=NFFT,hop_length=hop_length,win_length=NFFT)
#
#     RSD_v.append(RSD(spec_target, spec_est))
#     LSD_v.append(LSD(np.fft.fft(audio_target,NFFT),np.fft.fft(audio_est,NFFT)))
#
# print 'RSD: ',RSD_v
# print 'LSD: ',LSD_v
#
# plt.subplot(211)
# plt.title('Relative Spectral Distance')
# plt.plot(RSD_v,'k+-')
# plt.grid(True)
# plt.subplot(212)
# plt.title('Log Spectral Distance')
# plt.plot(LSD_v,'k+-')
# plt.grid(True)
# plt.show()
