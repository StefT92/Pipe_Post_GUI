#!/usr/bin/python

import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import numpy as np
import scipy.signal as SS
import detect_peaks

# normalizza -1/1
def normalize(sig):
    M = np.max(np.abs(sig))
    return sig * (1. / M)


'''
FFT plot utility. Examples:
Y, freqs = plot_fft(signal, Fs, "all", 0.5)

alpha = percent of the window with fade in and fadeout, guidelines:
1: Tukey==Hanning, good for noise, non time varying processes (very good windowing in the frequency domain
0: Tukey==Rectangular, only if you know that there is silence at the beginning and the end, e.g. musical instrument tone with attack and decay
0.25: Compromise, the fade should remove most rectangular window frequency artifacts, but allow to have most of the central portion of the signal available for FFT
Author: L.Gabrielli
'''
def plot_fft(signal, Fs, bins, alpha, plot=True):

    n = len(signal) # length of the signal
    if bins == "all":
        bins = n

    # Windowing
    win = SS.tukey(n, alpha)
    signal = win * signal

    k = np.arange(bins)
    T = float(bins)/float(Fs)
    frq = k/T # two sides frequency range
    frq = frq[range(bins/2)] # one side frequency range

    Y = np.fft.fft(signal, n=bins)/n # fft computing and normalization
    Y = np.abs(Y[range(bins/2)])
    Y = 10 * np.log10(Y)

    if plot == True:
        fig, ax = plt.subplots(1, 1)
        ax.plot(frq,Y,'k') # plotting the spectrum
        ax.set_xlabel('f (Hz)')
        ax.set_ylabel('log|Y|')
        fig.show()

    return Y,frq



# ## MAIN MAIN MAIN ##
#
# fig_ratio = (10,5) # width, height
#
# ### STENTOR VERY BAD
# filename = "originali/8_Principale_Stentor_IT[FT2]_[WE80]_[AL1]_[ID3511]/8_Principale_Stentor_IT[FT2]_[WE80]_[AL1]_[ID3511]_40.wav"
# audio, Fs, enc = scikits.audiolab.wavread(filename)
# #AUDIO PLOT pt1
# plt.figure(figsize=fig_ratio,tight_layout=True)
# plt.subplot(211)
# plt.plot(audio[0:2000],'k')
# plt.tick_params(axis='both', which='major', labelsize=20)
# ory, orx = plot_fft(normalize(audio), Fs, "all", 0, plot=False)
#
# filename = "generati/8_Principale_Stentor_IT[FT2]_[WE80]_[AL1]_[ID3511]_VERY_BAD/8_Principale_Stentor_IT[FT2]_[WE80]_[AL1]_[ID3511]_40.wav"
# audio, Fs, enc = scikits.audiolab.wavread(filename)
# #AUDIO PLOT pt2
# plt.subplot(212)
# plt.plot(1.2 * audio[0:2000],'k') # 1.2* trucco per far comparire il numero giusto di ticks sull'asse 7
# plt.xlabel("Time [samples]", fontsize=24)
# # non funziona: plt.locator_params(axis='y', nticks=6)
# plt.tick_params(axis='both', which='major', labelsize=20)
# figname = "waveform_Stentor_orig_vs_verybad.pdf"
# plt.savefig(figname, format='pdf')
# vby, vbx = plot_fft(normalize(audio), Fs, "all", 0, plot=False)
#
# # ATTENZIONE QUESTI VALGONO ANCHE PER I PLOT SUCCESSIVI
# f_0 = 440.0
# tot_harm = int(np.floor( Fs/2 / f_0 ) ) # max harm to find
# f_toler = f_0 / 10
# nbins = len(audio)
# delta = np.round(nbins * f_toler / (Fs))
# maxprobableindex = [(i * f_0 * nbins / Fs) for i in range(1,tot_harm+1)] # first tot_harm harm
#
# # PEAK DETECT EURISTICO NELL'INTORNO DELLE ARMONICHE, SE CONOSCO LA F0
# orpk = []
# orfr = []
# for i in maxprobableindex:
#     orpk.append(np.amax(ory[int(i-delta):int(i+delta)]))
#     orfr.append( orx[np.argmax(ory[int(i-delta):int(i+delta)])+int(i-delta)] )
#
# vbpk = []
# vbfr = []
# for i in maxprobableindex:
#     vbpk.append(np.amax(vby[int(i-delta):int(i+delta)]))
#     vbfr.append( vbx[np.argmax(vby[int(i-delta):int(i+delta)])+int(i-delta)] )
#
# # PEAK DETECT PIU GENERALE NEL NOSTRO CASO BECCA PIU ROBA SPURIA
# # peaks = detect_peaks.detect_peaks(ory, mph=-60, mpd=(np.round(len(audio) * f_0 / (Fs))-10), ax=orx)
# # orpk = ory[peaks]
# # orfr = orx[peaks]
# #
# # peaks = detect_peaks.detect_peaks(vby, mph=-60, mpd=(np.round(len(audio) * f_0 / (Fs))-10), ax=vbx)
# # vbpk = vby[peaks]
# # vbfr = vbx[peaks]vbpk[15] = vbpk[16] = vbpk[26] = vbpk[27] =  -70
#
# plt.figure(figsize=fig_ratio,tight_layout=True)
# plt.plot(vbx,vby,'grey')
# plt.plot(orx,ory,'k')
# plt.plot(vbfr,vbpk,'kx')
# plt.plot(orfr,orpk,'ko')
# figname = "comparison_Stentor_orig_vs_verybad.pdf"
#
# # SE I PICCHI SONO ALLINEATI
# # CALCOLA MSE
# MSE = np.average(np.square(np.array(orpk)-np.array(vbpk)))
# print 'MSE error for ' + figname + ' is ' + str(MSE)
#
# # CALCOLA MSE10 (first 10 harm)
# MSE10 = np.average(np.square(np.array(orpk[:10])-np.array(vbpk[:10])))
# MSE20 = np.average(np.square(np.array(orpk[:20])-np.array(vbpk[:20])))
# print 'MSE10 error for ' + figname + ' is ' + str(MSE10)
# f = open('comparison_Stentor_orig_vs_verybad.txt', 'w')
# f.write('MSE10: ' + "%0.2f" % MSE10 + 'dB - ' + 'MSE20: ' + "%0.2f" % MSE20 + 'dB - ' + 'MSE: ' + "%0.2f" % MSE + 'dB')  # python will convert \n to os.linesep
# f.close()
# #plt.title('MSE10: ' + "%0.2f" % MSE10 + 'dB - ' + 'MSE: ' + "%0.2f" % MSE + 'dB', fontsize=20)
# plt.xlabel("Frequency [Hz]", fontsize=24)
# plt.ylabel("Magnitude [dB]", fontsize=24)
# plt.tick_params(axis='both', which='major', labelsize=20)
#
#
# # SE I PICCHI NON SONO ALLINEATI (CON PEAK DETECT GENERALE)
# # # CALCOLA MSE
# # diff = list(range(0,tot_harm))
# # for i in range(0,tot_harm): # compare the first 30 harm
# #     if (orx[i] - vbx[i]) <= np.round(nbins * f_toler / (Fs)): # tollero errore di f_toler Hz
# #         diff[i] = (ory[i]-vby[i]) # BUUUUUUG!!!! dovrei fare la diff tra i picchi, cosi faccio la diff tra i primi 30 valori della fft!!!!!!
# # MSE = np.average(np.square(diff))
# # print 'MSE error for ' + figname + ' is ' + str(MSE)
# #
# # # CALCOLA MSE10 (first 10 harm)
# # diff = list(range(0,10))
# # for i in range(0,10): # compare the first 30 harm
# #     if (orx[i] - vbx[i]) <= np.round(nbins * f_toler / (Fs)): # tollero errore di f_toler Hz
# #         diff[i] = (ory[i]-vby[i])
# # MSE10 = np.average(np.square(diff))
# # print 'MSE10 error for ' + figname + ' is ' + str(MSE10)
# # plt.title('MSE: ' + str(MSE) + ' - ' + 'MSE10: ' + str(MSE10))
#
# axes = plt.gca()
# axes.set_ylim([-60, 0])
# plt.savefig(figname, format='pdf')
# #plt.show()
#
#
#
#
#
#
# ### STENTOR OK
# filename = "originali/8_Principale_Stentor_IT[FT2]_[WE80]_[AL1]_[ID3511]/8_Principale_Stentor_IT[FT2]_[WE80]_[AL1]_[ID3511]_40.wav"
# audio, Fs, enc = scikits.audiolab.wavread(filename)
# #AUDIO PLOT pt1
# plt.figure(figsize=fig_ratio,tight_layout=True)
# plt.subplot(211)
# plt.plot(audio[0:2000],'k')
# plt.tick_params(axis='both', which='major', labelsize=20)
# ory, orx = plot_fft(normalize(audio), Fs, "all", 0, plot=False)
#
# filename = "generati/8_Principale_Stentor_IT[FT2]_[WE80]_[AL1]_[ID3511]/8_Principale_Stentor_IT[FT2]_[WE80]_[AL1]_[ID3511]_40.wav"
# audio, Fs, enc = scikits.audiolab.wavread(filename)
# #AUDIO PLOT pt2
# plt.subplot(212)
# plt.plot(audio[0:2000],'k')
# plt.xlabel("Time [samples]", fontsize=24)
# plt.tick_params(axis='both', which='major', labelsize=20)
# figname = "waveform_Stentor_orig_vs_good.pdf"
# plt.savefig(figname, format='pdf')
#
# vby, vbx = plot_fft(normalize(audio), Fs, "all", 0, plot=False)
#
# # PEAK DETECT EURISTICO NELL'INTORNO DELLE ARMONICHE, SE CONOSCO LA F0
# orpk = []
# orfr = []
# for i in maxprobableindex:
#     orpk.append(np.amax(ory[int(i-delta):int(i+delta)]))
#     orfr.append( orx[np.argmax(ory[int(i-delta):int(i+delta)])+int(i-delta)] )
#
# vbpk = []
# vbfr = []
# for i in maxprobableindex:
#     vbpk.append(np.amax(vby[int(i-delta):int(i+delta)]))
#     vbfr.append( vbx[np.argmax(vby[int(i-delta):int(i+delta)])+int(i-delta)] )
#
# # PEAK DETECT PIU GENERALE NEL NOSTRO CASO BECCA PIU ROBA SPURIA
# # peaks = detect_peaks.detect_peaks(ory, mph=-60, mpd=(np.round(len(audio) * f_0 / (Fs))-10), ax=orx)
# # orpk = ory[peaks]
# # orfr = orx[peaks]
# #
# # peaks = detect_peaks.detect_peaks(vby, mph=-60, mpd=(np.round(len(audio) * f_0 / (Fs))-10), ax=vbx)
# # vbpk = vby[peaks]
# # vbfr = vbx[peaks]vbpk[15] = vbpk[16] = vbpk[26] = vbpk[27] =  -70
#
# plt.figure(figsize=fig_ratio,tight_layout=True)
# plt.plot(vbx,vby,'grey')
# plt.plot(orx,ory,'k')
# plt.plot(vbfr,vbpk,'kx')
# plt.plot(orfr,orpk,'ko')
# figname = "comparison_Stentor_orig_vs_good.pdf"
#
# # SE I PICCHI SONO ALLINEATI
# # CALCOLA MSE
# MSE = np.average(np.square(np.array(orpk)-np.array(vbpk)))
# print 'MSE error for ' + figname + ' is ' + str(MSE)
#
# # CALCOLA MSE10 (first 10 harm)
# MSE10 = np.average(np.square(np.array(orpk[:10])-np.array(vbpk[:10])))
# MSE20 = np.average(np.square(np.array(orpk[:20])-np.array(vbpk[:20])))
# print 'MSE10 error for ' + figname + ' is ' + str(MSE10)
# f = open('comparison_Stentor_orig_vs_good.txt', 'w')
# f.write('MSE10: ' + "%0.2f" % MSE10 + 'dB - ' + 'MSE20: ' + "%0.2f" % MSE20 + 'dB - ' + 'MSE: ' + "%0.2f" % MSE + 'dB')  # python will convert \n to os.linesep
# f.close()
#
# #plt.title('MSE10: ' + "%0.2f" % MSE10 + 'dB - ' + 'MSE: ' + "%0.2f" % MSE + 'dB', fontsize=20)
# plt.xlabel("Frequency [Hz]", fontsize=24)
# plt.ylabel("Magnitude [dB]", fontsize=24)
# plt.tick_params(axis='both', which='major', labelsize=20)
#
# # SE I PICCHI NON SONO ALLINEATI (CON PEAK DETECT GENERALE)
# # # CALCOLA MSE
# # diff = list(range(0,tot_harm))
# # for i in range(0,tot_harm): # compare the first 30 harm
# #     if (orx[i] - vbx[i]) <= np.round(nbins * f_toler / (Fs)): # tollero errore di f_toler Hz
# #         diff[i] = (ory[i]-vby[i]) # BUUUUUUG!!!! dovrei fare la diff tra i picchi, cosi faccio la diff tra i primi 30 valori della fft!!!!!!
# # MSE = np.average(np.square(diff))
# # print 'MSE error for ' + figname + ' is ' + str(MSE)
# #
# # # CALCOLA MSE10 (first 10 harm)
# # diff = list(range(0,10))
# # for i in range(0,10): # compare the first 30 harm
# #     if (orx[i] - vbx[i]) <= np.round(nbins * f_toler / (Fs)): # tollero errore di f_toler Hz
# #         diff[i] = (ory[i]-vby[i])
# # MSE10 = np.average(np.square(diff))
# # print 'MSE10 error for ' + figname + ' is ' + str(MSE10)
# # plt.title('MSE: ' + str(MSE) + ' - ' + 'MSE10: ' + str(MSE10))
#
# axes = plt.gca()
# axes.set_ylim([-60, 0])
# plt.savefig(figname, format='pdf')
# #plt.show()
#
#
#
#
#
#
#
# ### HW_DE
# filename = "originali/8_Prinzipal_HW_DE[FT2]_[WE80]_[AL1]_[ID4078]/8_Prinzipal_HW_DE[FT2]_[WE80]_[AL1]_[ID4078]_40.wav"
# audio, Fs, enc = scikits.audiolab.wavread(filename)
# #AUDIO PLOT pt1
# plt.figure(figsize=fig_ratio,tight_layout=True)
# plt.subplot(211)
# plt.plot(audio[0:2000],'k')
# plt.tick_params(axis='both', which='major', labelsize=20)
# ory, orx = plot_fft(normalize(audio), Fs, "all", 0, plot=False)
#
# filename = "generati/8_Prinzipal_HW_DE[FT2]_[WE80]_[AL1]_[ID4078]/8_Prinzipal_HW_DE[FT2]_[WE80]_[AL1]_[ID4078]_40.wav"
# audio, Fs, enc = scikits.audiolab.wavread(filename)
# #AUDIO PLOT pt2
# plt.subplot(212)
# plt.plot(audio[0:2000],'k')
# plt.xlabel("Time [samples]", fontsize=24)
# plt.tick_params(axis='both', which='major', labelsize=20)
# figname = "waveform_HW_DE_orig_vs_generato.pdf"
# plt.savefig(figname, format='pdf')
# vby, vbx = plot_fft(normalize(audio), Fs, "all", 0, plot=False)
#
# # PEAK DETECT EURISTICO NELL'INTORNO DELLE ARMONICHE, SE CONOSCO LA F0
# orpk = []
# orfr = []
# for i in maxprobableindex:
#     orpk.append(np.amax(ory[int(i-delta):int(i+delta)]))
#     orfr.append( orx[np.argmax(ory[int(i-delta):int(i+delta)])+int(i-delta)] )
#
# vbpk = []
# vbfr = []
# for i in maxprobableindex:
#     vbpk.append(np.amax(vby[int(i-delta):int(i+delta)]))
#     vbfr.append( vbx[np.argmax(vby[int(i-delta):int(i+delta)])+int(i-delta)] )
#
# # PEAK DETECT PIU GENERALE NEL NOSTRO CASO BECCA PIU ROBA SPURIA
# # peaks = detect_peaks.detect_peaks(ory, mph=-60, mpd=(np.round(len(audio) * f_0 / (Fs))-10), ax=orx)
# # orpk = ory[peaks]
# # orfr = orx[peaks]
# #
# # peaks = detect_peaks.detect_peaks(vby, mph=-60, mpd=(np.round(len(audio) * f_0 / (Fs))-10), ax=vbx)
# # vbpk = vby[peaks]
# # vbfr = vbx[peaks]vbpk[15] = vbpk[16] = vbpk[26] = vbpk[27] =  -70
#
# plt.figure(figsize=fig_ratio,tight_layout=True)
# plt.plot(vbx,vby,'grey')
# plt.plot(orx,ory,'k')
# plt.plot(vbfr,vbpk,'kx')
# plt.plot(orfr,orpk,'ko')
# figname = "comparison_HW_DE_orig_vs_generato.pdf"
#
# # SE I PICCHI SONO ALLINEATI
# # CALCOLA MSE
# MSE = np.average(np.square(np.array(orpk)-np.array(vbpk)))
# print 'MSE error for ' + figname + ' is ' + str(MSE)
#
# # CALCOLA MSE10 (first 10 harm)
# MSE10 = np.average(np.square(np.array(orpk[:10])-np.array(vbpk[:10])))
# MSE20 = np.average(np.square(np.array(orpk[:20])-np.array(vbpk[:20])))
# print 'MSE10 error for ' + figname + ' is ' + str(MSE10)
# f = open('comparison_HW_DE_orig_vs_generato.txt', 'w')
# f.write('MSE10: ' + "%0.2f" % MSE10 + 'dB - ' + 'MSE20: ' + "%0.2f" % MSE20 + 'dB - ' + 'MSE: ' + "%0.2f" % MSE + 'dB')  # python will convert \n to os.linesep
# f.close()
#
# #plt.title('MSE10: ' + "%0.2f" % MSE10 + 'dB - ' + 'MSE: ' + "%0.2f" % MSE + 'dB', fontsize=20)
# plt.xlabel("Frequency [Hz]", fontsize=24)
# plt.ylabel("Magnitude [dB]", fontsize=24)
# plt.tick_params(axis='both', which='major', labelsize=20)
#
# # SE I PICCHI NON SONO ALLINEATI (CON PEAK DETECT GENERALE)
# # # CALCOLA MSE
# # diff = list(range(0,tot_harm))
# # for i in range(0,tot_harm): # compare the first 30 harm
# #     if (orx[i] - vbx[i]) <= np.round(nbins * f_toler / (Fs)): # tollero errore di f_toler Hz
# #         diff[i] = (ory[i]-vby[i]) # BUUUUUUG!!!! dovrei fare la diff tra i picchi, cosi faccio la diff tra i primi 30 valori della fft!!!!!!
# # MSE = np.average(np.square(diff))
# # print 'MSE error for ' + figname + ' is ' + str(MSE)
# #
# # # CALCOLA MSE10 (first 10 harm)
# # diff = list(range(0,10))
# # for i in range(0,10): # compare the first 30 harm
# #     if (orx[i] - vbx[i]) <= np.round(nbins * f_toler / (Fs)): # tollero errore di f_toler Hz
# #         diff[i] = (ory[i]-vby[i])
# # MSE10 = np.average(np.square(diff))
# # print 'MSE10 error for ' + figname + ' is ' + str(MSE10)
# # plt.title('MSE: ' + str(MSE) + ' - ' + 'MSE10: ' + str(MSE10))
#
# axes = plt.gca()
# axes.set_ylim([-60, 0])
# plt.savefig(figname, format='pdf')
#
# #plt.show()
#
