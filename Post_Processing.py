import os
import numpy as np
import matplotlib.pyplot as plt
import wave
import librosa
import paper_spectr_plot
import re
import soundfile as sf
import matplotlib.gridspec as gridspec
import Distance_Obj_Eval
import scipy.signal as signal
import copy

from IPython import embed


# def harm_extraction(audio_in, f_0, fs, harm_matrix, N_peaks=14, tol_cents=10/100.):
#
#
#     NFFT_peaks = 14
#
#     audio_fft = np.fft.fft(a=audio_in, n=2 ** NFFT_peaks)
#     audio_fft /= (len(audio_fft))
#
#     audio_fft = 20*np.log10(np.abs(audio_fft))
#
#     import detect_peaks
#     xa = np.fft.fftfreq(2 ** NFFT_peaks, d=1 / float(fs))
#
#     peaks=detect_peaks.detect_peaks(audio_fft, mph=-180,  mpd=np.round(f_0/2.), ax=xa)
#
#     audio_fft_peaks = audio_fft[peaks]
#     audio_fft_freq = xa[peaks]
#     audio_fft_peaks = audio_fft_peaks[0:len(audio_fft_peaks)/2]
#     audio_fft_freq = audio_fft_freq[0:len(audio_fft_freq)/2]
#
#     frequency_vector = np.zeros(1000)
#     spectral_ampl = np.zeros(1000)
#
#     start_point = 0
#     for i in xrange(len(audio_fft_peaks)):
#
#         if audio_fft_freq[i]<f_0-tol_cents*f_0:
#             start_point += 1
#
#         else:
#             audio_fft_peaks = audio_fft_peaks[start_point:len(audio_fft_peaks)]
#             audio_fft_freq = audio_fft_freq[start_point:len(audio_fft_freq)]
#             break
#
#     for i in xrange(len(audio_fft_peaks)):
#
#         peak_index = i
#         if (audio_fft_freq[peak_index] >= (i+1)*f_0-tol_cents*(i+1)*f_0) and (audio_fft_freq[peak_index] <= (i+1)*f_0+tol_cents*(i+1)*f_0):
#
#             frequency_vector[i] = audio_fft_freq[peak_index]
#             spectral_ampl[i] = audio_fft_peaks[i]
#
#     frequency_vector = frequency_vector[np.nonzero(frequency_vector)][0:N_peaks]
#     spectral_ampl = spectral_ampl[np.nonzero(spectral_ampl)][0:N_peaks]
#
#     # # print audio_fft_peaks
#     # print frequency_vector
#     # print spectral_ampl
#
#     if len(spectral_ampl) == N_peaks:
#         # harm_current_vector = np.append(-50, spectral_ampl)
#         harm_current_vector = spectral_ampl
#     elif len(spectral_ampl) < N_peaks:
#         # harm_current_vector = np.append(-50, spectral_ampl)
#         harm_current_vector = spectral_ampl
#         harm_current_vector = np.append(harm_current_vector, -180*np.ones(N_peaks-len(spectral_ampl)))
#
#     harm_matrix=np.append(harm_matrix, harm_current_vector)
#     # harm_matrix = np.reshape(harm_matrix,[-1,N_peaks+1])
#     harm_matrix = np.reshape(harm_matrix, [-1, N_peaks])
#
#     # plt.subplot(211)
#     # plt.plot(xa[0:len(xa) / 2], audio_fft[0:len(audio_fft)/2])
#     # plt.xscale('log')
#     # plt.xlim(1, 20000)
#     # plt.grid()
#     # plt.subplot(212)
#     # plt.stem(frequency_vector, spectral_ampl)
#     # plt.xscale('log')
#     # plt.xlim(1, 20000)
#     # plt.grid()
#     # plt.show()
#
#     return harm_matrix, harm_current_vector


# def post_plot(diff, network_test_output, Y_test, output_recap_file,label_length):
#
#     diff_avg_plot = sum(np.abs(diff)) / diff.shape[0]
#     plt.subplot(411)
#     plt.stem(diff_avg_plot)
#     plt.grid(True)
#     plt.title('AVG Error on TEST set (parameter-wise)')
#     plt.subplot(412)
#     plt.stem(np.abs(np.mean(diff, axis=1)), 'black')
#     plt.grid(True)
#     plt.title('AVG Error on TEST set (note-wise)')
#     plt.subplot(413)
#     diff_tot = np.reshape(diff, [-1, 1])
#     plt.plot(diff_tot, '+-r')
#     plt.grid(True)
#     plt.title('Error on TEST set (set-wise)')
#     plt.subplot(414)
#     plt.stem(np.reshape(network_test_output, [-1, 1]), 'b')
#     plt.plot(np.reshape(Y_test, [-1, 1]), 'g+-')
#     plt.grid(True)
#     plt.title('Estimated output VS Test Target output')
#     plt.savefig('Error on TEST set.png')
#     out_file = open(output_recap_file, "w")
#     out_file.write("Mean difference = %s\n" % np.mean(diff))
#     out_file.write("RMS difference = %s\n" % np.sqrt(np.mean(diff ** 2)))
#     out_file.close()
#
#     plt.show()


# def plot_harm_difference(harm_target_matrix, harm_estimated_matrix, difference):
#
#     plt.subplot(311)
#     plt.stem(np.mean(np.abs(difference), axis=0), 'b')
#     plt.grid(True)
#     plt.title('AVG Error peak-wise [dBm]; MEAN = ' + ("%s" % np.mean(np.mean(np.abs(difference),axis=0))))
#     plt.subplot(312)
#
#     plt.stem(np.mean(np.abs(difference), axis=1), 'r')
#     plt.grid(True)
#     plt.title('AVG Error note-wise [dBm]; MEAN = ' + ("%s" % np.mean(np.mean(np.abs(difference),axis=1))))
#     plt.subplot(313)
#
#     harm_estimated_matrix = np.reshape(harm_estimated_matrix, [-1, 1])
#     harm_target_matrix = np.reshape(harm_target_matrix, [-1, 1])
#
#     plt.stem(harm_estimated_matrix, 'b')
#     plt.plot(harm_target_matrix, 'g+-')
#     plt.grid(True)
#     plt.title('Estimated output VS Test Target output [dBm]')
#     plt.show()


def harmonic_distance_calc(target_register_wav_path, estimated_register_wav_path, audio_format='.wav', audio_reshape=1,
                           audio_reshape_size=88200, fs_target=31250, fs_est=31250, PLT_FIG_SPEC=True, FIRST_NOTE=0,
                           LAST_NOTE=73, OFFSET=0, PLOT_SHOW=False):
    fig_ratio = (10, 5)  # width, height
    if PLOT_SHOW:
        # if plt.fignum_exists(2):
        #     plt.close('all')
        plt.figure(figsize=fig_ratio)

    RSD_mat = np.zeros(73)
    LSD_mat = np.zeros(73)

    gs1 = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 4])
    gs1.update(left=0.05, right=0.96, hspace=.9)

    audio_plt_res = 2000
    ylim = [-1, 1]

    file_audio_index = FIRST_NOTE + 1 + OFFSET
    file_audio_index_target = FIRST_NOTE + 1 + OFFSET
    harm_target_matrix = []
    harm_estimated_matrix = []
    midi_note_training_vector = []

    lst = os.listdir(estimated_register_wav_path)
    lst.sort()

    lst_target = os.listdir(target_register_wav_path)
    lst_target.sort()

    r = re.compile(".*wav")
    lst_target = filter(r.match, lst_target)

    r = re.compile(audio_format)
    lst = filter(r.search, lst)

    for filename in lst[FIRST_NOTE + OFFSET:LAST_NOTE + 1 + OFFSET]:
        try:
            print 'Estimated file name: ' + filename
            if filename.endswith(audio_format):
                if audio_format == '.wav':
                    audio_estimated,dsr = librosa.load("%s/%s" % (estimated_register_wav_path, lst[file_audio_index - 1]),sr=31250)
                    # audio_estimated = spf.readframes(-1)
                    # audio_estimated = np.fromstring(audio_estimated, 'Int16')
                    # audio_estimated = audio_estimated / 2 ** 15.
                elif audio_format == '.f32m':
                    audio_estimated, sr = sf.read("%s/%s" % (estimated_register_wav_path, lst[file_audio_index - 1]),
                                                  channels=1, samplerate=31250, format='RAW', subtype='FLOAT')
                    audio_estimated = paper_spectr_plot.normalize(audio_estimated)

                    if np.isnan(audio_estimated).sum():
                        print filename + ' CONTAINS NAN'
                        file_audio_index = file_audio_index + 1
                        file_audio_index_target = file_audio_index_target + 1
                        continue

                if audio_reshape == 1:
                    audio_estimated = audio_estimated[0:audio_reshape_size]

                if len(lst_target) <= file_audio_index - 1:
                    break

                audio_target, dsr = librosa.load("%s/%s" % (target_register_wav_path, lst_target[file_audio_index_target - 1]),sr=fs_target)
                # spf = wave.open("%s/%s" % (target_register_wav_path, lst_target[file_audio_index_target - 1]))
                # audio_target = spf.readframes(-1)
                # audio_target = np.fromstring(audio_target, 'Int16')
                # audio_target = audio_target / 2 ** 15.

                audio_target_original = copy.deepcopy(audio_target)
                if fs_target != fs_est:
                    audio_target = librosa.core.resample(audio_target, fs_target, fs_est)
                if audio_reshape == 1:
                    audio_target = paper_spectr_plot.normalize(audio_target[0:audio_reshape_size])

                p, m = librosa.core.piptrack(y=audio_target, sr=fs_est, S=None, n_fft=64000, hop_length=None, fmin=0.0,fmax=20000.0, threshold=0.1)

                if p.shape[0] > 1 and p.shape[1] > 1 and p.any():
                    i, j = np.nonzero(p)
                    f_0_target = p[i[1], j[1]]
                else:
                    continue

                p, m = librosa.core.piptrack(y=audio_estimated, sr=fs_est, S=None, n_fft=64000, hop_length=None,fmin=0.0, fmax=20000.0, threshold=0.1)

                if p.shape[0] > 1 and p.shape[1] > 1 and p.any():
                    i, j = np.nonzero(p)
                    f_0_estimated = p[i[1], j[1]]
                else:
                    continue

                midi_note_estimated = np.round(librosa.core.hz_to_midi(f_0_estimated))
                midi_note_target = np.round(librosa.core.hz_to_midi(f_0_target))

                print 'Target MIDI Note: ' + str(midi_note_target), 'Estimated MIDI Note: ' + str(midi_note_estimated)

                if (midi_note_estimated != midi_note_target) and FIRST_NOTE!=LAST_NOTE:
                    # if midi_note_target<midi_note_estimated:
                    file_audio_index += 1
                    file_audio_index_target += 1
                    continue

                midi_note_training_vector = np.append(midi_note_training_vector, midi_note_estimated)
                midi_note_training_vector = np.append(midi_note_training_vector, midi_note_target)
                # harm_estimated_matrix, harm_current_vector = harm_extraction(audio_estimated, f_0_estimated, fs_est, harm_estimated_matrix)
                # harm_target_matrix, harm_current_vector = harm_extraction(audio_target, f_0_target, fs_est, harm_target_matrix)
                if PLT_FIG_SPEC:

                    # AUDIO PLOT pt1
                    plt.subplot(gs1[-1, -1])
                    plt.title('Target audio waveform')
                    plt.plot(paper_spectr_plot.normalize(audio_target[0:audio_plt_res]), 'k')
                    plt.grid(True)
                    plt.ylim(ylim)
                    plt.tick_params(axis='both', which='major', labelsize=14)
                    ory, orx = paper_spectr_plot.plot_fft(paper_spectr_plot.normalize(audio_target), fs_est, "all", 0, plot=False)

                    plt.subplot(gs1[:-1, -1])
                    plt.title('Estimated audio waveform')
                    plt.plot(paper_spectr_plot.normalize(audio_estimated[0:audio_plt_res]),'k')  # 1.2* trucco per far comparire il numero giusto di ticks sull'asse 7
                    plt.ylim(ylim)
                    plt.xlabel("Time [samples]", fontsize=14)
                    plt.grid(True)
                    plt.tick_params(axis='both', which='major', labelsize=14)
                    if audio_format == '.f32m':
                        end_format_len = 5
                    else:
                        end_format_len = 4
                    figname = filename[0:len(filename) - end_format_len] + '.png'
                    vby, vbx = paper_spectr_plot.plot_fft(paper_spectr_plot.normalize(audio_estimated), fs_est, "all",                                 0, plot=False)

                    tot_harm = int(np.floor(fs_est / 2 / f_0_estimated))  # max harm to find
                    f_toler = f_0_estimated / 10
                    nbins = len(audio_estimated)
                    delta = np.round(nbins * f_toler / (fs_est))
                    maxprobableindex = [(i * f_0_estimated * nbins / fs_est) for i in range(1, tot_harm + 1)]  # first tot_harm harm
                    # PEAK DETECT EURISTICO NELL'INTORNO DELLE ARMONICHE, SE CONOSCO LA F0
                    orpk = []
                    orfr = []
                    for i in maxprobableindex:
                        orpk.append(np.amax(ory[int(i - delta):int(i + delta)]))
                        orfr.append(orx[np.argmax(ory[int(i - delta):int(i + delta)]) + int(i - delta)])

                    vbpk = []
                    vbfr = []
                    for i in maxprobableindex:
                        vbpk.append(np.amax(vby[int(i - delta):int(i + delta)]))
                        vbfr.append(vbx[np.argmax(vby[int(i - delta):int(i + delta)]) + int(i - delta)])

                    plt.subplot(gs1[:, :-1])
                    plt.title('FFT cfr')
                    plt.plot(vbx, vby, 'grey')
                    plt.plot(orx, ory, 'k', alpha=0.8)
                    plt.plot(vbfr, vbpk, 'kx',label='Estimated')
                    plt.plot(orfr, orpk, 'ko',label='Target')
                    plt.legend()

                    MSE10 = np.average(np.square(np.array(orpk[:10]) - np.array(vbpk[:10])))
                    MSE = np.average(np.square(np.array(orpk) - np.array(vbpk)))

                    plt.xlabel("Frequency [Hz]", fontsize=14)
                    plt.ylabel("Magnitude [dB]", fontsize=14)
                    plt.tick_params(axis='both', which='major', labelsize=14)

                    print 'MSE: ' + str(MSE) + ' - ' + 'MSE10: ' + str(MSE10)
                    axes = plt.gca()
                    axes.set_ylim([-60, 0])
                    plt.grid(True)
                    plt.suptitle('MSE: ' + str(MSE) + ' - ' + 'MSE10: ' + str(MSE10), fontsize=18)

                    if PLOT_SHOW:
                        plt.show()
                    else:
                        plt.savefig(estimated_register_wav_path + '/' + figname, format='png', bbox_inches='tight')
                        plt.clf()

                        # plt.figure(figsize=fig_ratio)
                        plt.title('Estimated waveform VS Target waveform')
                        plt.plot(paper_spectr_plot.normalize(
                            audio_estimated[fs_est:fs_est + int(8 * np.round(fs_est / f_0_estimated))]), 'gray')
                        plt.plot(paper_spectr_plot.normalize(signal.resample(
                            audio_target_original[fs_target:fs_target + int(8 * np.round(fs_target / f_0_estimated))],
                            len(audio_estimated[fs_est:fs_est + int(8 * np.round(fs_est / f_0_estimated))]))), 'k')
                        plt.ylim(ylim)
                        plt.xlabel("Time [samples]", fontsize=14)
                        plt.grid(True)
                        plt.tick_params(axis='both', which='major', labelsize=14)
                        figname = filename[0:len(filename) - end_format_len] + '_W.png'
                        plt.savefig(estimated_register_wav_path + '/' + figname, format='png', bbox_inches='tight')
                        plt.clf()

                spec_target = librosa.stft(audio_target, n_fft=8192, hop_length=1024, win_length=8192)
                spec_est = librosa.stft(audio_estimated, n_fft=8192, hop_length=1024, win_length=8192)

                RSD = Distance_Obj_Eval.RSD(spec_target, spec_est)
                LSD = Distance_Obj_Eval.LSD(np.fft.fft(audio_target, 8192), np.fft.fft(audio_estimated, 8192))

                RSD_mat[file_audio_index - 1] = RSD
                LSD_mat[file_audio_index - 1] = LSD
                # waveform_diff_MAE = np.mean(np.abs(audio_target_original[fs_target:int(fs_target + np.round(8 * fs_target / f_0_target))] - audio_estimated[fs:fs + int(np.round(8 * fs / f_0))]))

                print 'RSD: ' + str(RSD)
                print 'LSD: ' + str(LSD) + '\n'

                file_audio_index += 1
                file_audio_index_target += 1

        except Exception as e:

            print '\n' + str(e) + '\n'
            file_audio_index += 1
            file_audio_index_target += 1

            continue

    return LSD_mat, RSD_mat

