from Tkinter import *
import ttk
import tkMessageBox
import tkFileDialog
import numpy as np
import Post_Processing
import paper_spectr_plot
from PIL import ImageTk, Image
import os
import re
import sounddevice as sd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

class Application(Frame):

    def browseDir(self,destination):
        self.dirname = tkFileDialog.askdirectory()
        if self.dirname:
            destination.delete(0,END)
        destination.insert(0,self.dirname)

    def showfft(self,path):
        FIRST_NOTE = int(self.note.get())
        LAST_NOTE = FIRST_NOTE
        self.LSD_mat_first, self.RSD_mat_first=Post_Processing.harmonic_distance_calc(target_register_wav_path=self.reference_path, audio_reshape=int(self.check_cut.get()), audio_reshape_size=int(self.e_audio_length.get()), estimated_register_wav_path=path, fs_est=self.fs, fs_target=self.fs_target, audio_format=self.e_format.get(), FIRST_NOTE=FIRST_NOTE, LAST_NOTE=LAST_NOTE, OFFSET=int(self.e_offset.get()), PLOT_SHOW=True)

    def showwaveform(self,path):
        if self.e_format.get() == '.f32m':
            try:
                lst_complete = os.listdir(path)
                r = re.compile(".*" + self.audio_selected_file + ".f32m")
                lst = filter(r.search, lst_complete)
                lst.sort()
                audio, sr = sf.read("%s/%s" % (path, lst[0]), channels=1, samplerate=31250, format='RAW',
                                    subtype='FLOAT')
            except:
                self.frameEval.destroy()
                tkMessageBox.showinfo('Error', 'No f32m file')
        else:
            try:
                lst_complete = os.listdir(path)
                r = re.compile(".*" + self.audio_selected_file + ".wav")
                lst = filter(r.search, lst_complete)
                lst.sort()
            except:
                self.frameEval.destroy()
                tkMessageBox.showinfo('Error', 'No WAV file')
        audio = paper_spectr_plot.normalize(audio)
        lst_complete = os.listdir(self.reference_path)
        if int(self.e_offset.get()) != 0:
            r1 = re.compile(".*" + str(int(self.audio_selected_file) + int(self.e_offset.get()) + 1))
        else:
            r1 = re.compile(".*" + str(int(self.audio_selected_file)))
        r2 = re.compile(".*wav")
        lst = filter(r1.search, lst_complete)
        lst = filter(r2.search, lst)
        lst.sort()
        try:
            audio_ref = paper_spectr_plot.normalize(librosa.load(self.reference_path + '/' + lst[0], sr=self.fs)[0])
        except:
            self.frameEval.destroy()
            tkMessageBox.showinfo('Error', 'No WAV file')
        try:
            fig_ratio = (10, 10)
            plt.figure(figsize=fig_ratio)
            plt.subplot(311)
            plt.title('Audio Target Waveform')
            plt.plot(audio_ref,'k')
            plt.xlabel('Samples [n]')
            plt.ylabel('Normalized Amplitude')
            plt.grid(True)
            plt.subplot(312)
            plt.title('Audio Estimated Waveform')
            plt.plot(audio,'gray')
            plt.xlabel('Samples [n]')
            plt.ylabel('Normalized Amplitude')
            plt.grid(True)
            plt.suptitle('Audio Waveforms CFR')

            lst_complete = os.listdir(path)

            r = re.compile(".*" + self.audio_selected_file + ".txt")
            lst = filter(r.search, lst_complete)
            lst.sort()

            PHY_params = np.loadtxt(path + '/' + lst[0])
            plt.subplot(313)
            plt.plot(PHY_params, 'ko')
            plt.plot(PHY_params, 'gray', linewidth=2)
            plt.grid(True)
            plt.xlabel('PHY Params')
            plt.ylabel('Value')
            plt.title('PYH Params Plot')

            plt.tight_layout()
            plt.show()
            plt.clf()
            plt.close()
        except:
            self.frameEval.destroy()
            tkMessageBox.showinfo('Error', 'No WAV file')

    def exitFrame(self):
        result = tkMessageBox.askyesno('Message', 'Do you want to save current configuration?')

        if result:
            self.saveTmp()
        global root
        root.destroy()

    def loadTmp(self):

        f = open('tmp.txt', 'rb')
        self.e_1.delete(0,END)
        self.e_1.insert(0, f.readline()[:-1])
        self.e_2.delete(0,END)
        self.e_2.insert(0,f.readline()[:-1])
        self.e_3.delete(0,END)
        self.e_3.insert(0,f.readline()[:-1])
        self.e_4.delete(0,END)
        self.e_4.insert(0,f.readline()[:-1])
        self.e_5.delete(0,END)
        self.e_5.insert(0,f.readline()[:-1])
        self.e_format.delete(0,END)
        self.e_format.insert(0,f.readline()[:-1])
        self.e_audio_length.delete(0,END)
        self.e_audio_length.insert(0,f.readline()[:-1])
        self.e_offset.delete(0,END)
        self.e_offset.insert(0,f.readline()[:-1])
        self.note.delete(0,END)
        self.note.insert(0,f.readline()[:-1])

        self.check_cut.set(int(f.readline()))
        self.check_full_evaluation.set(int(f.readline()))
        self.check_eval.set(int(f.readline()))
        self.check_cfr.set(int(f.readline()))
        f.close()

        self.naccheck(self.e_3, self.check_cfr, DELETE=False)
        self.naccheck(self.e_audio_length, self.check_cut, DELETE=False)
        self.accheck(self.e_offset, self.check_full_evaluation, DELETE=False)

    def saveTmp(self):
        try:
            f = open('tmp.txt','wb')
            f.write(self.e_1.get()+ "\n")
            f.write(self.e_2.get()+ "\n")
            f.write(self.e_3.get()+ "\n")
            f.write(self.e_4.get()+ "\n")
            f.write(self.e_5.get()+ "\n")
            f.write(self.e_format.get()+ "\n")
            f.write(self.e_audio_length.get()+ "\n")
            f.write(self.e_offset.get()+ "\n")
            f.write(self.note.get()+ "\n")

            f.write(str(self.check_cut.get())+ "\n")
            f.write(str(self.check_full_evaluation.get())+ "\n")
            f.write(str(self.check_eval.get())+ "\n")
            f.write(str(self.check_cfr.get())+ "\n")
            f.close()
            tkMessageBox.showinfo('Info', 'Config SAVED')
        except:
            tkMessageBox.showinfo('Error', 'An error occurred, config NOT SAVED')

    def accheck(self, entry, var,DELETE=True):
        if var.get() == 1:
            if DELETE:
                entry.delete(0,END)
                entry.insert(0,0)
            entry.configure(state='disabled')
        else:
            entry.configure(state='normal')

    def naccheck(self, entry, var,DELETE=True):
        if var.get() == 0:
            if DELETE:
                entry.delete(0,END)
                entry.insert(0,0)
            entry.configure(state='disabled')
        else:
            entry.configure(state='normal')

    def selectNote(self):
        try:
            self.audio_selected_file = self.note.get()
            self.frameEval.destroy()
            if self.check_cfr.get() == 1:
                self.mainEval()
            else:
                self.mainEvalSingle()
        except:
            self.frameEval.destroy()
            tkMessageBox.showinfo('Error', 'Select a Note in range 00-73')

    def removeDraft(self):
        result = tkMessageBox.askyesno('Message', 'Do you really want to remove drafts?')
        try:
            if result:
                a = ['']
                np.savetxt(self.first_candidate_path + '/NOTES.txt', a, fmt="%s")
                np.savetxt(self.second_candidate_path + '/NOTES.txt', a, fmt="%s")
        except:
            tkMessageBox.showinfo('Error', 'Error on Save operation')

    def saveDraft(self):
        result = tkMessageBox.askyesno('Message', 'Do you really want to save drafts?')
        if result:
            try:
                a=['']
                a[0]=(str(self.draftFrame.T1.get("1.0", 'end')))
                np.savetxt(self.first_candidate_path + '/NOTES.txt',a,fmt="%s")
                a[0]=(str(self.draftFrame.T2.get("1.0", 'end')))
                np.savetxt(self.second_candidate_path + '/NOTES.txt',a,fmt="%s")
            except:
                tkMessageBox.showinfo('Error', 'Error on Save operation')

    def draftFrame(self):
        try:
            f = open(self.first_candidate_path + '/NOTES.txt')
            init_text = f.read()
            f.close()
            f = open(self.second_candidate_path + '/NOTES.txt')
            init_text_2=f.read()
            f.close()
            # init_text = np.loadtxt(self.first_candidate_path+'/NOTES.txt',dtype='str',delimiter=' ')
            # init_text_2 = np.loadtxt(self.second_candidate_path +'/NOTES.txt',dtype='str',delimiter=' ')

        except:
            init_text =''
            init_text_2=init_text

        self.draftFrame = Toplevel(self)
        self.draftFrame.geometry("1000x400")
        self.draftFrame.Label_1 = Label(self.draftFrame,text="First Candidate Drafts")
        self.draftFrame.S1 = Scrollbar(self.draftFrame)
        self.draftFrame.T1 = Text(self.draftFrame, height=6, width=60)
        self.draftFrame.S1.pack(side=RIGHT, fill=Y)
        self.draftFrame.T1.pack(side=LEFT, fill=Y)
        self.draftFrame.S1.config(command=self.draftFrame.T1.yview)
        self.draftFrame.T1.config(yscrollcommand=self.draftFrame.S1.set)
        self.draftFrame.T1.insert(END, init_text)
        # self.draftFrame.T1.grid(row=0)

        self.draftFrame.Label_2 = Label(self.draftFrame,text="Second Candidate Drafts")
        self.draftFrame.S2 = Scrollbar(self.draftFrame)
        self.draftFrame.T2 = Text(self.draftFrame, height=6, width=60)
        self.draftFrame.S2.pack(side=LEFT, fill=Y)
        self.draftFrame.T2.pack(side=RIGHT, fill=Y)
        self.draftFrame.S2.config(command=self.draftFrame.T2.yview)
        self.draftFrame.T2.config(yscrollcommand=self.draftFrame.S2.set)
        self.draftFrame.T2.insert(END, init_text_2)
        # self.draftFrame.T2.grid(row=0, column=1)

        self.draftFrame.buttonSave = Button(self.draftFrame, width=40, text="Save", command=self.saveDraft)
        self.draftFrame.buttonSave.pack(side=BOTTOM)
        # self.draftFrame.buttonSave.grid(row=9, column=0)
        self.draftFrame.butDel = Button(self.draftFrame, width=40, text="Delete", command=self.removeDraft)
        self.draftFrame.butDel.pack(side=BOTTOM)
        # self.draftFrame.butDel.grid(row=9, column=1)

    def plot_PHY_par(self,path):
        try:
            fig_ratio = (10, 5)  # width, height
            plt.figure(figsize=fig_ratio)
            lst_complete = os.listdir(path)

            r = re.compile(".*" + self.audio_selected_file + ".txt")
            lst = filter(r.search, lst_complete)
            lst.sort()

            PHY_params = np.loadtxt(path+'/'+lst[0])

            plt.plot(PHY_params,'ko')
            plt.plot(PHY_params,'gray',linewidth=2)
            plt.grid(True)
            plt.xlabel('PHY Params')
            plt.ylabel('Value')
            plt.tight_layout()
            plt.title('PYH Params Plot')

            plt.show()
            plt.close()
        except:
            tkMessageBox.showinfo('Error', 'Error on txt loading')

    def stop(self):

        sd.stop()

    def play_candidate(self,path):
        sd.sleep(100)
        if self.e_format.get() == '.f32m':
            try:
                lst_complete = os.listdir(path)
                r = re.compile(".*" + self.audio_selected_file + ".f32m")
                lst = filter(r.search, lst_complete)
                lst.sort()
                audio, sr = sf.read("%s/%s" % (path, lst[0]), channels=1, samplerate=31250, format='RAW',
                                    subtype='FLOAT')
                audio =self.volume.get()/100.*paper_spectr_plot.normalize(audio)
                sd.play(audio, self.fs_target,blocking=False)
            except:
                self.frameEval.destroy()
                tkMessageBox.showinfo('Error', 'No f32m file')
        else:
            try:
                lst_complete = os.listdir(path)
                r = re.compile(".*" + self.audio_selected_file + ".wav")
                lst = filter(r.search, lst_complete)
                lst.sort()
                sd.play(self.volume.get()/100.*paper_spectr_plot.normalize(librosa.load(path + '/' + lst[0], sr=self.fs)[0]), self.fs,blocking=False)
            except:
                self.frameEval.destroy()
                tkMessageBox.showinfo('Error', 'No WAV file')

    def play_ref(self):
        sd.sleep(100)
        sd.stop()
        # if self.e_format.get() == '.f32m':
        #     try:
        #         lst_complete = os.listdir(self.reference_path)
        #         r = re.compile(".*" + self.audio_selected_file + ".f32m")
        #         lst = filter(r.search, lst_complete)
        #         lst.sort()
        #         audio, sr = sf.read("%s/%s" % (self.reference_path, lst[0]), channels=1,samplerate=31250, format='RAW', subtype='FLOAT')
        #         audio =self.volume.get()/100.*paper_spectr_plot.normalize(audio)
        #         sd.play(audio, self.fs_target)
        #     except:
        #         try:
        #             lst_complete = os.listdir(self.reference_path)
        #             r = re.compile(".*" + str(int(self.audio_selected_file)+int(self.e_offset.get())+1))
        #             lst = filter(r.search, lst_complete)
        #             lst.sort()
        #             sd.play(self.volume.get()/100.*paper_spectr_plot.normalize(librosa.load(self.reference_path + '/' + lst[0], sr=self.fs)[0]), self.fs_target)
        #         except:
        #             try:
        #                 lst_complete = os.listdir(self.reference_path)
        #
        #                 r = re.compile(".*" + self.audio_selected_file + ".wav")
        #                 lst = filter(r.search, lst_complete)
        #                 lst.sort()
        #                 sd.play(self.volume.get() / 100. * paper_spectr_plot.normalize(
        #                     librosa.load(self.reference_path + '/' + lst[0], sr=self.fs)[0]),
        #                         self.fs_target)
        #             except:
        #                 self.frameEval.destroy()
        #                 tkMessageBox.showinfo('Error', 'No WAV file')
        # else:
        lst_complete = os.listdir(self.reference_path)
        if int(self.e_offset.get())!=0:
            r1 = re.compile(".*" + str(int(self.audio_selected_file)+int(self.e_offset.get())+1))
        else:
            r1 = re.compile(".*" + str(int(self.audio_selected_file)))
        r2 = re.compile(".*wav")
        lst = filter(r1.search, lst_complete)
        lst = filter(r2.search, lst)
        lst.sort()
        try:
            sd.play(self.volume.get()/100.*paper_spectr_plot.normalize(librosa.load(self.reference_path + '/' + lst[0], sr=self.fs)[0]), self.fs_target)
        except:
            self.frameEval.destroy()
            tkMessageBox.showinfo('Error', 'No WAV file')

    def mainEvalSingle(self):

        self.frameEval = Toplevel(self)
        self.frameEval.geometry("500x1024")

        sd.default.device = self.box_dev.get()

        row=0
        fft_1 = Button(self.frameEval, width=40, text="FFT", font="Verdana 8 bold",
                       command=lambda: self.showfft(self.first_candidate_path))
        fft_1.grid(row=0, column=0)
        row+=1
        # params_1 = Button(self.frameEval, width=40, text="Show PHY Params 1st", font="Verdana 8 bold",
        #                   command=lambda: self.plot_PHY_par(self.first_candidate_path))
        # params_1.grid(row=row, column=0)
        # row+=1

        try:
            lst_complete = os.listdir(self.first_candidate_path)

            r = re.compile(".*" + self.audio_selected_file + ".png")
            lst = filter(r.search, lst_complete)
            lst.sort()

            r = re.compile(".*" + self.audio_selected_file + "_W.png")
            lst_w = filter(r.search, lst_complete)
            lst_w.sort()

            width = 500
            height = 220

            img_1 = ImageTk.PhotoImage(
                Image.open(self.first_candidate_path + '/' + lst[0]).resize((width, height), Image.ANTIALIAS))
            label_1 = Label(self.frameEval, image=img_1)
            label_1.image = img_1
            label_1.grid(row=row, column=0)
            row+=1
            waveform_1 = Button(self.frameEval, width=40, text="waveform 1st", font="Verdana 8 bold",
                                command=lambda: self.showwaveform(self.first_candidate_path))
            waveform_1.grid(row=row, column=0)
            row+=1

            img_3 = ImageTk.PhotoImage(
                Image.open(self.first_candidate_path + '/' + lst_w[0]).resize((width, height), Image.ANTIALIAS))
            label_3 = Label(self.frameEval, image=img_3)
            label_3.image = img_3
            label_3.grid(row=row, column=0)
            row+=1
        except:
            tkMessageBox.showinfo('Error', 'Error on note loading')
            self.frameEval.destroy()
        try:
            title_RSD_1 = Label(self.frameEval,
                                text="RSD:  " + str(self.RSD_mat_first[int(self.audio_selected_file)])[0:5],
                                font="Verdana 10 bold")
            title_LSD_1 = Label(self.frameEval,
                                text="LSD:  " + str(self.LSD_mat_first[int(self.audio_selected_file)])[0:5],
                                font="Verdana 10 bold")

            title_RSD_1.grid(row=row, column=0)
            row+=1
            title_LSD_1.grid(row=row, column=0)
            row+=1

        except:
            tkMessageBox.showinfo('Error', 'No Metrics To Show')
        self.volumeLabel = Label(self.frameEval, text="Volume", font="Verdana 14 bold")
        self.volumeLabel.grid(row=row, column=0)
        row+=1
        self.volume = Scale(self.frameEval, length=300, from_=0, to=100, orient=HORIZONTAL, tickinterval=100)
        self.volume.set(75)
        self.volume.grid(row=row, column=0)
        row+=1
        play_1 = Button(self.frameEval, width=40, text="PLAY ESTIMATED", font="Verdana 8 bold",
                        command=lambda: self.play_candidate(self.first_candidate_path))
        play_1.grid(row=row, column=0)
        row+=1
        play_3 = Button(self.frameEval, width=40, text="PLAY REFERENCE", font="Verdana 8 bold", command=self.play_ref)
        play_3.grid(row=row, column=0)
        row+=1
        stop = Button(self.frameEval, width=40, text="STOP", font="Verdana 8 bold", command=self.stop)
        stop.grid(row=row, column=0)

    def mainEval(self):

        self.stop()
        self.frameEval = Toplevel(self)
        self.frameEval.geometry("1280x1024")

        sd.default.device = self.box_dev.get()

        lst_complete = os.listdir(self.first_candidate_path)

        r = re.compile(".*" + self.audio_selected_file + ".png")
        lst = filter(r.search, lst_complete)
        lst.sort()

        r = re.compile(".*" + self.audio_selected_file + "_W.png")
        lst_w = filter(r.search, lst_complete)
        lst_w.sort()

        width = 600
        height = 270

        row=0
        #
        # title_1 = Label(self.frameEval, text="Fisrst Candidate  ", font="Verdana 10 bold")
        # title_2 = Label(self.frameEval, text=" Second Candidate ", font="Verdana 10 bold")
        #
        # title_1.grid(row=row, column=0)
        # title_2.grid(row=row, column=1)
        # row+=1

        fft_1 = Button(self.frameEval, width=40, text="FFT 1st", font="Verdana 8 bold", command=lambda: self.showfft(self.first_candidate_path))
        fft_1.grid(row=row, column=0)
        fft_2 = Button(self.frameEval, width=40, text="FFT 2nd", font="Verdana 8 bold", command=lambda: self.showfft(self.second_candidate_path))
        fft_2.grid(row=row, column=1)
        row+=1
        #
        # params_1 = Button(self.frameEval, width=40, text="Show PHY Params 1st", font="Verdana 8 bold", command=lambda: self.plot_PHY_par(self.first_candidate_path))
        # params_1.grid(row=row, column=0)
        # params_2 = Button(self.frameEval, width=40, text="Show PHY Params 2nd", font="Verdana 8 bold", command=lambda: self.plot_PHY_par(self.second_candidate_path))
        # params_2.grid(row=row, column=1)
        # row+=1

        img_1 = ImageTk.PhotoImage(Image.open(self.first_candidate_path + '/' + lst[0]).resize((width, height), Image.ANTIALIAS))
        label_1 = Label(self.frameEval, image=img_1)
        label_1.image = img_1
        label_1.grid(row=row, column=0)

        img_2 = ImageTk.PhotoImage(Image.open(self.second_candidate_path + '/' + lst[0]).resize((width, height), Image.ANTIALIAS))
        label_2 = Label(self.frameEval, image=img_2)
        label_2.image = img_2
        label_2.grid(row=row, column=1)
        row+=1

        waveform_1 = Button(self.frameEval, width=40, text="waveform 1st", font="Verdana 8 bold",
                       command=lambda: self.showwaveform(self.first_candidate_path))
        waveform_1.grid(row=row, column=0)
        waveform_2 = Button(self.frameEval, width=40, text="waveform 2nd", font="Verdana 8 bold",
                       command=lambda: self.showwaveform(self.second_candidate_path))
        waveform_2.grid(row=row, column=1)
        row += 1

        img_3 = ImageTk.PhotoImage(Image.open(self.first_candidate_path + '/' + lst_w[0]).resize((width, height), Image.ANTIALIAS))
        label_3 = Label(self.frameEval, image=img_3)
        label_3.image = img_3
        label_3.grid(row=row, column=0)

        img_4 = ImageTk.PhotoImage(Image.open(self.second_candidate_path + '/' + lst_w[0]).resize((width, height), Image.ANTIALIAS))
        label_4 = Label(self.frameEval, image=img_4)
        label_4.image = img_4
        label_4.grid(row=row, column=1)
        row+=1

        try:
            title_RSD_1 = Label(self.frameEval, text="RSD:  "+str(self.RSD_mat_first[int(self.audio_selected_file)+int(self.e_offset.get())])[0:5],font="Verdana 10 bold")
            title_LSD_1 = Label(self.frameEval, text="LSD:  "+str(self.LSD_mat_first[int(self.audio_selected_file)+int(self.e_offset.get())])[0:5],font="Verdana 10 bold")
            title_RSD_2 = Label(self.frameEval, text="RSD:  "+str(self.RSD_mat_second[int(self.audio_selected_file)+int(self.e_offset.get())])[0:5],font="Verdana 10 bold")
            title_LSD_2 = Label(self.frameEval, text="LSD:  "+str(self.LSD_mat_second[int(self.audio_selected_file)+int(self.e_offset.get())])[0:5],font="Verdana 10 bold")

            title_RSD_1.grid(row=row, column=0)
            title_RSD_2.grid(row=row, column=1)
            row += 1
            title_LSD_1.grid(row=row, column=0)
            title_LSD_2.grid(row=row, column=1)
        except:
            tkMessageBox.showinfo('Error', 'No Metrics To Show')

        row+=1

        self.volumeLabel=Label(self.frameEval,text="Volume",font="Verdana 14 bold")
        self.volumeLabel.grid(row=row,column=0)
        self.volume = Scale(self.frameEval,length=600, from_=0, to=100, orient=HORIZONTAL,tickinterval=100)
        self.volume.set(75)
        self.volume.grid(row=row, column=1)
        row+=1

        play_1 = Button(self.frameEval, width=20, text="PLAY FIRST", font="Verdana 8 bold", command=lambda: self.play_candidate(self.first_candidate_path))
        play_1.grid(row=row, column=0)
        play_2 = Button(self.frameEval, width=20, text="PLAY SECOND", font="Verdana 8 bold", command=lambda: self.play_candidate(self.second_candidate_path))
        play_2.grid(row=row, column=1)
        row+=1

        play_3 = Button(self.frameEval, width=20, text="PLAY REFERENCE", font="Verdana 8 bold", command=self.play_ref)
        play_3.grid(row=row, column=0,columnspan=2)

        # row+=1
        #
        # stop = Button(self.frameEval, width=40, text="STOP", font="Verdana 8 bold", command=self.stop)
        # stop.grid(row=row, column=0, columnspan=2)

    def spectrumEvaluate(self):

        if self.check_full_evaluation.get() == 1:
            FIRST_NOTE = 0
            LAST_NOTE = 73
        else:
            FIRST_NOTE = int(self.note.get())
            LAST_NOTE = FIRST_NOTE

        self.LSD_mat_first, self.RSD_mat_first=Post_Processing.harmonic_distance_calc(target_register_wav_path=self.reference_path, audio_reshape=int(self.check_cut.get()), audio_reshape_size=int(self.e_audio_length.get()), estimated_register_wav_path=self.first_candidate_path, fs_est=self.fs, fs_target=self.fs_target, audio_format=self.e_format.get(), FIRST_NOTE=FIRST_NOTE, LAST_NOTE=LAST_NOTE, OFFSET=int(self.e_offset.get()))
        if self.check_cfr.get()==1:
            self.LSD_mat_second, self.RSD_mat_second=Post_Processing.harmonic_distance_calc(target_register_wav_path=self.reference_path, estimated_register_wav_path=self.second_candidate_path, audio_reshape=int(self.check_cut.get()), audio_reshape_size=int(self.e_audio_length.get()), fs_est=self.fs, fs_target=self.fs_target, audio_format=self.e_format.get(), FIRST_NOTE=FIRST_NOTE, LAST_NOTE=LAST_NOTE, OFFSET=int(self.e_offset.get()))

    def loadPathsFunction(self):
        try:
            self.frameEval.destroy()
        except:
            tkMessageBox.showinfo('Info', 'First Metrics Evaluation ')
        try:
            self.reference_path = self.e_1.get()
            self.first_candidate_path = self.e_2.get()
            self.second_candidate_path = self.e_3.get()
            self.fs = int(self.e_4.get())
            self.fs_target = int(self.e_5.get())
            result = tkMessageBox.askyesno('Message', 'Do you really want to proceed?')
            if result:
                if self.check_eval.get()==1:
                    self.spectrumEvaluate()
                    self.audio_selected_file = self.note.get()
                    if self.check_cfr.get()==1:
                        self.mainEval()
                    else:
                        self.mainEvalSingle()
                    self.note_sel.config(state=NORMAL)
                    self.mainloop()
                    # self.check_eval.set(0)

                else:
                    self.audio_selected_file = self.note.get()
                    if self.check_cfr.get() == 1:
                        self.mainEval()
                    else:
                        self.mainEvalSingle()
                        self.note_sel.config(state=NORMAL)
        except:
            tkMessageBox.showinfo('Error', 'Compile Fields Correctly')

    def loadPaths(self):

        self.label_1 = Label(self,text='Reference Path',relief=RIDGE,width=25 )
        self.label_1.grid(row=0)
        self.e_1 = Entry(self, width=100)
        self.e_1.insert(END, '/home/stefano/Projects/CNN/Flue_Pipe_Framework/Audio_Test/Testing_principale/Testing_8_Principale_VS[FT2]_[WE80]_[AL1]_[ID154]')
        self.e_1.grid(row=0,column=1,columnspan=4)
        self.label_2 = Label(self,text='First Candidate Path',relief=RIDGE,width=25 )
        self.label_2.grid(row=1)
        self.e_2 = Entry(self, width=100)
        self.e_2.grid(row=1, column=1,columnspan=4)
        self.label_3 = Label(self,text='Second Candidate Path',relief=RIDGE,width=25)
        self.label_3.grid(row=2)
        self.e_3 = Entry(self, width=100)
        self.e_3.grid(row=2, column=1,columnspan=4)
        self.label_4 = Label(self,text='fs',relief=RIDGE,width=25)
        self.label_4.grid(row=3)
        self.e_4 = Entry(self, width=100)
        self.e_4.grid(row=3, column=1,columnspan=4)
        self.e_4.insert(END,'31250')
        self.e_4.configure(state='disabled')
        self.label_5 = Label(self,text='fs target',relief=RIDGE,width=25)
        self.label_5.grid(row=4)
        self.e_5 = Entry(self, width=100)
        self.e_5.grid(row=4, column=1,columnspan=4)
        self.e_5.insert(END,'31250')
        self.label_format = Label(self, text='audio format', relief=RIDGE, width=25)
        self.label_format.grid(row=5)
        self.e_format = Entry(self, width=100)
        self.e_format.grid(row=5, column=1,columnspan=4)
        self.e_format.insert(END, '.wav')

        self.evaluate = Button(self,width=40, text="Evaluate",command=self.loadPathsFunction)
        self.evaluate.grid(row=6)

        self.check_eval=IntVar()
        Checkbutton(self, text="Compute Spectral Distance",variable=self.check_eval).grid(row=6, column=1,sticky=W)

        self.label_offset = Label(self, text='Note Offset', relief=RIDGE, width=25)
        self.label_offset.grid(row=6,column=2)
        self.e_offset = Entry(self, width=10)
        self.e_offset.grid(row=6, column=3, columnspan=4)
        self.e_offset.insert(END, '0')

        self.check_cfr = IntVar()
        self.check_cfr.set(1)
        Checkbutton(self,  text="CFR First VS Second", font="Verdana 8 bold", variable=self.check_cfr, command=lambda e=self.e_3, v=self.check_cfr: self.naccheck(e, v, DELETE=False)).grid(row=7, column=0, sticky=W)

        self.check_full_evaluation = IntVar()
        Checkbutton(self, text="Full Stop Distance Eval", variable=self.check_full_evaluation,command=lambda e=self.e_offset, v=self.check_full_evaluation: self.accheck(e,v)).grid(row=7, column=1, sticky=W)

        self.e_audio_length = Entry(self, width=20)
        self.e_audio_length.grid(row=7, column=3, columnspan=3)
        self.e_audio_length.insert(END, '0')
        self.check_cut = IntVar(value=0)
        Checkbutton(self, text="Audio length cut", variable=self.check_cut, command=lambda e=self.e_audio_length, v=self.check_cut: self.naccheck(e, v)).grid(row=7, column=2, sticky=W)

        self.device_label = Label(self, text='Audio Device',font="Verdana 8 bold")
        self.device_label.grid(row=8, column=2, columnspan=1)

        self.box_dev = ttk.Combobox(self)
        app = sd.query_devices()
        val=[]
        for i in range(len(app)):
            val.append(app[i]['name'])
        self.box_dev['values'] = val
        self.box_dev.set('pulse')

        self.box_dev.grid(row=8,column=3)

        self.note = Entry(self, width=10)
        self.note.insert(END, '00')
        self.note.grid(row=8, column=1,columnspan=1)

        self.note_sel = Button(self, width=40, text="Select Note", command=self.selectNote,state=DISABLED)
        self.note_sel.grid(row=8, column=0)

        self.open_draft = Button(self, width=40, text="Open Draft", command=self.draftFrame)
        self.open_draft.grid(row=9, column=0)

        self.save_button = Button(self, width=10, text="Save", command=self.saveTmp)
        self.save_button.grid(row=9, column=3)

        self.exit_button = Button(self, width=10, text="Exit", command=self.exitFrame)
        self.exit_button.grid(row=9, column=4)

        self.ld1 = Button(self, width=10, text="Search", command=lambda: self.browseDir(destination=self.e_1))
        self.ld1.grid(row=0, column=5)

        self.ld2 = Button(self, width=10, text="Search", command=lambda: self.browseDir(destination=self.e_2))
        self.ld2.grid(row=1, column=5)

        self.ld3 = Button(self, width=10, text="Search", command=lambda: self.browseDir(destination=self.e_3))
        self.ld3.grid(row=2, column=5)

        result = tkMessageBox.askyesno('Message', 'Do you want to LOAD last configuration?')
        if result:
            self.loadTmp()

        self.mainloop()

    def __init__(self, master=None):
        Frame.__init__(self, master)
        root.protocol('WM_DELETE_WINDOW', self.exitFrame)
        self.pack()
        self.loadPaths()

root = Tk()
root.wm_title("Pipe GUI Post Processing")
Application().mainloop()
