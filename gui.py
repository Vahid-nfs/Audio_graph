from PySide2.QtMultimedia import QAudioDeviceInfo, QAudio
from PyQt5.QtCore import pyqtSlot
from PyQt5 import uic
from PyQt5 import QtCore, QtWidgets
import sounddevice as sd
import numpy as np
import queue
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import os
import matplotlib
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget , QFileDialog ,QMessageBox
import soundfile as sf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg ,NavigationToolbar2QT as Navigationtoolbar
from scipy.io import wavfile
import multiprocessing
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")


# uses QAudio to obtain all the available devices on the system
input_audio_deviceInfos = QAudioDeviceInfo.availableDevices(QAudio.AudioInput)
output_audio_deviceInfos = QAudioDeviceInfo.availableDevices(QAudio.AudioOutput)

# class with all the specification for plotting the matplotlib figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3.5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        # fig.patch.set_facecolor("#00B28C")
        self.axes = fig.add_subplot(111)

        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()
        


# The main window that is called to run the application


class Plot_win(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi("plot_win.ui", self)
        self.setWindowTitle('Magnitude plot')
        self.resize(600,400)
        

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        # import the QT designer created ui for the application
        self.ui = uic.loadUi("GUI_plotter.ui", self)
        self.resize(900, 795)  # reset the size
        self.ui.setWindowTitle('Voice Recorder')
        self.record_btn.setIcon(QIcon(os.path.join('icons', 'record.png')))
        self.stop_btn.setIcon(QIcon(os.path.join('icons', 'stop.png')))
        self.play_btn.setIcon(QIcon(os.path.join('icons', 'play.png')))
        self.ui.setWindowIcon(QIcon(os.path.join('icons', 'sound.png')))

        self.threadpool = QtCore.QThreadPool()
        self.threadpool.setMaxThreadCount(1)
        self.input_devices_list = []
        self.output_devices_list = []
        for device in input_audio_deviceInfos:
            self.input_devices_list.append(device.deviceName())
            
        for device in output_audio_deviceInfos:
            self.output_devices_list.append(device.deviceName())

        # add all the available device name to the combo box
        self.comboBox_in.addItems(self.input_devices_list)
        # when the combobox selection changes run the function update_now
        self.comboBox_in.currentIndexChanged["QString"].connect(self.update_now_in)
        self.comboBox_in.currentTextChanged["QString"].connect(self.update_in)
        self.comboBox_dtype.currentTextChanged["QString"].connect(self.update_dtype)
        self.comboBox_in.setCurrentIndex(0)

        self.comboBox_out.addItems(self.output_devices_list)
        
        # when the combobox selection changes run the function update_now
        self.comboBox_out.currentTextChanged["QString"].connect(self.update_out)
        self.comboBox_out.setCurrentIndex(0)
        
        
        self.canvas = MplCanvas(self, width=5, height=2, dpi=100)
        self.canvas.axes.set_facecolor("#D5F9FF")
        self.toolbar=Navigationtoolbar(self.canvas,self)
        self.ui.gridLayout_1.addWidget(self.toolbar)
        self.ui.gridLayout_2.addWidget(self.canvas, 2, 1, 1, 1)
        
        
        
        
        self.reference_plot = None
        self.q = queue.Queue()

        # plot specifications
        self.dtype="int 8"
        self.channels = 1
        self.rec=np.empty([1, self.channels])
        self.recordingTime=300
        self.window_length = 340*self.recordingTime # for obtaining sound
        self.downsample = 1  # for obtaining sound
        self.interval = 30  # update plot every 30/1000 second
        self.yrangeMinVal = -0.5
        self.yrangeMaxVal = 0.5
        
        self.sound=None
        self.selected_sound=None
        self.play=True
        # self.all_devices = list(sd.query_devices())
        # print(len(self.all_devices))
        self.device_success_in = 0
        for self.in_device in range(len(input_audio_deviceInfos)):
            try:
                in_device_info = sd.query_devices(self.in_device, "input")
                if in_device_info:
                    self.device_success_in = 1
                    break
            except:
                pass
            
        self.device_success_out = 0
        for self.out_device in range(len(input_audio_deviceInfos),len(input_audio_deviceInfos)+len(output_audio_deviceInfos)):
            try:
                out_device_info = sd.query_devices(self.out_device, "output")
                if out_device_info:
                    self.device_success_out = 1
                    break
            except:
                pass
            
        if self.device_success_in:  # run if the device connection is successful
            # print(device_info)
            self.samplerate = in_device_info["default_samplerate"]
            length = int(self.window_length * self.samplerate /
                         (1000 * self.downsample))
            sd.default.samplerate = self.samplerate
            self.plotdata = np.zeros((length,1))
        else:
            self.disable_buttons()
            self.stop_btn.setEnabled(False)
            self.stop_btn.setStyleSheet(
                "QPushButton" "{" "background-color : lightblue;" "}"
            )
            
            self.input_devices_list.append("No Devices Found")
            self.comboBox_in.addItems(self.input_devices_list)
            
        if self.device_success_out:
            pass
        else:
            self.play_btn.setEnabled(False)
            self.output_devices_list.append("No Devices Found")
            self.comboBox_out.addItems(self.output_devices_list)
            
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.interval)  # msec
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        self.data = [0]
        # self.winLen_lineEdit.textChanged["QString"].connect(self.update_window_length)
        self.sampleRate_lineEdit.textChanged.connect(self.update_sample_rate)
        self.spinBox_recordingTime.valueChanged.connect(self.update_recordingTime)
        self.spinBox_num_channels.valueChanged.connect(self.update_channels)        
        # self.spinBox_downsample.valueChanged.connect(self.update_down_sample)
        # self.spinBox_updateInterval.valueChanged.connect(self.update_interval)

        self.doubleSpinBox_yrangemin.valueChanged.connect(
            self.update_yrange_min)  # change the yminvalues when the Yrange is changed
        self.doubleSpinBox_yrangemax.valueChanged.connect(
            self.update_yrange_max)


        self.stop_btn.clicked.connect(self.stop_worker)
        self.actionOpen.triggered.connect(self.Open)
        self.actionSave.triggered.connect(self.Save)
        self.actionabout.triggered.connect(self.about)
        
        self.play_btn.clicked.connect(self.play_pause_action)
        self.plot_btn.clicked.connect(self.plot_btn_action)
        self.record_btn.clicked.connect(self.start_worker)
        self.worker = None
        self.go_on = False

    def getAudio(self):
        try:
            QtWidgets.QApplication.processEvents()

            def audio_callback(indata, frames, time, status):
                self.q.put(indata.copy())
                
            # uses sounddevice to obtain the input stream, check the InputStream for details
            stream = sd.InputStream(
                device=self.in_device,
                channels=self.channels,
                samplerate=self.samplerate,
                callback=audio_callback,
                dtype=self.dtype.replace(" ","")
            )
            with stream:

                while True:
                    QtWidgets.QApplication.processEvents()
                    
                    if self.go_on:
                        break
            self.enable_buttons()

        except Exception as e:
            print("ERROR: ", e)
            self.stop_worker
            pass
    
    def update_dtype(self,value):
        self.dtype=self.comboBox_dtype.currentText()
        
        
    def update_in(self,value):
        sd.default.device=[value,self.comboBox_out.currentText()]
        
    def update_out(self,value):
        sd.default.device=[self.comboBox_in.currentText(),value]
    
    def play_pause_action(self):
        sd.default.device=[self.comboBox_in.currentText(),self.comboBox_out.currentText()]
        if self.play:
            self.play=False
            self.play_btn.setText("pause")
            self.play_btn.setIcon(QIcon(os.path.join('icons', 'pause.png')))
            self.crop_sound()
            sd.play(self.selected_sound)
            
        else:
            self.play=True
            self.play_btn.setText("Play")
            self.play_btn.setIcon(QIcon(os.path.join('icons', 'play.png')))
            sd.stop()
        
    def crop_sound(self):
        start=int(mainWindow.doubleSpinBox_xlimStart.value()*self.Fs)
        end=int(mainWindow.doubleSpinBox_xlimEnd.value()*self.Fs)
        self.selected_sound=self.sound[start:end]
        
    
        
    def plot_btn_action(self):
        mag_plot = Plot_win()
        self.resize(1400, 795)
        self.crop_sound()
        start=mainWindow.doubleSpinBox_xlimStart.value()
        end=mainWindow.doubleSpinBox_xlimEnd.value()
        sp=self.spectogram_plot(self._plot_sound,self.Fs,t_min=start,t_max=end,mode="magnitude")
        pre_emp=self.pre_emp_plot(self._plot_sound,self.Fs,start)
        atp=self.Amplitude_time_plot(self._plot_sound,self.Fs,t_min=start,t_max=end)
        self.ui.gridLayout_1.removeWidget(self.toolbar)
        self.toolbar=Navigationtoolbar(atp,self)
        self.ui.gridLayout_1.addWidget(self.toolbar)
        Fp=self.Fourier_plot(self._plot_sound,self.Fs,start)
        Qp=self.quefrency_plot()
        Pp=self.pitch_plot()
        AWp=self.Amplitude_windowing_plot(self._plot_sound,self.Fs,time=start)
        MFWp=self.magnitude_freq_W_plot(self._plot_sound,self.Fs,time=start)
        Mp,_,_=self.magnitude_plot(self._plot_sound,self.Fs,time=start)
        mag_plot.toolbar=Navigationtoolbar(Mp,mag_plot)
        mag_plot.ui.gridLayout_2.addWidget(mag_plot.toolbar)
        
        

        self.ui.gridLayout_Spectrogram.addWidget(sp, 2, 1, 1, 1)
        self.ui.gridLayout_pre_emp.addWidget(pre_emp, 2, 1, 1, 1)
        self.ui.gridLayout_2.addWidget(atp,2, 1, 1, 1)
        self.ui.gridLayout_3.addWidget(Fp, 2, 1, 1, 1)
        self.ui.gridLayout_4.addWidget(Qp, 2, 1, 1, 1)
        self.ui.gridLayout_5.addWidget(Pp, 2, 1, 1, 1)
        self.ui.gridLayout_6.addWidget(AWp, 2, 1, 1, 1)
        self.ui.gridLayout_7.addWidget(MFWp, 2, 1, 1, 1)
        mag_plot.ui.gridLayout.addWidget(Mp, 2, 1, 1, 1)
        mag_plot.show()

        

    def disable_buttons(self):
        # self.winLen_lineEdit.setEnabled(False)
        self.sampleRate_lineEdit.setEnabled(False)
        # self.spinBox_downsample.setEnabled(False)
        # self.spinBox_updateInterval.setEnabled(False)
        self.spinBox_recordingTime.setEnabled(False)
        self.comboBox_in.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.record_btn.setStyleSheet(
            "QPushButton" "{" "background-color : lightblue;" "}"
        )

        self.canvas.axes.clear()

    def enable_buttons(self):
        self.record_btn.setEnabled(True)
        # self.winLen_lineEdit.setEnabled(True)
        self.sampleRate_lineEdit.setEnabled(True)
        # self.spinBox_downsample.setEnabled(True)
        # self.spinBox_updateInterval.setEnabled(True)
        self.spinBox_recordingTime.setEnabled(True)
        self.comboBox_in.setEnabled(True)

    def start_worker(self):
        self.rec=np.empty([1,self.channels])
        self.ui.gridLayout_1.removeWidget(self.toolbar)
        self.ui.gridLayout_2.addWidget(self.canvas, 2, 1, 1, 1)
        self.toolbar=Navigationtoolbar(self.canvas,self)
        self.ui.gridLayout_1.addWidget(self.toolbar)
        
        self.disable_buttons()

        self.canvas.axes.clear()
        self.go_on = False
        self.worker = Worker(
            self.start_stream,
        )
        self.threadpool.start(self.worker)
        self.reference_plot = None
        self.timer.setInterval(self.interval)  # msec


    def stop_worker(self):
        self.sound=np.squeeze(self.rec[1:])
        if len(self.sound.shape)>1:
            self._plot_sound=self.sound.mean(axis=1)
        else:
            self._plot_sound=self.sound
        if self.dtype.rsplit(" ")[0]=="int":
            self._plot_sound=self._plot_sound/2**int(self.dtype.rsplit(" ")[1])
        self.Fs=int(self.sampleRate_lineEdit.text())
        time=round((len(self.sound)/self.Fs),2)
        self.doubleSpinBox_xlimStart.setMaximum(time)
        self.doubleSpinBox_xlimEnd.setMaximum(time)
        self.doubleSpinBox_xlimEnd.setValue(time)
        
        
        self.go_on = True
        # with self.q.mutex:
            # self.q.queue.clear()
        self.record_btn.setStyleSheet(
            "QPushButton"
            "{"
            "background-color : rgb(92, 186, 102);"
            "}"
            "QPushButton"
            "{"
            "color : white;"
            "}"
        )
        self.enable_buttons()

    def start_stream(self):
        self.getAudio()

    def update_now_in(self, value):
        self.in_device = self.input_devices_list.index(value)
        

    def update_window_length(self, value):
        self.window_length = int(value)
        length = int(self.window_length * self.samplerate /
                     (1000 * self.downsample))
        self.plotdata = np.zeros((length, 1))

    def update_sample_rate(self, value):
        try:
            self.samplerate = int(value)
            sd.default.samplerate = self.samplerate
            length = int(
                self.window_length * self.samplerate / (1000 * self.downsample)
            )
            print(self.samplerate, sd.default.samplerate)
            self.plotdata = np.zeros((length, 1))
        except:
            pass

    def update_channels(self,value):
        self.channels=value
        self.rec=np.empty([1, self.channels])

    # def update_down_sample(self, value):
    #     self.downsample = int(value)
    #     length = int(self.window_length * self.samplerate /
    #                  (1000 * self.downsample))
    #     self.plotdata = np.zeros((length, channels))

    # def update_interval(self, value):
    #     self.interval = int(value)
    
    def update_recordingTime(self, value):
        self.recordingTime = int(value)
        self.window_length = 340*self.recordingTime

    def update_yrange_min(self, minval):
        self.yrangeMinVal = float(minval)

    def update_yrange_max(self, maxval):
        self.yrangeMaxVal = float(maxval)
    
    def about (self):
        msgbox=QMessageBox()
        msgbox.setText("insert your text")
        msgbox.exec()
    
    def Open(self):
        fname=Load()
        try:
    
            self.Fs, self.sound = wavfile.read(fname.name)
            if len(self.sound.shape)>1:
                self._plot_sound=self.sound.mean(axis=1)
            else:
                self._plot_sound=self.sound
            time=round((len(self.sound)/self.Fs),2)
            self.doubleSpinBox_xlimStart.setMaximum(time)
            self.doubleSpinBox_xlimEnd.setMaximum(time)
            self.doubleSpinBox_xlimEnd.setValue(time)
            msgbox=QMessageBox()
            msgbox.setText("File opened")
            msgbox.exec()
        except Exception as e:
            msgbox=QMessageBox()
            msgbox.setText(f"{e.args[-1]}")
            msgbox.exec()
        
    def Save(self):
        try:
            fname=Write()
            self.fname=fname
            wavfile.write(fname.name,self.Fs,self.sound)
            msgbox=QMessageBox()
            msgbox.setText("File saved")
            msgbox.exec()
        except Exception as e:
            msgbox=QMessageBox()
            msgbox.setText(f"{e.args[-1]}")
            msgbox.exec()
    
    
    
    def Fourier_plot(self,signal_data,Fs,t_min):
        if signal_data.max()>2:
            x=signal_data/2**16
        else:
            x=signal_data
        sc = MplCanvas(self, width=5, height=1.7, dpi=100)
        frame_size=int(.025*Fs)
        signal=x[int(t_min*Fs):int((t_min+.025)*Fs)]
        windowed_signal = np.hanning(frame_size) * signal
        dt = 1/Fs
        self.freq_vector = np.fft.rfftfreq(frame_size, d=dt)
        X = np.fft.rfft(windowed_signal)
        self.log_X = np.log(np.abs(X))
        sc.axes.plot(self.freq_vector, self.log_X)
        sc.axes.set_xlabel('frequency (Hz)',fontsize=5)
        sc.axes.set_title('Fourier spectrum',fontsize=5)
        sc.axes.tick_params(labelsize=5)
        return(sc)

        
    def quefrency_plot(self):
        sc = MplCanvas(self, width=5, height=1.7, dpi=100)

        self.cepstrum = np.fft.rfft(self.log_X)
        df = self.freq_vector[1] - self.freq_vector[0]
        self.quefrency_vector = np.fft.rfftfreq(self.log_X.size, df)
        
        sc.axes.plot(self.quefrency_vector, np.abs(self.cepstrum))
        sc.axes.set_xlabel('quefrency (s)',fontsize=5)
        sc.axes.set_title('cepstrum',fontsize=5)
        sc.axes.tick_params(labelsize=5)
        return(sc)
    
    def pitch_plot(self):
        import matplotlib.collections as collections
        from scipy.signal import find_peaks
        import pandas as pd
        
        sc = MplCanvas(self, width=5, height=1.7, dpi=100)
        sc.axes.vlines(1/440, 0, np.max(np.abs(self.cepstrum)), alpha=.2, lw=3, label='expected peak')
        sc.axes.plot(self.quefrency_vector, np.abs(self.cepstrum))
        valid = (self.quefrency_vector > 1/640) & (self.quefrency_vector <= 1/82)
        collection = collections.BrokenBarHCollection.span_where(
            self.quefrency_vector, ymin=0, ymax=np.abs(self.cepstrum).max(), where=valid, facecolor='green', alpha=0.5, label='valid pitches')
        sc.axes.add_collection(collection)
        sc.axes.set_xlabel('quefrency (s)',fontsize=5)
        sc.axes.set_ylabel('cepstrum',fontsize=5)
        sc.axes.set_title('pitch plot',fontsize=5)
        sc.axes.set_xlim([.0003,None])
        sc.axes.set_ylim(0,100)
        sc.axes.legend(fontsize=5)
        sc.axes.tick_params(labelsize=5)


        peaks, _ = find_peaks(np.abs(self.cepstrum), distance=6)  
        valid_cep=pd.Series(np.abs(self.cepstrum))[valid]
        valid_ind=list(set(valid_cep.index) & set(peaks))
        sc.axes.scatter(self.quefrency_vector[valid_ind],np.abs(self.cepstrum)[valid_ind],c="red",s=15)
        return(sc)
        

    
    def Amplitude_time_plot(self,snd_data,Fs,t_min=None,t_max=None):
        N = snd_data.shape[0]
        sc = MplCanvas(self, width=5, height=2.1, dpi=100)
        sc.axes.plot(np.arange(N) / Fs, snd_data)
        sc.axes.set_xlabel('Time [s]',fontsize=5)
        sc.axes.set_ylabel('Amplitude',fontsize=5)
        sc.axes.set_title('Amplitude plot',fontsize=5)
        sc.axes.tick_params(labelsize=5)
        sc.axes.set_xlim([t_min,t_max])
        return(sc)
    

    def Amplitude_windowing_plot(self,snd_data,Fs,time=None):
        star_frame=int(time*Fs)
        end_frame=int((time+.025)*Fs)
        sn=snd_data[star_frame:end_frame]
        frame_size=len(sn)
        N = sn.shape[0]
        sc = MplCanvas(self, width=5, height=3, dpi=100)
        sc.axes.plot(np.arange(N) / Fs,  np.hanning(frame_size) * sn,c="red",linewidth=1)
        sc.axes.plot(np.arange(N) / Fs,  np.hamming(frame_size) * sn,c="blue",linewidth=1)
        sc.axes.set_xlabel('Time [s]',fontsize=5)
        sc.axes.set_ylabel('Amplitude',fontsize=5)
        sc.axes.tick_params(labelsize=5)
        sc.axes.set_title('magnitude windowing plot',fontsize=5)
        sc.axes.legend(["hanning window" , "hamming window"],fontsize=5)
        sc.axes.tick_params(labelsize=5)
        return(sc)

    
    
    def pre_emp_plot(self,snd_data,Fs,time,freq=10000):
        sc = MplCanvas(self, width=5, height=3, dpi=100)
        star_frame=int(time*Fs)
        end_frame=int((time+.025)*Fs)
        sn=snd_data[star_frame:end_frame]
        sn_1=np.roll(sn,1)
        sn_1[0]=0
        pre_emp=sn-.96*sn_1
        
        spectrum,freqs,line=sc.axes.magnitude_spectrum(sn,freq,scale="dB",linewidth=1,c="blue")
        spectrum,freqs,line=sc.axes.magnitude_spectrum(pre_emp,freq,scale="dB",linewidth=1,c="red")
        # _,magnitude=line.get_data()
        sc.axes.tick_params(labelsize=5)
        sc.axes.legend(["Raw DFT Coeffiecents","pre_emphasized DFT Coeffiecents"],fontsize=5)
        sc.axes.set_title('pre_emphasized plot',fontsize=5)
        return(sc)
    
    
    def magnitude_plot(self,snd_data,Fs,time,freq=10000):
        sc = MplCanvas(self, width=5, height=3, dpi=100)
        star_frame=int(time*Fs)
        end_frame=int((time+.025)*Fs)
        sn=snd_data[star_frame:end_frame]
        spectrum,freqs,line=sc.axes.magnitude_spectrum(sn,freq,scale="dB",visible=True)
        sc.axes.set_title('magnitude plot',fontsize=5)
        sc.axes.tick_params(labelsize=5)
        sc.axes.set_xlabel('Frequency ',fontsize=5)
        sc.axes.set_ylabel('Magnitude (dB)',fontsize=5)
        return(sc,freqs,spectrum)
    
    
    
    
    
    def magnitude_freq_W_plot(self,snd_data,Fs,time,freq=10000):
        sc,freqs,spectrum=self.magnitude_plot(snd_data,Fs,time,freq=10000)
        # plt.set_title("Energy plot")
        frame_size=len(freqs)
        sc.axes.plot(freqs,np.hanning(frame_size) * (20 * np.log10(spectrum)),linewidth=1,c="blue")
        sc.axes.plot(freqs,np.hamming(frame_size) * (20 * np.log10(spectrum)),linewidth=1,c="red")
        sc.axes.set_xlabel('Frequency ',fontsize=5)
        sc.axes.set_ylabel('Magnitude',fontsize=5)
        sc.axes.tick_params(labelsize=5)
        sc.axes.set_ylim(None,60)
        sc.axes.legend(["Magniyude raw","hanning window" , "hamming window"],fontsize=5)
        sc.axes.set_title('magnitude windowing plot',fontsize=5)
        return(sc)  
    
    
    
    
    def spectogram_plot(self,snd_data,samplingFrequency,t_min=None,t_max=None,mode="magnitude"):
        sc = MplCanvas(self, width=5, height=3, dpi=100)
        sc.axes.specgram(snd_data,Fs=samplingFrequency,cmap='gray_r',scale='dB',mode=mode)
        sc.axes.set_xlabel('Time',fontsize=5)
        sc.axes.set_ylabel('Frequency',fontsize=5)
        sc.axes.set_xlim([t_min,t_max]) 
        sc.axes.tick_params(labelsize=5)
        sc.axes.set_title('Spectogram plot',fontsize=5)
        return(sc)
    
        
    

    def update_plot(self):
        try:

            # print("ACTIVE THREADS:", self.threadpool.activeThreadCount(), end=" \r")
            # self.label_18.setText(f"{self.threadpool.activeThreadCount()}")
            while self.go_on is False:
                QtWidgets.QApplication.processEvents()
                try:
                    self.data = self.q.get_nowait()
                    self.rec=np.vstack((self.rec,self.data))
                except queue.Empty:
                    break

                shift = len(self.data)
                self.plotdata = np.roll(self.plotdata, -shift, axis=0)
                if self.dtype.rsplit(" ")[0] == "int" :
                    power=int(self.dtype.rsplit(" ")[1])
                    self.plotdata[-shift:, :] = self.data[:,0:1]/2**power
                else:
                    self.plotdata[-shift:, :] = self.data[:,0:1]
                self.ydata = self.plotdata[:]
                self.canvas.axes.set_facecolor("#D5F9FF")

                if self.reference_plot is None:
                    plot_refs = self.canvas.axes.plot(
                        self.ydata, color="green")
                    self.reference_plot = plot_refs[0]
                else:
                    self.reference_plot.set_ydata(self.ydata)

            self.canvas.axes.yaxis.grid(True, linestyle="--")
            start, end = self.canvas.axes.get_ylim()
            self.canvas.axes.yaxis.set_ticks(np.arange(start, end, 0.1))
            self.canvas.axes.yaxis.set_major_formatter(
                ticker.FormatStrFormatter("%0.1f")
            )
            self.canvas.axes.set_ylim(
                ymin=self.yrangeMinVal, ymax=self.yrangeMaxVal)

            self.canvas.draw()
        except Exception as e:
            print("Error:", e)
            pass


class Worker(QtCore.QRunnable):
    def __init__(self, function, *args, **kwargs):
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        self.function(*self.args, **self.kwargs)
        



class Load(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs' 
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.name=""
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.openFileNameDialog() 

    
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, a = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;wav file (*.wav)", options=options)
        self.name=fileName
        
        
class Write(QWidget):
    def __init__(self,Type="open"):
        super().__init__()
        self.title = 'PyQt5 file dialogs' 
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.name=""
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.saveFileDialog()
            
    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;wav Files (*.wav)", options=options)
        self.name=fileName





app = QtWidgets.QApplication(sys.argv)

if __name__ == "__main__":
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
