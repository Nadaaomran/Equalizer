from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QUrl, QTimer, QObject, pyqtSignal
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import librosa
import os
import pandas as pd
from scipy.fft import fft, fftfreq, ifft
from scipy import signal
from PyQt5.QtCore import QRectF
import soundfile as sf
from itertools import permutations
from pyqtgraph import PlotWidget

class PlotUpdater(QObject):
    update_signal = pyqtSignal(int)

    def __init__(self, position, update_interval):
        super().__init__()
        self.position = position
        self.update_interval = update_interval
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)

    def start(self):
        self.timer.start(self.update_interval)

    def stop(self):
        self.timer.stop()

    def set_update_interval(self, interval):
        self.update_interval = interval

    def set_position(self, position):
        self.position = position
        
    def update(self):
        self.update_signal.emit(self.position)
        self.position += 1 

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1900, 970)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        MainWindow.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        style_sheet = """
            QWidget#centralwidget {
                background-color: #444444; /* Dark gray matte background */
                color: #ccc; /* Light gray text color */
            }

            QPushButton {
                border: 1px solid #666666; /* Darker border */
                border-radius: 5px;
                background-color: #555555; /* Dark gray */
                color: #ccc;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #666666; /* Slightly lighter gray on hover */
            }
            QPushButton:pressed {
                background-color: #444444; /* Darker gray when pressed */
            }

            QComboBox {
                border: 1px solid #666666;
                border-radius: 5px;
                background-color: #555555;
                color: #ccc;
                padding: 1px 18px 1px 3px;
                min-width: 6em;
            }
            QComboBox:editable {
                background: white;
            }
            QComboBox:!editable, QComboBox::drop-down:editable {
                background: #555555;
            }
            QComboBox:!editable:on, QComboBox::drop-down:editable:on {
                background: #666666;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left: none;
            }
            QComboBox::down-arrow {
                width: 15px;
                height: 15px;
                image: url(down_arrow.png); /* Path to your custom down arrow */
            }

            QSlider::groove:horizontal, QSlider::groove:vertical {
                border: 1px solid #666666;
                background: #555555;
                border-radius: 5px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                margin: 2px 0;
            }
            QSlider::groove:vertical {
                width: 8px;
                margin: 0 2px;
            }
            QSlider::handle:horizontal, QSlider::handle:vertical {
                background: #666666;
                border: 1px solid #666666;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                margin: -2px 0;
            }
            QSlider::handle:vertical {
                height: 18px;
                margin: 0 -2px;
            }

            QCheckBox {
                spacing: 5px;
                color: #ccc;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border: 1px solid #666666;
                border-radius: 3px;
                background: #555555;
            }
            QCheckBox::indicator:checked {
                background: #ccc; /* Adjust to your preferred checked color */
            }

            QRadioButton {
                spacing: 5px;
                color: #ccc;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
                border: 1px solid #666666;
                border-radius: 7px;
                background: #555555;
            }
            QRadioButton::indicator:checked {
                background: #ccc;
            }

            QLabel {
                color: #ccc;
            }

            QGroupBox {
                border: 1px solid #666666;
                border-radius: 5px;
                margin-top: 6px;
                background-color: #555555;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                color: #ccc;
            }

            PlotWidget {
                background-color: #555555;
                border: 1px solid #666666;
            }
        """


        self.centralwidget.setStyleSheet(style_sheet)
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.input_graph = PlotWidget(self.frame)
        # self.input_graph.setMinimumSize(QtCore.QSize(0, 130))
        # self.input_graph.setMaximumSize(QtCore.QSize(16777215, 200))
        # self.input_graph.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.input_graph.setObjectName("input_graph")

        self.input_plot = self.input_graph.getPlotItem()
        self.input_plot.showGrid(x=True, y=True)
        self.input_plot.setLabel('left', 'Amplitude')
        self.input_plot.setLabel('bottom', 'Time (s)')
        self.input_curve = self.input_plot.plot()

        self.verticalLayout.addWidget(self.input_graph)
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.output_graph = PlotWidget(self.frame)
        # self.output_graph.setMinimumSize(QtCore.QSize(0, 130))
        # self.output_graph.setMaximumSize(QtCore.QSize(16777215, 200))
        # self.output_graph.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.output_graph.setObjectName("output_graph")

        self.output_plot = self.output_graph.getPlotItem()
        self.output_plot.showGrid(x=True, y=True)
        self.output_plot.setLabel('left', 'Amplitude')
        self.output_plot.setLabel('bottom', 'Time (s)')
        self.output_curve = self.output_plot.plot()
        self.verticalLayout.addWidget(self.output_graph)

        self.scroll_slider = QtWidgets.QSlider(self.frame)
        self.scroll_slider.setOrientation(QtCore.Qt.Horizontal)
        self.scroll_slider.setObjectName("scroll_slider")
        self.verticalLayout.addWidget(self.scroll_slider)
        self.horizontalLayout3 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout3.setObjectName("horizontalLayout3")
        self.browse_button = QtWidgets.QPushButton(self.frame)
        self.browse_button.setObjectName("browse_button")
        self.horizontalLayout3.addWidget(self.browse_button)
        self.horizontalLayout3.setStretch(0,1)
        self.pause_button = QtWidgets.QPushButton(self.frame)
        self.pause_button.setObjectName("pause_button")
        self.horizontalLayout3.addWidget(self.pause_button)
        self.horizontalLayout3.setStretch(1,1)
        self.zoomin_button = QtWidgets.QPushButton(self.frame)
        self.zoomin_button.setObjectName("zoomin_button")
        self.horizontalLayout3.addWidget(self.zoomin_button)
        self.horizontalLayout3.setStretch(2,1)
        self.zoomout_button = QtWidgets.QPushButton(self.frame)
        self.zoomout_button.setObjectName("zoomout_button")
        self.horizontalLayout3.addWidget(self.zoomout_button)
        self.horizontalLayout3.setStretch(3,1)
        self.speed_label = QtWidgets.QLabel(self.frame)
        self.speed_label.setObjectName("speed_label")
        self.horizontalLayout3.addWidget(self.speed_label)
        self.horizontalLayout3.setStretch(4,0)
        self.speed_combobox = QtWidgets.QComboBox(self.frame)
        self.speed_combobox.setObjectName("speed_combobox")
        speed_values = ['0.5', '1', '1.5', '2']
        self.speed_combobox.addItems(speed_values)
        self.speed_combobox.setCurrentIndex(-1)
        self.horizontalLayout3.addWidget(self.speed_combobox)
        self.horizontalLayout3.setStretch(5,1)
        self.verticalLayout.addLayout(self.horizontalLayout3)
        self.gridLayout_3.addWidget(self.frame, 0, 0, 1, 1)
        self.frame2 = QtWidgets.QFrame(self.centralwidget)
        self.frame2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame2.setObjectName("frame2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.frame2)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.addWidget(self.label_3)
        self.input_spectrogram = PlotWidget(self.frame2)
        # self.input_spectrogram.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.input_spectrogram.setObjectName("input_spectrogram")
        self.verticalLayout_3.addWidget(self.input_spectrogram)
        self.label_4 = QtWidgets.QLabel(self.frame2)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.output_spectrogram = PlotWidget(self.frame2)
        # self.output_spectrogram.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.output_spectrogram.setObjectName("output_spectrogram")
        self.verticalLayout_3.addWidget(self.output_spectrogram)
        self.verticalLayout_3.addSpacing(30)
        self.horizontalLayout2 = QtWidgets.QHBoxLayout(self.frame2)
        self.horizontalLayout2.setObjectName("horizontalLayout2")
        self.hide_checkbox = QtWidgets.QCheckBox(self.frame2)
        self.hide_checkbox.setObjectName("hide_checkbox")
        self.horizontalLayout2.addSpacing(15)
        self.horizontalLayout2.addWidget(self.hide_checkbox)
        self.reset_button = QtWidgets.QPushButton(self.frame2)
        self.reset_button.setObjectName("reset_button")
        self.horizontalLayout2.addStretch()
        self.horizontalLayout2.addWidget(self.reset_button)
        self.verticalLayout_3.addLayout(self.horizontalLayout2)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.verticalLayout_8.addSpacing(30)
        self.mode_groupbox = QtWidgets.QGroupBox(self.centralwidget)
        self.mode_groupbox.setObjectName("mode_groupbox")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.mode_groupbox)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.uniform_mood = QtWidgets.QRadioButton(self.mode_groupbox)
        self.uniform_mood.setObjectName("uniform_mood")
        self.verticalLayout_6.addWidget(self.uniform_mood)
        self.verticalLayout_6.addSpacing(20)
        self.uniform_mood.setChecked(True) 
        self.musical_mood = QtWidgets.QRadioButton(self.mode_groupbox)
        self.musical_mood.setObjectName("musical_mood")
        self.verticalLayout_6.addWidget(self.musical_mood)
        self.verticalLayout_6.addSpacing(20)
        self.animals_mood = QtWidgets.QRadioButton(self.mode_groupbox)
        self.animals_mood.setObjectName("animals_mood")
        self.verticalLayout_6.addWidget(self.animals_mood)
        self.verticalLayout_6.addSpacing(20)
        self.ecg_mood = QtWidgets.QRadioButton(self.mode_groupbox)
        self.ecg_mood.setObjectName("ecg_mood")
        self.verticalLayout_6.addWidget(self.ecg_mood)
        self.verticalLayout_6.addSpacing(20)
        self.verticalLayout_8.addWidget(self.mode_groupbox)
        self.window_groupbox = QtWidgets.QGroupBox(self.centralwidget)
        self.window_groupbox.setObjectName("window_groupbox")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.window_groupbox)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.window_combobox = QtWidgets.QComboBox(self.window_groupbox)
        self.window_combobox.setObjectName("window_combobox")
        self.gridLayout_8.addWidget(self.window_combobox, 0, 0, 1, 1)
        self.verticalLayout_8.addWidget(self.window_groupbox)
        self.verticalLayout_8.addSpacing(80)
        self.gridLayout_3.addWidget(self.frame2, 0, 1, 1, 1)
        self.gridLayout_3.addLayout(self.verticalLayout_8, 0, 2, 1, 1)
        
        self.frame3 = QtWidgets.QFrame(self.centralwidget)
        self.frame3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame3.setObjectName("frame3")
        self.gridLayout_1 = QtWidgets.QGridLayout(self.frame3)
        self.gridLayout_1.setObjectName("gridLayout_3")
        self.verticalLayout5 = QtWidgets.QVBoxLayout(self.frame3)
        self.verticalLayout5.setObjectName("verticalLayout5")
        self.FT_graph = PlotWidget(self.frame3)
        # self.FT_graph.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.FT_graph.setObjectName("FT_graph")
        self.verticalLayout5.addWidget(self.FT_graph)
        self.verticalLayout5.addSpacing(30)
        self.gridLayout_1.addLayout(self.verticalLayout5, 0,0,1,1)
        self.verticalLayout6 = QtWidgets.QVBoxLayout(self.frame3)
        self.verticalLayout6.setObjectName("verticalLayout6")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame3)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame3)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.slider1 = QtWidgets.QSlider(self.frame3)
        self.slider1.setOrientation(QtCore.Qt.Vertical)
        self.slider1.setObjectName("slider1")
        self.verticalLayout_7.addWidget(self.slider1)
        self.slider1.sliderReleased.connect(lambda: self.fourier_transform(self.data))
        self.slider1.sliderReleased.connect(lambda: self.plot_selected_window(0,self.slider1.value()))
        self.label1 = QtWidgets.QLabel(self.frame3)
        self.label1.setObjectName("label1")
        self.verticalLayout_7.addWidget(self.label1)
        self.horizontalLayout_7.addLayout(self.verticalLayout_7)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frame3)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.slider2 = QtWidgets.QSlider(self.frame3)
        self.slider2.setOrientation(QtCore.Qt.Vertical)
        self.slider2.setObjectName("slider2")
        self.verticalLayout_8.addWidget(self.slider2)
        self.slider2.sliderReleased.connect(lambda: self.fourier_transform(self.data))
        self.slider2.sliderReleased.connect(lambda: self.plot_selected_window(1,self.slider2.value()))
        self.label2 = QtWidgets.QLabel(self.frame3)
        self.label2.setObjectName("label2")
        self.verticalLayout_8.addWidget(self.label2)
        self.horizontalLayout_7.addLayout(self.verticalLayout_8)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.frame3)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.slider3 = QtWidgets.QSlider(self.frame3)
        self.slider3.setOrientation(QtCore.Qt.Vertical)
        self.slider3.setObjectName("slider3")
        self.verticalLayout_9.addWidget(self.slider3)
        self.slider3.sliderReleased.connect(lambda: self.fourier_transform(self.data))
        self.slider3.sliderReleased.connect(lambda: self.plot_selected_window(2,self.slider3.value()))
        self.label3 = QtWidgets.QLabel(self.frame3)
        self.label3.setObjectName("label3")
        self.verticalLayout_9.addWidget(self.label3)
        self.horizontalLayout_7.addLayout(self.verticalLayout_9)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.slider4 = QtWidgets.QSlider(self.frame3)
        self.slider4.setOrientation(QtCore.Qt.Vertical)
        self.slider4.setObjectName("slider4")
        self.verticalLayout_10.addWidget(self.slider4)
        self.slider4.sliderReleased.connect(lambda: self.fourier_transform(self.data))
        self.slider4.sliderReleased.connect(lambda: self.plot_selected_window(3,self.slider4.value()))
        self.label4 = QtWidgets.QLabel(self.frame3)
        self.label4.setObjectName("label4")
        self.verticalLayout_10.addWidget(self.label4)
        self.horizontalLayout_7.addLayout(self.verticalLayout_10)
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.frame3)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.slider5 = QtWidgets.QSlider(self.frame3)
        self.slider5.setOrientation(QtCore.Qt.Vertical)
        self.slider5.setObjectName("slider5")
        self.verticalLayout_11.addWidget(self.slider5)
        self.slider5.sliderReleased.connect(lambda: self.fourier_transform(self.data))
        self.slider5.sliderReleased.connect(lambda: self.plot_selected_window(4,self.slider5.value()))
        self.label5 = QtWidgets.QLabel(self.frame3)
        self.label5.setObjectName("label5")
        self.verticalLayout_11.addWidget(self.label5)
        self.horizontalLayout_7.addLayout(self.verticalLayout_11)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.frame3)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.slider6 = QtWidgets.QSlider(self.frame3)
        self.slider6.setOrientation(QtCore.Qt.Vertical)
        self.slider6.setObjectName("slider6")
        self.verticalLayout_12.addWidget(self.slider6)
        self.slider6.sliderReleased.connect(lambda: self.fourier_transform(self.data))
        self.slider6.sliderReleased.connect(lambda: self.plot_selected_window(5,self.slider6.value()))
        self.label6 = QtWidgets.QLabel(self.frame3)
        self.label6.setObjectName("label6")
        self.verticalLayout_12.addWidget(self.label6)
        self.horizontalLayout_7.addLayout(self.verticalLayout_12)
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.slider7 = QtWidgets.QSlider(self.frame3)
        self.slider7.setOrientation(QtCore.Qt.Vertical)
        self.slider7.setObjectName("slider7")
        self.verticalLayout_13.addWidget(self.slider7)
        self.slider7.sliderReleased.connect(lambda: self.fourier_transform(self.data))
        self.slider7.sliderReleased.connect(lambda: self.plot_selected_window(6,self.slider7.value()))
        self.label7 = QtWidgets.QLabel(self.frame3)
        self.label7.setObjectName("label7")
        self.verticalLayout_13.addWidget(self.label7)
        self.horizontalLayout_7.addLayout(self.verticalLayout_13)
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.frame3)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.slider8 = QtWidgets.QSlider(self.frame3)
        self.slider8.setOrientation(QtCore.Qt.Vertical)
        self.slider8.setObjectName("slider8")
        self.verticalLayout_14.addWidget(self.slider8)
        self.slider8.sliderReleased.connect(lambda: self.fourier_transform(self.data))
        self.slider8.sliderReleased.connect(lambda: self.plot_selected_window(7,self.slider8.value()))
        self.label8 = QtWidgets.QLabel(self.frame3)
        self.label8.setObjectName("label8")
        self.verticalLayout_14.addWidget(self.label8)
        self.horizontalLayout_7.addLayout(self.verticalLayout_14)
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.frame3)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.slider9 = QtWidgets.QSlider(self.frame3)
        self.slider9.setOrientation(QtCore.Qt.Vertical)
        self.slider9.setObjectName("slider9")
        self.verticalLayout_15.addWidget(self.slider9)
        self.slider9.sliderReleased.connect(lambda: self.fourier_transform(self.data))
        self.slider9.sliderReleased.connect(lambda: self.plot_selected_window(8,self.slider9.value()))
        self.label9 = QtWidgets.QLabel(self.frame3)
        self.label9.setObjectName("label9")
        self.verticalLayout_15.addWidget(self.label9)
        self.horizontalLayout_7.addLayout(self.verticalLayout_15)
        self.verticalLayout_16 = QtWidgets.QVBoxLayout(self.frame3)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.slider10 = QtWidgets.QSlider(self.frame3)
        self.slider10.setOrientation(QtCore.Qt.Vertical)
        self.slider10.setObjectName("slider10")
        self.verticalLayout_16.addWidget(self.slider10)
        self.slider10.sliderReleased.connect(lambda: self.fourier_transform(self.data))
        self.slider10.sliderReleased.connect(lambda: self.plot_selected_window(9,self.slider10.value()))
        self.label10 = QtWidgets.QLabel(self.frame3)
        self.label10.setObjectName("label10")
        self.verticalLayout_16.addWidget(self.label10)
        self.horizontalLayout_7.addLayout(self.verticalLayout_16)
        
        # self.window_groupbox = QtWidgets.QGroupBox(self.frame3)
        # self.window_groupbox.setObjectName("window_groupbox")
        # self.gridLayout_8 = QtWidgets.QGridLayout(self.window_groupbox)
        # self.gridLayout_8.setObjectName("gridLayout_8")
        # self.window_combobox = QtWidgets.QComboBox(self.window_groupbox)
        # self.window_combobox.setObjectName("window_combobox")
        # self.gridLayout_8.addWidget(self.window_combobox, 0, 0, 1, 1)
        # self.horizontalLayout_7.addWidget(self.window_groupbox)
        self.verticalLayout6.addLayout(self.horizontalLayout_7)
        self.gridLayout_1.addLayout(self.verticalLayout6, 2,0,2,1)
        self.gridLayout_3.addWidget(self.frame3, 1, 0, 1, 3)
        MainWindow.setCentralWidget(self.centralwidget)
        # self.menubar = QtWidgets.QMenuBar(MainWindow)
        # self.menubar.setGeometry(QtCore.QRect(0, 0, 840, 21))
        # self.menubar.setObjectName("menubar")
        # MainWindow.setMenuBar(self.menubar)
        # self.statusbar = QtWidgets.QStatusBar(MainWindow)
        # self.statusbar.setObjectName("statusbar")
        # MainWindow.setStatusBar(self.statusbar)

        self.input_graph.scene().sigMouseClicked.connect(lambda: self.play_input_signal(True))
        self.output_graph.scene().sigMouseClicked.connect(lambda: self.play_input_signal(False))
        self.scroll_slider.sliderReleased.connect(self.update_plotting_interval)
        self.hide_checkbox.clicked.connect(self.hide_spectrogram)
        self.reset_button.clicked.connect(self.reset)
        self.browse_button.clicked.connect(self.get_file)
        self.pause_button.clicked.connect(self.start_plotting)
        self.speed_combobox.activated.connect(self.control_plotting_speed)

        self.player_input = QMediaPlayer()
        self.player_input.setMuted(True)
        self.player_output= QMediaPlayer()
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.musical_mood.clicked.connect(lambda: self.mood_change(1))
        self.animals_mood.clicked.connect(lambda: self.mood_change(2))
        self.ecg_mood.clicked.connect(lambda: self.mood_change(3))
        self.uniform_mood.clicked.connect(lambda: self.mood_change(0))
        self.sliders_labels=[["","","","","","","","","",""],
                        ["Drums","Cello","Flute","Mandolin"],
                        ["Crow","Owl","Wolf","Frog"],
                        ["Arrythmia 1","Arrythmia 2","Arrythmia 3"]]
        
        self.condition=[[[0,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,90],[90,100]],
                        [[0,500],[700,1000],[500,700],[1000,4000]],
                        [[1000,2000],[500,1000],[0,500],[2000,3000]],
                        [[50,51],[52,54],[55,57]]]
        self.sliders_names=['slider1','slider2','slider3','slider4', 'slider5','slider6', 'slider7', 'slider8','slider9','slider10']
        self.index=0
        self.set_sliders_labels()
        self.file_path= None
        self.zoomin_button.clicked.connect(self.zoom_in)
        self.zoomout_button.clicked.connect(self.zoom_out)
        window_names = ['Rectangle', 'Hamming', 'Hann', 'Gaussian']
        self.window_combobox.addItems(window_names)
        self.window_combobox.setCurrentIndex(0)
        #make a timer object from our class 
        self.plot_updater = PlotUpdater(0, 300)
        #connect the signal emitted by the timer to the function which gets the realtime data
        self.plot_updater.update_signal.connect(self.get_data)
        self.scale_factor=1
        for slider_name in (self.sliders_names):
            slider= getattr(self, slider_name)
            slider.setValue(1)
            slider.setMaximum(10)
            slider.setSingleStep(1)

        
    def set_media_to_mediaplayers(self, content):
        self.player_input.setMedia(content)
        self.player_output.setMedia(content)

    def stop_media_players(self):
        self.player_input.pause()
        self.player_output.pause()

    def play_media_players(self):
        self.player_input.play()
        self.player_output.play()

    def set_media_players_position(self, position):
        self.player_input.setPosition(position)
        self.player_output.setPosition(position)

    def set_media_players_speed(self, speed):
        self.player_input.setPlaybackRate(speed)
        self.player_output.setPlaybackRate(speed)

    def set_plotter_speed(self, speed):
        self.plot_updater.set_update_interval(speed) 
        self.plot_updater.start()

    def mood_change(self, index):
        #return all the controls to its intial position 
        self.reset()
        #empty the content of the media players 
        self.set_media_to_mediaplayers(QMediaContent())
        #unselect the data file 
        self.file_path= None
        #return the scrolling slider to its minimum value
        self.scroll_slider.setValue(0)
        #update the index according to the selected mood 
        self.index=index
        #update the UI according to the selected mood
        self.toggle_slider_visibility()
        #clear the plots 
        self.clear_plots()
    
    def clear_plots(self):
        self.input_plot.clear()
        self.output_plot.clear()
        self.input_curve = self.input_plot.plot()
        self.output_curve = self.output_plot.plot()
        self.FT_graph.clear()
        self.input_spectrogram.clear()
        self.output_spectrogram.clear()

    #to hide the spectrogarms if the hide checkbox is checked 
    def hide_spectrogram(self):
        self.input_spectrogram.setHidden(self.hide_checkbox.isChecked())
        self.output_spectrogram.setHidden(self.hide_checkbox.isChecked())
        self.label_3.setHidden(self.hide_checkbox.isChecked())
        self.label_4.setHidden(self.hide_checkbox.isChecked())

    #zoom in in both the input and output graph by the same factor
    def zoom_in(self):
         self.scale_factor*=3/4
         
    #zoom out in both the input and output graph by the same factor
    def zoom_out(self):
        self.scale_factor*=5/4

    def reset(self):
        if self.file_path:
            #restart the signal
            self.plot_updater.set_position(0)
            self.scale_factor=1
            #if the current signal has sound as in the animal and musical moods, restart it 
            if self.index in [1,2]:
                self.set_media_players_position(0)
                self.play_media_players()
            #return the slider values to its default position
            for slider_index, slider_name in enumerate(self.sliders_names):
                slider= getattr(self, slider_name)
                slider.setValue(1)
            #if the signal is paused, resume it
            if self.pause_button.text() == "Resume":
                self.start_plotting()
            #return the speed to its normal value 
            self.speed_combobox.setCurrentIndex(1)
            self.control_plotting_speed()
            #update the fourier transform plot
            self.fourier_transform(self.data)

    #play only the input or the output audio according to which widget the user pressed
    def play_input_signal(self, flag):
        if (flag):
            self.player_output.setMuted(True)
            self.player_input.setMuted(False)
        else:
            self.player_output.setMuted(False)
            self.player_input.setMuted(True)

    #update the UI according to which mood is selected
    def toggle_slider_visibility(self):
        if self.index:
            for slider_index, slider_name in enumerate(self.sliders_names[len(self.sliders_labels[self.index]):], start=len(self.sliders_labels[self.index])):
                self.horizontalLayout_7.setContentsMargins(300,0,0,0)
                self.hide_object(slider_name)
                self.hide_object('label'+ str(slider_index +1))
        else:
            for slider_index, slider_name in enumerate(self.sliders_names):
                self.horizontalLayout_7.setContentsMargins(0,0,0,0)
                self.hide_object(slider_name)
                self.hide_object('label'+ str(slider_index +1))

        self.set_sliders_labels()

    def hide_object(self, object_name):
        obj= getattr(self, object_name)
        obj.setHidden(self.index)
    
    #update the labels of the sliders accourding to which mood is selected 
    def set_sliders_labels(self):
        for i in range(1, len(self.sliders_labels[self.index])+1):
            label= getattr(self, 'label'+ str(i))
            label.setText(self.sliders_labels[self.index][i-1])
    
    def get_file(self):
        file_dialog = QFileDialog()
        #select the data file
        file_path, _ = file_dialog.getOpenFileName(None, "Open CSV or Audio File", os.path.expanduser("~"), "CSV or Audio Files (*.csv *.mp3 *.wav)")
        if file_path:
            self.file_path= file_path
            #update the position to start from the beginning of the signal 
            position=0
        else: 
            return
        if self.ecg_mood.isChecked() or self.uniform_mood.isChecked(): 
            self.read_csv_data()
            
        else:
            self.read_audio_data()
            
        #set the output data to be the same as the original data as there is no controls done yet 
        self.y_output= self.data
        #start plotting the loaded signal
        self.start_realtime_plot(position)
        #group the frequencies of the signal so that each group of frequencies is controlled through a slider
        self.get_sliders_indices()
        #get the fourier transform for the selected data 
        self.fourier_transform(self.data)  
        #plot the spectrogram for the original data
        self.add_spectrogram("input_spectrogram", self.data)
        self.reset()
    
    def read_csv_data(self):
        #load the dataframe from the data file
        self.df = pd.read_csv(self.file_path)
        #get the voltage amplitudes to apply fourier transform
        self.data = self.df['Voltage'].values
        #get the sampling frequency from the data file
        self.sampling_freq= self.df['Sampling frequency'].values[0]

    def read_audio_data(self):
        #get the content from the audio file
        content = QMediaContent(QUrl.fromLocalFile(self.file_path))
        #set the input and output player's content to that of the selected audio
        self.set_media_to_mediaplayers(content)
        #get the sampling rate of the audio
        self.sampling_freq= librosa.get_samplerate(self.file_path)
        #get the waveform of the audio
        self.data,_= librosa.load(self.file_path ,sr= self.sampling_freq)
        #start playing the audio
        self.play_media_players()

    #start the timer object we made from our class        
    def start_realtime_plot(self, position): 
        self.plot_updater.set_position(position)
        self.plot_updater.start()
        #if the signal is paused, resume it
        if self.pause_button.text() == "Resume":
            self.pause_button.setText("Pause")

    def get_sliders_indices(self):
        if self.file_path:
            self.compute_fourier_magnitude(self.data)
            if not self.index:
                self.get_uniform_range()
            self.sliders_indices ={i: [] for i in range(10)}
            for index, value in enumerate(self.frequencies):
                for i in range(len(self.condition[self.index])):
                    if self.condition[self.index][i][0] <= value <= self.condition[self.index][i][1]:
                        self.sliders_indices[i].append(index)
            self.add_spectrogram("output_spectrogram", self.y_output)

            
    #the function which is repeatedly called to get the new data to be plotted 
    def get_data(self, position):
        #if a data file is selected
        if self.file_path:
            #check if the data file type is CSV
            if self.ecg_mood.isChecked() or self.uniform_mood.isChecked(): 
                #check that the end of the data file isn't reached yet
                if position <= len(self.data)- 400:
                    #set the length of the signal which will be plotted in each call
                    samples_per_frame=400
                    self.get_current_data(position, position+ samples_per_frame, samples_per_frame)
            else:
                #get the current position of the mediaplayer 
                position = self.player_input.position() / 1000.0
                #set the length of the signal which will be plotted in each call
                samples_per_frame = int(self.plot_updater.update_interval * self.sampling_freq/1000)
                #start index
                start_index= int(position * self.sampling_freq)
                #end index
                end_index= start_index + samples_per_frame
                if start_index < len(self.data)- samples_per_frame:
                    self.get_current_data(start_index, end_index, samples_per_frame)
                    #if the signal is fully loaded by the qmediaplayer 
            if self.player_input.duration() or self.index in [0,3]:
                #update the scrolling slider to move with the signal 
                self.update_scrolling_slider_value()
    
    def get_current_data(self, start_index, end_index, samples_per_frame):
         #get the values which correspond to the current time 
        self.y_values= self.data[start_index: end_index]
        #read from the processed signal the values which correspond to the current time to be plotted in the output graph
        current_y_output= self.y_output[start_index: end_index]
        #send the new data to the plotting function
        self.update_plot(self.input_plot, self.input_curve, start_index, end_index, samples_per_frame, self.y_values)
        self.update_plot(self.output_plot, self.output_curve, start_index, end_index, samples_per_frame, current_y_output)

    #update the graphs based on the new data
    def update_plot(self, plot_widget, plot_data_item, start_index,end_index, samples_per_frame, y_values):
        plot_data_item.setData(np.linspace(start_index/self.sampling_freq, end_index/self.sampling_freq, samples_per_frame), y_values)
        plot_widget.setXRange(start_index/self.sampling_freq , end_index/self.sampling_freq )
        if abs(y_values.min()) < pow(10, -10) and not abs(y_values.min()) == 0:
            plot_widget.setYRange(1, -1)
        else:
            plot_widget.setYRange(y_values.min() *self.scale_factor, y_values.max() *self.scale_factor)
    
    #function of the pause button
    def start_plotting(self):
        #if the signal is running stop it
        if self.pause_button.text() == "Pause":
            self.pause_button.setText("Resume")
            if self.ecg_mood.isChecked() or self.uniform_mood.isChecked(): 
                self.plot_updater.stop()
            else:
                self.stop_media_players()
        #if the signal is paused resume it
        else:
            self.pause_button.setText("Pause")
            if self.ecg_mood.isChecked() or self.uniform_mood.isChecked():
                self.plot_updater.start()
            else:
                self.play_media_players()


    #change the timer update interval based on the selected speed
    def control_plotting_speed(self):
        if self.speed_combobox.currentIndex() == 0:   #0.5x
            if self.ecg_mood.isChecked() or self.uniform_mood.isChecked():
                self.set_plotter_speed(300)
            else:
                self.set_media_players_speed(0.5)
        elif self.speed_combobox.currentIndex() == 1 :     #1x
            if self.ecg_mood.isChecked() or self.uniform_mood.isChecked():
                self.set_plotter_speed(200)
            else:
                self.set_media_players_speed(1)
        elif self.speed_combobox.currentIndex() == 2 :   #1.5x
            if self.ecg_mood.isChecked() or self.uniform_mood.isChecked():
                self.set_plotter_speed(100)
            else:
                self.set_media_players_speed(1.5)
        elif self.speed_combobox.currentIndex() == 3 :     #2x
            if self.ecg_mood.isChecked() or self.uniform_mood.isChecked():
                self.set_plotter_speed(50)
            else:
                self.set_media_players_speed(2)
    
    #scroll the signal, forward and backward
    def update_plotting_interval(self):
        if self.ecg_mood.isChecked() or self.uniform_mood.isChecked():
            self.plot_updater.set_position(int(self.scroll_slider.value() * len(self.data)/self.scroll_slider.maximum()))  
            if self.pause_button.text() == "Resume":
                self.get_data(int(self.scroll_slider.value() * len(self.data)/self.scroll_slider.maximum()))
        else:
            self.set_media_players_position(int((self.scroll_slider.value() *self.player_input.duration())/self.scroll_slider.maximum()))

    #update the scrolling slider value to move with the signal
    def update_scrolling_slider_value(self):
        if self.ecg_mood.isChecked() or self.uniform_mood.isChecked():
            self.scroll_slider.setValue(int(self.plot_updater.position* self.scroll_slider.maximum()/ (len(self.data)- int(self.input_plot.width()))))
        else:
            self.scroll_slider.setValue(int((self.player_input.position() * self.scroll_slider.maximum()) /self.player_input.duration()))

    def compute_fourier_magnitude(self, data):
        self.fourier_magnitude = np.abs(fft(self.data))
        self.frequencies = fftfreq(len(data), d=1/self.sampling_freq)
        self.positive_freq_indices = self.frequencies >= 0
        self.negative_freq_indices= self.frequencies<0
        self.fourier_magnitude[self.negative_freq_indices]= 0
        self.fourier_magnitude[self.positive_freq_indices]*= 2
        self.positive_freq= self.frequencies[self.positive_freq_indices]
        

    def fourier_transform(self, data):
        self.FT_graph.clear()
        self.compute_fourier_magnitude(data)
        self.get_y_output()
        self.FT_graph.plot(self.positive_freq, self.fourier_magnitude[self.positive_freq_indices])
        self.FT_graph.setLabel("left", "Amplitude")
        self.FT_graph.setLabel("bottom", "Frequency")

    def plot_selected_window(self,i, value):
        if len(self.sliders_indices[i]) != 0 :
            x,y= self.window_calculations(i, value, self.fourier_magnitude.max()/50)
            self.FT_graph.plot(x, y, pen='r')

    def plot_spectrogram(self, data):
        f, t, Sxx = signal.spectrogram(data, fs=self.sampling_freq)
        small_offset = 1e-10
        Sxx_db = 10 * np.log10(Sxx + small_offset)
        img_data= Sxx_db.T
        img = pg.ImageItem()
        img.setImage(img_data)
        colormap = pg.colormap.get('viridis')  
        img.setColorMap(colormap)
        img.setRect(QRectF(0, 0, t[-1], self.positive_freq[-1]))
        return(img)
    
    def add_spectrogram(self,name, data):
        spectrogram = getattr(self, name)
        spectrogram.addItem(self.plot_spectrogram(data))
        spectrogram.setLabel('bottom', "Time", units='s')
        spectrogram.setLabel('left', "Frequency", units='Hz')

    def window_calculations(self, i, value, factor):
        selected_window = self.window_combobox.currentText()
        slider_range=[self.condition[self.index][i][0], self.condition[self.index][i][1]]
        x = np.linspace(slider_range[0], slider_range[1], len(self.sliders_indices[i])-1)
        y=[]
        if selected_window == 'Rectangle':
            y = np.ones_like(x)*value * factor
        elif selected_window == 'Hamming':
            y = np.hamming(len(x))*value * factor
        elif selected_window == 'Hann':
            y = np.hanning(len(x))* value * factor
        elif selected_window == 'Gaussian':
            y = np.exp(-0.5 * ((x - np.mean(x)) / np.std(x)) ** 2)* value * factor
        return(x,y)

    def smoothing_window_multiplication(self,i , value):
        x, y= self.window_calculations(i, value, 1)
        self.z = np.concatenate((np.ones_like(self.frequencies[:self.sliders_indices[i][0]]), y, np.ones_like(self.frequencies[self.sliders_indices[i][-1]:])))
        return(self.z)

    def get_uniform_range(self):
        step= self.sampling_freq/(10 *2)
        for i in range(0,10):
            label= getattr(self, 'label'+ str(i+1))
            label.setText(str(step * i) + ":" + str(step * (i + 1)))
            self.condition[self.index][i][0]= step*i
            self.condition[self.index][i][1]= step*(i+1)

    def get_y_output(self):
        for slider_index, slider_name in enumerate(self.sliders_names):
            slider = getattr(self, slider_name)
            if slider.value() != 1 and len(self.sliders_indices[slider_index]) != 0 :
                self.fourier_magnitude*= self.smoothing_window_multiplication(slider_index, slider.value())
        phase = np.angle(fft(self.data))
        fourier_transform = self.fourier_magnitude * np.exp(1j * phase)
        self.y_output = ifft(fourier_transform).real
        if self.index in [1, 2]:
            self.handle_audio_output()
        self.add_spectrogram("output_spectrogram",self.y_output)

    def handle_audio_output(self):
        self.player_output.setMedia(QMediaContent())
        output_file_path = 'processed_signal.mp3'
        sf.write(output_file_path, self.y_output, self.sampling_freq)
        full_path = r'C:\Users\DELL\Downloads\processed_signal.mp3'
        self.player_output.setMedia(QMediaContent(QUrl.fromLocalFile(full_path)))
        self.player_output.setPosition(self.player_input.position())
        if self.pause_button.text() == "Pause":
            self.player_output.play()
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Original Plot"))
        self.label_2.setText(_translate("MainWindow", "Processed Plot"))
        self.label_3.setText(_translate("MainWindow", "Original Spectrogram"))
        self.label_4.setText(_translate("MainWindow", "Processed Spectrogram"))
        self.hide_checkbox.setText(_translate("MainWindow", "Hide Spectrograms"))
        self.reset_button.setText(_translate("MainWindow", "Reset"))
        self.uniform_mood.setText(_translate("MainWindow", "Uniform Signals"))
        self.musical_mood.setText(_translate("MainWindow", "Musical Instruments"))
        self.animals_mood.setText(_translate("MainWindow", "Animals"))
        self.ecg_mood.setText(_translate("MainWindow", "ECG Arrhythmias"))
        self.browse_button.setText(_translate("MainWindow", "Browse"))
        self.pause_button.setText(_translate("MainWindow", "Pause"))
        self.zoomin_button.setText(_translate("MainWindow", "Zoom In"))
        self.zoomout_button.setText(_translate("MainWindow", "Zoom Out"))
        self.speed_label.setText(_translate("MainWindow", "Speed"))
        self.label1.setText(_translate("MainWindow", "Text"))
        self.label2.setText(_translate("MainWindow", "Text"))
        self.label3.setText(_translate("MainWindow", "Text"))
        self.label4.setText(_translate("MainWindow", "Text"))
        self.label5.setText(_translate("MainWindow", "Text"))
        self.label6.setText(_translate("MainWindow", "Text"))
        self.label7.setText(_translate("MainWindow", "Text"))
        self.label8.setText(_translate("MainWindow", "Text"))
        self.label9.setText(_translate("MainWindow", "Text"))
        self.label10.setText(_translate("MainWindow", "Text"))
        self.window_groupbox.setTitle(_translate("MainWindow", "Smoothing Window"))
        self.mode_groupbox.setTitle(_translate("MainWindow", "Mode"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
