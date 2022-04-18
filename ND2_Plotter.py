#!/usr/bin/env /usr/bin/python3

import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import (
							FigureCanvasQTAgg as FigureCanvas,
							NavigationToolbar2QT as NavigationToolbar
							)
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors, ticker, cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from PIL import Image
from scipy import ndimage as ndimage
from scipy.ndimage import filters as filters
from scipy.spatial import Delaunay, Voronoi, ConvexHull
import mahotas as mh
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtGui import QIntValidator, QMouseEvent
from PyQt5.QtWidgets import (
							QApplication, QLabel, QWidget,
							QPushButton, QHBoxLayout, QVBoxLayout,
							QComboBox, QCheckBox, QSlider, QProgressBar,
							QFormLayout, QLineEdit, QTabWidget,
							QSizePolicy, QFileDialog, QMessageBox
							)
from pathlib import Path
from nd2reader import ND2Reader

################################################################################
# colormaps for matplotlib #
############################

red_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		
		'green': ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		
		'blue':  ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		
		'alpha': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		}

red_cmap = LinearSegmentedColormap('red_cmap', red_cdict)
cm.register_cmap(cmap=red_cmap)

green_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		
		'green': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		
		'blue':  ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		
		'alpha': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		}

green_cmap = LinearSegmentedColormap('green_cmap', green_cdict)
cm.register_cmap(cmap=green_cmap)

transparent_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		
		'green': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		
		'blue':  ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		
		'alpha': ((0, 1.0, 1.0),
				  (1, 0.0, 0.0)),
		}

transparent_cmap = LinearSegmentedColormap('transparent_cmap',
											transparent_cdict)
cm.register_cmap(cmap=transparent_cmap)

################################################################################
# canvas widget to put matplotlib plot #
########################################

class MPLCanvas(FigureCanvas):
	def __init__ (self, parent=None, width=8, height=8, dpi=100):
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.ax = self.fig.add_subplot(111)
		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self,
				QSizePolicy.Expanding,
				QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.fig.tight_layout()
		self.dapi_image = np.ones((512,512),dtype=int)
		self.green_image = np.zeros((512,512),dtype=int)
		self.red_image = np.zeros((512,512),dtype=int)
		self.box = np.array([[0,512], [0,512]])
		self.dapi_plot = None
		self.green_plot = None
		self.red_plot = None
		self.box_plot = None
		self.show_green = True
		self.show_red = True
		self.show_box = False
		self.select_box = None
		self.dapi_centres = np.zeros((0,2))
		self.green_cells = np.zeros((0,1))
		self.red_cells = np.zeros((0,1))
		self.dapi_centres_plot = None
		self.green_centres_plot = None
		self.red_centres_plot = None
		self.plot()
	
	def update_images (self, dapi_image, green_image, red_image,
						show_green = True, show_red = True,
						box = np.array([[0,512], [0,512]]),
						show_box = False):
		self.dapi_image = dapi_image
		self.green_image = green_image
		self.red_image = red_image
		self.show_green = show_green
		self.show_red = show_red
		self.box = box
		self.show_box = show_box
		self.plot()
	
	def update_centres (self, dapi_centres, green_cells, red_cells):
		self.dapi_centres = dapi_centres
		self.green_cells = green_cells
		self.red_cells = red_cells
		self.plot()
	
	def plot (self):
		if self.dapi_plot is not None:
			self.dapi_plot.remove()
		if self.green_plot is not None:
			self.green_plot.remove()
		if self.red_plot is not None:
			self.red_plot.remove()
		# plots
		self.ax.set_xlim(left = 0, right = len(self.dapi_image[0,:]))
		self.ax.set_ylim(bottom = 0, top = len(self.dapi_image[:,0]))
		self.dapi_plot = self.ax.imshow(self.dapi_image ,cmap=transparent_cmap)
		if self.show_green:
			self.green_plot = self.ax.imshow(self.green_image ,cmap=green_cmap)
		else:
			self.green_plot = None
		if self.show_red:
			self.red_plot = self.ax.imshow(self.red_image ,cmap=red_cmap)
		else:
			self.red_plot = None
		self.plot_box()
		self.plot_centres()
	#	self.fig.canvas.draw_idle()
		self.draw()
	
	def plot_box (self):
		self.remove_box()
		if self.show_box:
			self.box_plot = self.ax.plot((self.box[0,0], self.box[0,1],
											self.box[0,1], self.box[0,0],
												self.box[0,0]),
										 (self.box[1,0], self.box[1,0],
											self.box[1,1], self.box[1,1],
												self.box[1,0]),
										 color='yellow', linestyle='-')
		else:
			self.box_plot = None
	
	def remove_box (self):
		if self.box_plot is not None:
			if isinstance(self.box_plot,list):
				for line in self.box_plot:
					line.remove()
			else:
			#	self.box_plot.remove()
				self.box_plot = None
	
	def plot_centres (self):
		self.remove_centres()
		scale = 800./(self.dapi_image.shape[0]) + \
				800./(self.dapi_image.shape[1])
		if self.dapi_centres.shape[0] > 0:
			self.dapi_centres_plot = self.ax.plot(
									self.dapi_centres[:,0],
									self.dapi_centres[:,1],
									color = 'white', linestyle = '', 
									marker = 'o', markersize = scale)
			if self.red_cells is not None:
				if self.red_cells.shape[0] > 0:
					self.red_centres_plot = self.ax.plot(
									self.dapi_centres[self.red_cells,0],
									self.dapi_centres[self.red_cells,1],
									color = 'crimson', linestyle = '', 
									marker = '+', markersize = scale)
			if self.green_cells is not None:
				if self.green_cells.shape[0] > 0:
					self.green_centres_plot = self.ax.plot(
									self.dapi_centres[self.green_cells,0],
									self.dapi_centres[self.green_cells,1],
									color = 'seagreen', linestyle = '',
									marker = 'x', markersize = scale)
	
	def remove_centres (self):
		if self.dapi_centres_plot is not None:
			if isinstance(self.dapi_centres_plot,list):
				for line in self.dapi_centres_plot:
					line.remove()
			self.dapi_centres_plot = None
		if self.green_centres_plot is not None:
			if isinstance(self.green_centres_plot,list):
				for line in self.green_centres_plot:
					line.remove()
			self.green_centres_plot = None
		if self.red_centres_plot is not None:
			if isinstance(self.red_centres_plot,list):
				for line in self.red_centres_plot:
					line.remove()
			self.red_centres_plot = None
	
	def plot_selector (self, p_1, p_2):
		self.remove_selector()
		self.select_box = self.ax.plot((p_1[0], p_2[0], p_2[0], p_1[0],
										p_1[0]),
									   (p_1[1], p_1[1], p_2[1], p_2[1],
										p_1[1]),
									 c='white', ls='-')
		self.draw()
	
	def remove_selector (self):
		if self.select_box:
			if isinstance(self.select_box,list):
				for line in self.select_box:
					line.remove()
			self.select_box = None

################################################################################
# main window widget #
######################

class Window(QWidget):
	def __init__ (self):
		super().__init__()
		self.green_active = True
		self.red_active = True
		self.green_cutoff_active = False
		self.red_cutoff_active = False
		self.threshold_defaults = np.array([180,1000,1000,180,1000,1000])
		self.green_lower = self.threshold_defaults[0]
		self.green_upper = self.threshold_defaults[1]
		self.green_cutoff = self.threshold_defaults[2]
		self.green_max = np.amax([self.threshold_defaults[1],
								  self.threshold_defaults[2]])
		self.red_lower = self.threshold_defaults[3]
		self.red_upper = self.threshold_defaults[4]
		self.red_cutoff = self.threshold_defaults[5]
		self.red_max = np.amax([self.threshold_defaults[4],
								self.threshold_defaults[5]])
		self.x_lower = 0
		self.x_upper = 0
		self.x_size = 512
		self.y_lower = 0
		self.y_upper = 0
		self.y_size = 512
		self.z_level = 0
		self.z_size = 1
		self.z_lower = 0
		self.z_upper = 0
		self.zoomed = False
		self.dapi_image = np.ones((512,512),dtype=int)
		self.green_image = np.zeros((512,512),dtype=int)
		self.red_image = np.zeros((512,512),dtype=int)
		self.nd2_file = None
		self.image_stack = None
		self.title = "ND2 Nuclear Positions Tool"
		self.canvas = MPLCanvas()
		self.toolbar = NavigationToolbar(self.canvas, self)
		self.selecting_area = False
		self.click_id = 0
		self.move_id = 0
		self.position = np.array([0,0])
		self.advanced_defaults = np.array([9,1,4,1,6,4])
		self.neighbourhood_size = self.advanced_defaults[0]
		self.threshold_difference = self.advanced_defaults[1]
		self.minimum_distance = self.advanced_defaults[2]
		self.gauss_deviation = self.advanced_defaults[3]
		self.max_layer_distance = self.advanced_defaults[4]
		self.number_layer_cell = self.advanced_defaults[5]
		self.scale = np.array([0.232, 0.232, 0.479])
		self.dapi_centres = np.zeros((0,2))
		self.green_cells = np.zeros((0,1))
		self.red_cells = np.zeros((0,1))
		self.plot_dapi = True
		#
		self.setupGUI()
	
	def setupGUI (self):
		self.setWindowTitle(self.title)
		# layout for full window
		outer_layout = QVBoxLayout()
		# top section for plot and sliders
		main_layout = QHBoxLayout()
		# main left for plot
		plot_layout = QVBoxLayout()
		plot_layout.addWidget(self.canvas)
		toolbar_layout = QHBoxLayout()
		toolbar_layout.addWidget(self.toolbar)
		toolbar_layout.addWidget(QLabel('Z:'))
		self.textbox_z = QLineEdit()
		self.textbox_z.setMaxLength(4)
		self.textbox_z.setFixedWidth(50)
		self.textbox_z.setText(str(self.z_level))
		self.textbox_z.setValidator(QIntValidator())
		self.textbox_z.editingFinished.connect(self.z_textbox_select)
		toolbar_layout.addWidget(self.textbox_z)
		self.button_z_min = QPushButton()
		self.button_z_min.setText('Set Z Min')
		self.button_z_min.clicked.connect(self.z_min_button)
		toolbar_layout.addWidget(self.button_z_min)
		self.button_z_max = QPushButton()
		self.button_z_max.setText('Set Z Max')
		self.button_z_max.clicked.connect(self.z_max_button)
		toolbar_layout.addWidget(self.button_z_max)
		plot_layout.addLayout(toolbar_layout)
		main_layout.addLayout(plot_layout)
		# main right for options
		options_layout = QHBoxLayout()
		z_select_layout = QVBoxLayout()
		self.slider_z = QSlider(Qt.Vertical)
		self.setup_z_slider()
		self.slider_z.valueChanged.connect(self.z_slider_select)
		z_select_layout.addWidget(self.slider_z)
		options_layout.addLayout(z_select_layout)
		tabs = QTabWidget()
		tabs.setMinimumWidth(180)
		tabs.setMaximumWidth(180)
		# green channel options tab
		tab_green = QWidget()
		tab_green.layout = QVBoxLayout()
		# checkbox to turn off green channel
		self.checkbox_green = QCheckBox("green channel active")
		self.checkbox_green.setChecked(self.green_active)
		self.checkbox_green.stateChanged.connect(self.green_checkbox)
		tab_green.layout.addWidget(self.checkbox_green)
		#checkbox to turn on green cutoff feature
		self.checkbox_green_cutoff = QCheckBox("green cutoff active")
		self.checkbox_green_cutoff.setChecked(self.green_cutoff_active)
		self.checkbox_green_cutoff.stateChanged.connect(
												self.green_cutoff_checkbox)
		tab_green.layout.addWidget(self.checkbox_green_cutoff)
		# sliders for green thresholds
		threshold_layout_green = QHBoxLayout()
		# green min
		threshold_layout_green_min = QVBoxLayout()
		slider_layout_green_min = QHBoxLayout()
		self.slider_green_min = QSlider(Qt.Vertical)
		self.slider_green_min.valueChanged.connect(self.threshold_green_lower)
		slider_layout_green_min.addWidget(self.slider_green_min)
		label_green_min = QLabel('lower')
		label_green_min.setAlignment(Qt.AlignCenter)
		self.textbox_green_min = QLineEdit()
		self.textbox_green_min.setMaxLength(4)
		self.textbox_green_min.setFixedWidth(50)
		self.textbox_green_min.setValidator(QIntValidator())
		self.textbox_green_min.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_green_min.addLayout(slider_layout_green_min)
		threshold_layout_green_min.addWidget(label_green_min)
		threshold_layout_green_min.addWidget(self.textbox_green_min)
		# green max
		threshold_layout_green_max = QVBoxLayout()
		slider_layout_green_max = QHBoxLayout()
		self.slider_green_max = QSlider(Qt.Vertical)
		self.slider_green_max.valueChanged.connect(self.threshold_green_upper)
		slider_layout_green_max.addWidget(self.slider_green_max)
		label_green_max = QLabel('upper')
		label_green_max.setAlignment(Qt.AlignCenter)
		self.textbox_green_max = QLineEdit()
		self.textbox_green_max.setMaxLength(4)
		self.textbox_green_max.setFixedWidth(50)
		self.textbox_green_max.setValidator(QIntValidator())
		self.textbox_green_max.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_green_max.addLayout(slider_layout_green_max)
		threshold_layout_green_max.addWidget(label_green_max)
		threshold_layout_green_max.addWidget(self.textbox_green_max)
		# green cutoff
		threshold_layout_green_cut = QVBoxLayout()
		slider_layout_green_cut = QHBoxLayout()
		self.slider_green_cut = QSlider(Qt.Vertical)
		self.slider_green_cut.valueChanged.connect(self.threshold_green_cutoff)
		slider_layout_green_cut.addWidget(self.slider_green_cut)
		label_green_cut = QLabel('cutoff')
		label_green_cut.setAlignment(Qt.AlignCenter)
		self.textbox_green_cut = QLineEdit()
		self.textbox_green_cut.setMaxLength(4)
		self.textbox_green_cut.setFixedWidth(50)
		self.textbox_green_cut.setValidator(QIntValidator())
		self.textbox_green_cut.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_green_cut.addLayout(slider_layout_green_cut)
		threshold_layout_green_cut.addWidget(label_green_cut)
		threshold_layout_green_cut.addWidget(self.textbox_green_cut)
		#
		threshold_layout_green.addLayout(threshold_layout_green_min)
		threshold_layout_green.addLayout(threshold_layout_green_max)
		threshold_layout_green.addLayout(threshold_layout_green_cut)
		tab_green.layout.addLayout(threshold_layout_green)
		tab_green.setLayout(tab_green.layout)
		tabs.addTab(tab_green, 'green')
		# red channel options tab
		tab_red = QWidget()
		tab_red.layout = QVBoxLayout()
		# checkbox to turn off red channel
		self.checkbox_red = QCheckBox("red channel active")
		self.checkbox_red.setChecked(self.red_active)
		self.checkbox_red.stateChanged.connect(self.red_checkbox)
		tab_red.layout.addWidget(self.checkbox_red)
		#checkbox to turn on red cutoff feature
		self.checkbox_red_cutoff = QCheckBox("red cutoff active")
		self.checkbox_red_cutoff.setChecked(self.red_cutoff_active)
		self.checkbox_red_cutoff.stateChanged.connect(
												self.red_cutoff_checkbox)
		tab_red.layout.addWidget(self.checkbox_red_cutoff)
		# sliders for red thresholds
		threshold_layout_red = QHBoxLayout()
		# red min
		threshold_layout_red_min = QVBoxLayout()
		slider_layout_red_min = QHBoxLayout()
		self.slider_red_min = QSlider(Qt.Vertical)
		self.slider_red_min.valueChanged.connect(self.threshold_red_lower)
		slider_layout_red_min.addWidget(self.slider_red_min)
		label_red_min = QLabel('lower')
		label_red_min.setAlignment(Qt.AlignCenter)
		self.textbox_red_min = QLineEdit()
		self.textbox_red_min.setMaxLength(4)
		self.textbox_red_min.setFixedWidth(50)
		self.textbox_red_min.setValidator(QIntValidator())
		self.textbox_red_min.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_red_min.addLayout(slider_layout_red_min)
		threshold_layout_red_min.addWidget(label_red_min)
		threshold_layout_red_min.addWidget(self.textbox_red_min)
		# red max
		threshold_layout_red_max = QVBoxLayout()
		slider_layout_red_max = QHBoxLayout()
		self.slider_red_max = QSlider(Qt.Vertical)
		self.slider_red_max.valueChanged.connect(self.threshold_red_upper)
		slider_layout_red_max.addWidget(self.slider_red_max)
		label_red_max = QLabel('upper')
		label_red_max.setAlignment(Qt.AlignCenter)
		self.textbox_red_max = QLineEdit()
		self.textbox_red_max.setMaxLength(4)
		self.textbox_red_max.setFixedWidth(50)
		self.textbox_red_max.setValidator(QIntValidator())
		self.textbox_red_max.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_red_max.addLayout(slider_layout_red_max)
		threshold_layout_red_max.addWidget(label_red_max)
		threshold_layout_red_max.addWidget(self.textbox_red_max)
		# red cutoff
		threshold_layout_red_cut = QVBoxLayout()
		slider_layout_red_cut = QHBoxLayout()
		self.slider_red_cut = QSlider(Qt.Vertical)
		self.slider_red_cut.valueChanged.connect(self.threshold_red_cutoff)
		slider_layout_red_cut.addWidget(self.slider_red_cut)
		label_red_cut = QLabel('cutoff')
		label_red_cut.setAlignment(Qt.AlignCenter)
		self.textbox_red_cut = QLineEdit()
		self.textbox_red_cut.setMaxLength(4)
		self.textbox_red_cut.setFixedWidth(50)
		self.textbox_red_cut.setValidator(QIntValidator())
		self.textbox_red_cut.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_red_cut.addLayout(slider_layout_red_cut)
		threshold_layout_red_cut.addWidget(label_red_cut)
		threshold_layout_red_cut.addWidget(self.textbox_red_cut)
		#
		threshold_layout_red.addLayout(threshold_layout_red_min)
		threshold_layout_red.addLayout(threshold_layout_red_max)
		threshold_layout_red.addLayout(threshold_layout_red_cut)
		tab_red.layout.addLayout(threshold_layout_red)
		tab_red.setLayout(tab_red.layout)
		tabs.addTab(tab_red, 'red')
		self.setup_threshold_sliders()
		self.setup_threshold_textboxes()
		options_layout.addWidget(tabs)
		zoom_layout = QVBoxLayout()
		#
		x_min_layout = QHBoxLayout()
		x_min_label = QLabel('X min:')
		x_min_label.setAlignment(Qt.AlignCenter)
		x_min_layout.addWidget(x_min_label)
		self.textbox_x_min = QLineEdit()
		self.textbox_x_min.setMaxLength(4)
		self.textbox_x_min.setFixedWidth(50)
		self.textbox_x_min.setValidator(QIntValidator())
		self.textbox_x_min.editingFinished.connect(self.bound_textbox_select)
		x_min_layout.addWidget(self.textbox_x_min)
		zoom_layout.addLayout(x_min_layout)
		#
		x_max_layout = QHBoxLayout()
		x_max_label = QLabel('X max:')
		x_max_label.setAlignment(Qt.AlignCenter)
		x_max_layout.addWidget(x_max_label)
		self.textbox_x_max = QLineEdit()
		self.textbox_x_max.setMaxLength(4)
		self.textbox_x_max.setFixedWidth(50)
		self.textbox_x_max.setValidator(QIntValidator())
		self.textbox_x_max.editingFinished.connect(self.bound_textbox_select)
		x_max_layout.addWidget(self.textbox_x_max)
		zoom_layout.addLayout(x_max_layout)
		#
		y_min_layout = QHBoxLayout()
		y_min_label = QLabel('Y min:')
		y_min_label.setAlignment(Qt.AlignCenter)
		y_min_layout.addWidget(y_min_label)
		self.textbox_y_min = QLineEdit()
		self.textbox_y_min.setMaxLength(4)
		self.textbox_y_min.setFixedWidth(50)
		self.textbox_y_min.setValidator(QIntValidator())
		self.textbox_y_min.editingFinished.connect(self.bound_textbox_select)
		y_min_layout.addWidget(self.textbox_y_min)
		zoom_layout.addLayout(y_min_layout)
		#
		y_max_layout = QHBoxLayout()
		y_max_label = QLabel('Y max:')
		y_max_label.setAlignment(Qt.AlignCenter)
		y_max_layout.addWidget(y_max_label)
		self.textbox_y_max = QLineEdit()
		self.textbox_y_max.setMaxLength(4)
		self.textbox_y_max.setFixedWidth(50)
		self.textbox_y_max.setValidator(QIntValidator())
		self.textbox_y_max.editingFinished.connect(self.bound_textbox_select)
		y_max_layout.addWidget(self.textbox_y_max)
		zoom_layout.addLayout(y_max_layout)
		#
		z_min_layout = QHBoxLayout()
		z_min_label = QLabel('Z min:')
		z_min_label.setAlignment(Qt.AlignCenter)
		z_min_layout.addWidget(z_min_label)
		self.textbox_z_min = QLineEdit()
		self.textbox_z_min.setMaxLength(4)
		self.textbox_z_min.setFixedWidth(50)
		self.textbox_z_min.setText('0')
		self.textbox_z_min.setValidator(QIntValidator())
		self.textbox_z_min.editingFinished.connect(self.bound_textbox_select)
		z_min_layout.addWidget(self.textbox_z_min)
		zoom_layout.addLayout(z_min_layout)
		#
		z_max_layout = QHBoxLayout()
		z_max_label = QLabel('Z max:')
		z_max_label.setAlignment(Qt.AlignCenter)
		z_max_layout.addWidget(z_max_label)
		self.textbox_z_max = QLineEdit()
		self.textbox_z_max.setMaxLength(4)
		self.textbox_z_max.setFixedWidth(50)
		self.textbox_z_max.setText(str(self.z_size))
		self.textbox_z_max.setValidator(QIntValidator())
		self.textbox_z_max.editingFinished.connect(self.bound_textbox_select)
		z_max_layout.addWidget(self.textbox_z_max)
		zoom_layout.addLayout(z_max_layout)
		#
		self.setup_bound_textboxes()
		#
		self.button_select = QPushButton()
		self.button_select.setText('Select Box')
		self.button_select.clicked.connect(self.select_bounds)
		zoom_layout.addWidget(self.button_select)
		#
		self.button_reset = QPushButton()
		self.button_reset.setText('Select All')
		self.button_reset.clicked.connect(self.reset_bounds)
		zoom_layout.addWidget(self.button_reset)
		#
		self.checkbox_zoom = QCheckBox("zoomed")
		self.checkbox_zoom.setChecked(self.zoomed)
		self.checkbox_zoom.stateChanged.connect(self.zoom_checkbox)
		zoom_layout.addWidget(self.checkbox_zoom)
		#
		self.checkbox_dapi = QCheckBox("plot dapi")
		self.checkbox_dapi.setChecked(self.plot_dapi)
		self.checkbox_dapi.stateChanged.connect(self.dapi_checkbox)
		zoom_layout.addWidget(self.checkbox_dapi)
		#
		options_layout.addLayout(zoom_layout)
		main_layout.addLayout(options_layout)
		# horizontal row of buttons
		buttons_layout = QHBoxLayout()
		#
		self.button_open_nd2 = QPushButton()
		self.button_open_nd2.setText('Open ND2')
		self.button_open_nd2.clicked.connect(self.open_nd2)
		buttons_layout.addWidget(self.button_open_nd2)
		#
		self.button_preview = QPushButton()
		self.button_preview.setText('Preview')
		self.button_preview.clicked.connect(self.preview)
		buttons_layout.addWidget(self.button_preview)
		#
		self.button_execute = QPushButton()
		self.button_execute.setText('Execute')
		self.button_execute.clicked.connect(self.execute)
		buttons_layout.addWidget(self.button_execute)
		#
		self.button_open_csv = QPushButton()
		self.button_open_csv.setText('Open CSV')
		self.button_open_csv.clicked.connect(self.open_csv)
		buttons_layout.addWidget(self.button_open_csv)
		# Layouts for advanced settings boxes
		advanced_layout = QHBoxLayout()
		neighbourhood_label = QLabel('Neighbourhood:')
		neighbourhood_label.setAlignment(Qt.AlignCenter)
		advanced_layout.addWidget(neighbourhood_label)
		self.textbox_neighbourhood = QLineEdit()
		self.textbox_neighbourhood.setMaxLength(3)
		self.textbox_neighbourhood.setFixedWidth(40)
		self.textbox_neighbourhood.setValidator(QIntValidator())
		self.textbox_neighbourhood.setText(str(self.neighbourhood_size))
		self.textbox_neighbourhood.editingFinished.connect(
											self.advanced_textbox_select)
		advanced_layout.addWidget(self.textbox_neighbourhood)
		#
		threshold_label = QLabel('Threshold Diff:')
		threshold_label.setAlignment(Qt.AlignCenter)
		advanced_layout.addWidget(threshold_label)
		self.textbox_threshold = QLineEdit()
		self.textbox_threshold.setMaxLength(3)
		self.textbox_threshold.setFixedWidth(40)
		self.textbox_threshold.setValidator(QIntValidator())
		self.textbox_threshold.setText(str(self.threshold_difference))
		self.textbox_threshold.editingFinished.connect(
												self.advanced_textbox_select)
		advanced_layout.addWidget(self.textbox_threshold)
		#
		distance_label = QLabel('Minimum Dist:')
		distance_label.setAlignment(Qt.AlignCenter)
		advanced_layout.addWidget(distance_label)
		self.textbox_distance = QLineEdit()
		self.textbox_distance.setMaxLength(3)
		self.textbox_distance.setFixedWidth(40)
		self.textbox_distance.setValidator(QIntValidator())
		self.textbox_distance.setText(str(self.minimum_distance))
		self.textbox_distance.editingFinished.connect(
												self.advanced_textbox_select)
		advanced_layout.addWidget(self.textbox_distance)
		#
		guassian_label = QLabel('Gaussian Dev:')
		guassian_label.setAlignment(Qt.AlignCenter)
		advanced_layout.addWidget(guassian_label)
		self.textbox_guassian = QLineEdit()
		self.textbox_guassian.setMaxLength(3)
		self.textbox_guassian.setFixedWidth(40)
		self.textbox_guassian.setValidator(QIntValidator())
		self.textbox_guassian.setText(str(self.gauss_deviation))
		self.textbox_guassian.editingFinished.connect(
												self.advanced_textbox_select)
		advanced_layout.addWidget(self.textbox_guassian)
		#
		layer_dist_label = QLabel('Max Layer Dist:')
		layer_dist_label.setAlignment(Qt.AlignCenter)
		advanced_layout.addWidget(layer_dist_label)
		self.textbox_layer_distance = QLineEdit()
		self.textbox_layer_distance.setMaxLength(3)
		self.textbox_layer_distance.setFixedWidth(40)
		self.textbox_layer_distance.setValidator(QIntValidator())
		self.textbox_layer_distance.setText(str(self.max_layer_distance))
		self.textbox_layer_distance.editingFinished.connect(
												self.advanced_textbox_select)
		advanced_layout.addWidget(self.textbox_layer_distance)
		#
		number_layer_label = QLabel('Min Layer Num:')
		number_layer_label.setAlignment(Qt.AlignCenter)
		advanced_layout.addWidget(number_layer_label)
		self.textbox_layer_number = QLineEdit()
		self.textbox_layer_number.setMaxLength(3)
		self.textbox_layer_number.setFixedWidth(40)
		self.textbox_layer_number.setValidator(QIntValidator())
		self.textbox_layer_number.setText(str(self.number_layer_cell))
		self.textbox_layer_number.editingFinished.connect(
												self.advanced_textbox_select)
		advanced_layout.addWidget(self.textbox_layer_number)
		#
		self.button_advanced_defaults = QPushButton()
		self.button_advanced_defaults.setText('Defaults')
		self.button_advanced_defaults.clicked.connect(self.reset_defaults)
		advanced_layout.addWidget(self.button_advanced_defaults)
		# Nest the inner layouts into the outer layout
		outer_layout.addLayout(main_layout)
		outer_layout.addLayout(buttons_layout)
		self.progress_bar = QProgressBar()
		outer_layout.addWidget(self.progress_bar)
		outer_layout.addLayout(advanced_layout)
		# Set the window's main layout
		self.setLayout(outer_layout)
	
	def replot (self):
		dapi_display = self.dapi_image
		green_display = np.where(self.green_image > self.green_lower,
							np.where(self.green_image < self.green_upper,
										self.green_image, self.green_upper), 0)
		red_display = np.where(self.red_image > self.red_lower,
							np.where(self.red_image < self.red_upper,
									self.red_image, self.red_upper), 0)
		if self.zoomed:
			self.canvas.update_images(
						dapi_display[self.y_lower:self.y_upper,
									 self.x_lower:self.x_upper],
						green_display[self.y_lower:self.y_upper,
									  self.x_lower:self.x_upper],
						red_display[self.y_lower:self.y_upper,
									self.x_lower:self.x_upper],
						show_green = self.green_active,
						show_red = self.red_active,
						box = np.array([[self.x_lower,
										 self.x_upper],
										[self.y_lower,
										 self.y_upper]]),
						show_box = False
					)
			self.canvas.update_centres(self.dapi_centres,
									   self.green_cells,
									   self.red_cells)
		else:
			self.canvas.update_images(
						dapi_display,
						green_display,
						red_display,
						show_green = self.green_active,
						show_red = self.red_active,
						box = np.array([[self.x_lower,
										 self.x_upper],
										[self.y_lower,
										 self.y_upper]]),
						show_box = True
					)
			self.canvas.update_centres(self.dapi_centres + \
								np.array([self.x_lower,self.y_lower]),
									   self.green_cells,
									   self.red_cells)
	
	def setup_z_slider (self):
		self.slider_z.setMinimum(0)
		self.slider_z.setMaximum(self.z_size-1)
		self.slider_z.setSingleStep(1)
		self.slider_z.setValue(0)
	
	def setup_threshold_sliders (self):
		self.slider_green_min.setMinimum(0)
		self.slider_green_min.setMaximum(self.green_max)
		self.slider_green_min.setSingleStep(1)
		self.slider_green_min.setValue(self.green_lower)
		self.slider_green_max.setMinimum(0)
		self.slider_green_max.setMaximum(self.green_max)
		self.slider_green_max.setSingleStep(1)
		self.slider_green_max.setValue(self.green_upper)
		self.slider_green_cut.setMinimum(0)
		self.slider_green_cut.setMaximum(self.green_max)
		self.slider_green_cut.setSingleStep(1)
		self.slider_green_cut.setValue(self.green_cutoff)
		self.slider_red_min.setMinimum(0)
		self.slider_red_min.setMaximum(self.red_max)
		self.slider_red_min.setSingleStep(1)
		self.slider_red_min.setValue(self.red_lower)
		self.slider_red_max.setMinimum(0)
		self.slider_red_max.setMaximum(self.red_max)
		self.slider_red_max.setSingleStep(1)
		self.slider_red_max.setValue(self.red_upper)
		self.slider_red_cut.setMinimum(0)
		self.slider_red_cut.setMaximum(self.red_max)
		self.slider_red_cut.setSingleStep(1)
		self.slider_red_cut.setValue(self.red_cutoff)
	
	def setup_bound_textboxes (self):
		self.textbox_x_min.setText(str(self.x_lower))
		self.textbox_x_max.setText(str(self.x_upper))
		self.textbox_y_min.setText(str(self.y_lower))
		self.textbox_y_max.setText(str(self.y_upper))
		self.textbox_z_min.setText(str(self.z_lower))
		self.textbox_z_max.setText(str(self.z_upper))
	
	def setup_advanced_textboxes (self):
		self.textbox_neighbourhood.setText(str(self.neighbourhood_size))
		self.textbox_threshold.setText(str(self.threshold_difference))
		self.textbox_distance.setText(str(self.minimum_distance))
		self.textbox_guassian.setText(str(self.gauss_deviation))
		self.textbox_layer_distance.setText(str(self.max_layer_distance))
		self.textbox_layer_number.setText(str(self.number_layer_cell))
	
	def setup_threshold_textboxes (self):
		self.textbox_green_min.setText(str(self.green_lower))
		self.textbox_green_max.setText(str(self.green_upper))
		self.textbox_red_min.setText(str(self.red_lower))
		self.textbox_red_max.setText(str(self.red_upper))
	
	def z_min_button (self):
		self.z_lower = self.z_level
		self.setup_bound_textboxes()
	
	def z_max_button (self):
		self.z_upper = self.z_level
		self.setup_bound_textboxes()
	
	def z_textbox_select (self):
		input_z = int(self.textbox_z.text())
		if input_z > 0 and input_z < self.z_size:
			self.z_level = input_z
			self.slider_z.setValue(input_z)
			self.dapi_image, self. green_image, self.red_image = \
											self.extract_image(self.z_level)
			self.replot()
	
	def z_slider_select (self):
		self.z_level = self.slider_z.value()
		self.textbox_z.setText(str(self.z_level))
		self.dapi_image, self. green_image, self.red_image = \
											self.extract_image(self.z_level)
		self.replot()
	
	def threshold_green_lower (self):
		self.green_lower = self.slider_green_min.value()
		self.textbox_green_min.setText(str(self.green_lower))
		self.replot()
	
	def threshold_green_upper (self):
		self.green_upper = self.slider_green_max.value()
		self.textbox_green_max.setText(str(self.green_upper))
		self.replot()
	
	def threshold_green_cutoff (self):
		self.green_cutoff = self.slider_green_cut.value()
		self.textbox_green_cut.setText(str(self.green_cutoff))
		self.replot()
	
	def threshold_red_lower (self):
		self.red_lower = self.slider_red_min.value()
		self.textbox_red_min.setText(str(self.red_lower))
		self.replot()
	
	def threshold_red_upper (self):
		self.red_upper = self.slider_red_max.value()
		self.textbox_red_max.setText(str(self.red_upper))
		self.replot()
	
	def threshold_red_cutoff (self):
		self.red_cutoff = self.slider_red_cut.value()
		self.textbox_red_cut.setText(str(self.red_cutoff))
		self.replot()
	
	def green_checkbox (self):
		self.green_active = self.checkbox_green.isChecked()
		self.replot()
	
	def red_checkbox (self):
		self.red_active = self.checkbox_red.isChecked()
		self.replot()
	
	def green_cutoff_checkbox (self):
		self.green_cutoff_active = self.checkbox_green_cutoff.isChecked()
		self.replot()
	
	def red_cutoff_checkbox (self):
		self.red_cutoff_active = self.checkbox_red_cutoff.isChecked()
		self.replot()
	
	def zoom_checkbox (self):
		self.zoomed = self.checkbox_zoom.isChecked()
		self.replot()
	
	def dapi_checkbox (self):
		self.plot_dapi = self.checkbox_dapi.isChecked()
		self.replot()
	
	def bound_textbox_select (self):
		self.x_lower = int(self.textbox_x_min.text())
		if self.x_lower < 0:
			self.x_lower = 0
		self.x_upper = int(self.textbox_x_max.text())
		if self.x_upper >= self.x_size:
			self.x_upper = self.x_size-1
		if self.x_upper < self.x_lower:
			self.x_upper = self.x_lower
		self.y_lower = int(self.textbox_y_min.text())
		if self.y_lower < 0:
			self.y_lower = 0
		self.y_upper = int(self.textbox_y_max.text())
		if self.y_upper >= self.y_size:
			self.y_upper = self.y_size-1
		if self.y_upper < self.y_lower:
			self.y_upper = self.y_lower
		self.z_lower = int(self.textbox_z_min.text())
		if self.z_lower < 0:
			self.z_lower = 0
		self.z_upper = int(self.textbox_z_max.text())
		if self.z_upper >= self.z_size:
			self.z_upper = self.z_size-1
		if self.z_upper < self.z_lower:
			self.z_upper = self.z_lower
		self.setup_bound_textboxes()
		self.replot()
	
	def threshold_textbox_select (self):
		self.green_lower = int(self.textbox_green_min.text())
		if self.green_lower < 0:
			self.green_lower = 0
		self.green_upper = int(self.textbox_green_max.text())
		if self.green_upper > self.green_max:
			self.green_upper = self.green_max
		self.green_cutoff = int(self.textbox_green_cut.text())
		if self.green_cutoff > self.green_max:
			self.green_cutoff = self.green_max
		self.red_lower = int(self.textbox_red_min.text())
		if self.red_lower < 0:
			self.red_lower = 0
		self.red_upper = int(self.textbox_red_max.text())
		if self.red_upper > self.red_max:
			self.red_upper = self.red_max
		self.red_cutoff = int(self.textbox_red_cut.text())
		if self.red_cutoff > self.red_max:
			self.red_cutoff = self.red_max
		self.setup_threshold_textboxes()
		self.setup_threshold_sliders()
	
	def advanced_textbox_select (self):
		self.neighbourhood_size = int(self.textbox_neighbourhood.text())
		self.threshold_difference = int(self.textbox_threshold.text())
		self.minimum_distance = int(self.textbox_distance.text())
		self.gauss_deviation = int(self.textbox_guassian.text())
		self.max_layer_distance = int(self.textbox_layer_distance.text())
		self.number_layer_cell = int(self.textbox_layer_number.text())
	
	def reset_defaults (self):
		self.neighbourhood_size = self.advanced_defaults[0]
		self.threshold_difference = self.advanced_defaults[1]
		self.minimum_distance = self.advanced_defaults[2]
		self.gauss_deviation = self.advanced_defaults[3]
		self.max_layer_distance = self.advanced_defaults[4]
		self.number_layer_cell = self.advanced_defaults[5]
		self.setup_advanced_textboxes()
		self.green_lower = self.threshold_defaults[0]
		self.green_upper = self.threshold_defaults[1]
		self.green_cutoff = self.threshold_defaults[2]
		self.red_lower = self.threshold_defaults[3]
		self.red_upper = self.threshold_defaults[4]
		self.red_cutoff = self.threshold_defaults[5]
		self.setup_threshold_textboxes()
		self.setup_threshold_sliders()
		self.checkbox_green_cutoff.setChecked(False)
		self.green_cutoff_active = False
		self.checkbox_red_cutoff.setChecked(False)
		self.red_cutoff_active = False
	
	def select_bounds (self):
		self.zoomed = False
		self.checkbox_zoom.setChecked(False)
		self.replot()
		self.dapi_centres = np.zeros((0,2))
		self.green_cells = np.zeros((0))
		self.red_cells = np.zeros((0))
		self.selecting_area = True
		self.click_id = self.canvas.mpl_connect(
							'button_press_event', self.on_click)
	
	def on_click (self, event):
		if self.selecting_area:
			self.position = np.array([int(np.floor(event.xdata)),
									  int(np.floor(event.ydata))])
			self.canvas.mpl_disconnect(self.click_id)
			self.click_id = self.canvas.mpl_connect(
								'button_release_event', self.off_click)
			self.move_id = self.canvas.mpl_connect(
								'motion_notify_event', self.mouse_moved)
	
	def mouse_moved (self, event):
		if self.selecting_area:
			p_1 = np.array([int(np.floor(event.xdata)),
							int(np.floor(event.ydata))])
			p_2 = self.position
			self.canvas.plot_selector(p_1, p_2)
	
	def off_click (self, event):
		if self.selecting_area:
			p_1 = np.array([int(np.floor(event.xdata)),
							int(np.floor(event.ydata))])
			p_2 = self.position
			self.x_lower = np.amin(np.array([p_1[0], p_2[0]]))
			self.x_upper = np.amax(np.array([p_1[0], p_2[0]]))
			self.y_lower = np.amin(np.array([p_1[1], p_2[1]]))
			self.y_upper = np.amax(np.array([p_1[1], p_2[1]]))
			self.canvas.mpl_disconnect(self.click_id)
			self.canvas.mpl_disconnect(self.move_id)
			self.canvas.remove_selector()
			self.selecting_area = False
			self.setup_bound_textboxes()
			self.bound_textbox_select()
	
	def reset_bounds (self):
		self.x_lower = 0
		self.x_upper = self.x_size
		self.y_lower = 0
		self.y_upper = self.y_size
		self.setup_bound_textboxes()
		self.replot()
	
	def open_nd2 (self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getOpenFileName(self,
								"Open ND2 File",
								"",
								"ND2 Files (*.nd2);;All Files (*)",
								options=options)
		if file_name == '':
			return
		else:
			self.nd2_file = Path(file_name)
		try:
			self.image_stack = ND2Reader(str(self.nd2_file))
			self.x_size = self.image_stack.sizes['x']
			self.y_size = self.image_stack.sizes['y']
			self.z_size = self.image_stack.sizes['z']
			self.x_lower = 0
			self.x_upper = self.x_size-1
			self.y_lower = 0
			self.y_upper = self.y_size-1
			self.z_lower = 0
			self.z_upper = self.z_size-1
			self.setup_bound_textboxes()
			self.scale[0] = self.image_stack.metadata['pixel_microns']
			self.scale[1] = self.image_stack.metadata['pixel_microns']
			self.scale[2] = self.image_stack.metadata['z_coordinates'][1] - \
							self.image_stack.metadata['z_coordinates'][0]
		except:
			self.nd2_file = None
			msg = QMessageBox()
			msg.setIcon(QMessageBox.Critical)
			msg.setText("Error")
			msg.setInformativeText('Could not open file!')
			msg.setWindowTitle("Error")
			msg.exec_()
			return
		self.dapi_image, self. green_image, self.red_image = \
											self.extract_image(self.z_level)
		self.setup_z_slider()
	#	self.setup_threshold_sliders()
		self.replot()
	
	def preview (self):
		if self.nd2_file == None or self.nd2_file == '':
			return
		if self.green_active:
			green_image = self.green_image
		else:
			green_image = None
		if self.red_active:
			red_image = self.red_image
		else:
			red_image = None
		self.dapi_centres, self.green_cells, self.red_cells = \
					self.process_image(self.dapi_image,
										green_image, red_image)
		self.replot()
	
	def execute (self):
		if self.nd2_file == None or self.nd2_file == '':
			return
		if self.z_upper <= self.z_lower:
			return
		positions_layer = np.zeros((0,3), dtype = float)
		green_cells_layer = np.zeros(0, dtype = bool)
		red_cells_layer = np.zeros(0, dtype = bool)
		self.progress_bar.setRange(self.z_lower, self.z_upper)
		for z_level in range(self.z_lower, self.z_upper+1):
			dapi_image, green_image, red_image = self.extract_image(z_level)
			if not self.green_active:
				green_image = None
			if not self.red_active:
				red_image = None
			dapi_centres, green_cells, red_cells = self.process_image(
											dapi_image, green_image, red_image)
			positions_layer = np.vstack([positions_layer,
				np.vstack([(dapi_centres + np.array([self.x_lower,
													 self.y_lower])).T,
						np.ones(dapi_centres.shape[0])*z_level]).T])
			green_cells_layer = np.append(green_cells_layer, green_cells)
			red_cells_layer = np.append(red_cells_layer, red_cells)
			self.progress_bar.setValue(z_level)
		self.progress_bar.reset()
		self.progress_bar.setMinimum(0)
		positions_layer_size = positions_layer.shape[0]
		self.progress_bar.setMaximum(positions_layer.shape[0])
		positions = np.zeros((0,3), dtype = float)
		green_cells = np.zeros(0, dtype = bool)
		red_cells = np.zeros(0, dtype = bool)
		while positions_layer.shape[0] > 0:
			x_0, y_0, z_0 = positions_layer[0]
			found_on_next_layer = True
			layer_index = 1
			positions_temp = np.zeros((0,3), dtype = float)
			green_cells_temp = np.zeros(0, dtype = bool)
			red_cells_temp = np.zeros(0, dtype = bool)
			while found_on_next_layer and \
				  layer_index < positions_layer.shape[0]:
				found_on_next_layer = False
				# find next layer
				on_layer = (positions_layer[:,2] == z_0+1)
				layer_index = np.argmax(on_layer)
				if layer_index > 0 and \
				   layer_index < positions_layer.shape[0]:
					z_0 = positions_layer[layer_index,2]
					distances = np.linalg.norm(
										positions_layer[on_layer,:2] - \
											np.array([x_0,y_0]), axis=1)
					local_index = np.argmin(distances)
					if distances[local_index] < self.minimum_distance:
						found_on_next_layer = True
						index = layer_index + local_index
						positions_temp = np.vstack([positions_temp,
													positions_layer[index]])
						x_0 = (x_0 * positions_temp.shape[0] + \
								positions_temp[-1,0]) / \
									(positions_temp.shape[0]+1)
						y_0 = (y_0 * positions_temp.shape[0] + \
								positions_temp[-1,1]) / \
									(positions_temp.shape[0]+1)
						positions_layer = np.delete(positions_layer, index,
														axis=0)
						green_cells_temp = np.append(green_cells_temp,
													green_cells_layer[index])
						green_cells_layer = np.delete(green_cells_layer,
															index, axis=0)
						red_cells_temp = np.append(red_cells_temp,
													red_cells_layer[index])
						red_cells_layer = np.delete(red_cells_layer,
															index, axis=0)
			if positions_temp.shape[0] >= self.number_layer_cell:
				positions = np.vstack([positions, np.mean(positions_temp,
															axis=0)])
				green_cells = np.append(green_cells,
										np.count_nonzero(green_cells_temp) >= \
													green_cells_temp.shape[0]/2)
				red_cells = np.append(red_cells,
										np.count_nonzero(red_cells_temp) >= \
													red_cells_temp.shape[0]/2)
			positions_layer = np.delete(positions_layer, 0, axis=0)
			green_cells_layer = np.delete(green_cells_layer,0)
			red_cells_layer = np.delete(red_cells_layer,0)
			self.progress_bar.setValue(positions_layer_size - \
									   positions_layer.shape[0])
		positions = positions * self.scale
		self.save_csv(positions, green_cells, red_cells)
		self.plot_3d(positions, green_cells, red_cells)
	
	def save_csv (self, positions, green_cells, red_cells):
		output_array = np.vstack([positions.T, green_cells, red_cells]).T
		data_format = '%.18e', '%.18e', '%.18e', '%1d', '%1d'
		np.savetxt(self.nd2_file.with_suffix(
				'.{0:s}.csv'.format(time.strftime("%Y.%m.%d-%H.%M.%S"))),
				output_array, fmt = data_format, delimiter = ',')
	
	def open_csv (self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getOpenFileName(self,
								"Open CSV File",
								"",
								"CSV Files (*.csv);;All Files (*)",
								options=options)
		if file_name == '':
			return
		else:
			csv_file = Path(file_name)
		try:
			data_format = np.dtype([ ('positions', float, 3),
									 ('green_cells', bool),
									 ('red_cells', bool) ])
			input_data = np.loadtxt(str(csv_file), dtype = data_format,
									delimiter=',').view(np.recarray)
			self.plot_3d(input_data.positions, input_data.green_cells,
											   input_data.red_cells)
		except:
			msg = QMessageBox()
			msg.setIcon(QMessageBox.Critical)
			msg.setText("Error")
			msg.setInformativeText('Could not open file!')
			msg.setWindowTitle("Error")
			msg.exec_()
			return
	
	def extract_image (self, z_value):
		dapi_image = np.zeros((0,2))
		green_image = np.zeros((0,2))
		red_image = np.zeros((0,2))
		try:
			channels = self.image_stack.metadata['channels']
			for index, channel in enumerate(channels):
				if channel == 'DAPI':
					dapi_image = self.image_stack.get_frame_2D(
										c = index,
										z = z_value)
				elif channel == 'Green':
					green_image = self.image_stack.get_frame_2D(
										c = index,
										z = z_value)
				elif channel == 'Red':
					red_image = self.image_stack.get_frame_2D(
										c = index,
										z = z_value)
		except:
			self.nd2_file = None
			msg = QMessageBox()
			msg.setIcon(QMessageBox.Critical)
			msg.setText("Error")
			msg.setInformativeText('Problem extracting data!')
			msg.setWindowTitle("Error")
			msg.exec_()
		return dapi_image, green_image, red_image
	
	def process_image (self, dapi_image, green_image = None,
										red_image = None):
		dapi_centres = self.find_centres(dapi_image)
		delta = self.neighbourhood_size # int(self.neighbourhood_size/2)
		green_cells = np.zeros(dapi_centres.shape[0], dtype = bool)
		if green_image is not None:
			green_blur = green_image[self.y_lower:self.y_upper,
									 self.x_lower:self.x_upper]
			green_blur = mh.gaussian_filter(green_blur, self.gauss_deviation)
		#	green_blur = np.where(green_blur > self.green_lower,
		#					np.where(green_blur < self.green_upper,
		#								green_blur, self.green_upper), 0)
			green_blur = np.where(green_blur < self.green_upper,
										green_blur, self.green_upper)
		red_cells = np.zeros(dapi_centres.shape[0], dtype = bool)
		if red_image is not None:
			red_blur = red_image[self.y_lower:self.y_upper,
								 self.x_lower:self.x_upper]
			red_blur = mh.gaussian_filter(red_blur, self.gauss_deviation)
		#	red_blur = np.where(red_blur > self.red_lower,
		#					np.where(red_blur < self.red_upper,
		#								red_blur, self.red_upper), 0)
			red_blur = np.where(red_blur < self.red_upper,
										red_blur, self.red_upper)
		for index,(c_x,c_y) in enumerate(dapi_centres):
			# median seems to work better than mean.
			if green_image is not None:
				if np.median(green_blur[c_y-delta:c_y+delta, # mean ?
									  c_x-delta:c_x+delta]) > self.green_lower:
					green_cells[index] = True
			if red_image is not None:
				if np.median(red_blur[c_y-delta:c_y+delta, # mean ?
									c_x-delta:c_x+delta]) > self.red_lower:
					red_cells[index] = True
			if red_image is not None and green_image is not None:
				if self.green_cutoff_active and \
				   np.median(green_blur[c_y-delta:c_y+delta, # mean ?
								c_x-delta:c_x+delta]) > self.green_cutoff:
					red_cells[index] = False
				if self.red_cutoff_active and \
				   np.median(red_blur[c_y-delta:c_y+delta, # mean ?
								c_x-delta:c_x+delta]) > self.red_cutoff:
					green_cells[index] = False
		return dapi_centres, green_cells, red_cells
	
	def find_centres (self, image):
		frame = image[self.y_lower:self.y_upper,
					  self.x_lower:self.x_upper]
		frame = mh.gaussian_filter(frame, self.gauss_deviation)
		frame_max = filters.maximum_filter(frame, self.neighbourhood_size)
		maxima = (frame == frame_max)
		frame_min = filters.minimum_filter(frame, self.neighbourhood_size)
		differences = ((frame_max - frame_min) > self.threshold_difference)
		maxima[differences == 0] = 0
		maximum = np.amax(frame)
		minimum = np.amin(frame)
		outside_filter = (frame_max > (maximum-minimum)*0.1 + minimum)
		maxima[outside_filter == 0] = 0
		labeled, num_objects = ndimage.label(maxima)
		slices = ndimage.find_objects(labeled)
		centres = np.zeros((len(slices),2), dtype = int)
		good_centres = 0
		for (dy,dx) in slices:
			centres[good_centres,0] = int((dx.start + dx.stop - 1)/2)
			centres[good_centres,1] = int((dy.start + dy.stop - 1)/2)
			if centres[good_centres,0] < self.neighbourhood_size/2 or \
			   centres[good_centres,0] > (self.x_upper-self.x_lower) - \
			   								self.neighbourhood_size/2 or \
			   centres[good_centres,1] < self.neighbourhood_size/2 or \
			   centres[good_centres,1] > (self.y_upper-self.y_lower) - \
			   								self.neighbourhood_size/2:
				good_centres -= 1
			good_centres += 1
		centres = centres[:good_centres]
		return centres
	
	def plot_3d (self, positions, green_cells, red_cells):
		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(111, projection='3d')
		if self.plot_dapi:
			ax.plot(positions[:,0], positions[:,1], positions[:,2],
					linestyle = '', marker = '.', color = 'gray')
		if self.red_active and self.green_active:
			ax.plot(positions[np.logical_and(red_cells,
								np.logical_not(green_cells)),0],
					positions[np.logical_and(red_cells,
								np.logical_not(green_cells)),1],
					positions[np.logical_and(red_cells,
								np.logical_not(green_cells)),2],
					linestyle = '', marker = '+', color = 'red')
			ax.plot(positions[np.logical_and(red_cells, green_cells),0],
					positions[np.logical_and(red_cells, green_cells),1],
					positions[np.logical_and(red_cells, green_cells),2],
					linestyle = '', marker = '+', color = 'purple')
			ax.plot(positions[np.logical_and(green_cells,
								np.logical_not(red_cells)),0],
					positions[np.logical_and(green_cells,
								np.logical_not(red_cells)),1],
					positions[np.logical_and(green_cells,
								np.logical_not(red_cells)),2],
						linestyle = '', marker = 'x', color = 'green')
		elif self.red_active:
				ax.plot(positions[red_cells,0], positions[red_cells,1],
								positions[red_cells,2],
						linestyle = '', marker = '+', color = 'red')
		elif self.green_active:
				ax.plot(positions[green_cells,0], positions[green_cells,1],
								positions[green_cells,2],
						linestyle = '', marker = 'x', color = 'green')
		ax.set_xlim([ np.amin(positions[:,0]), np.amax(positions[:,0]) ])
		ax.set_ylim([ np.amin(positions[:,1]), np.amax(positions[:,1]) ])
		ax.set_zlim([ np.amin(positions[:,2]), np.amax(positions[:,2]) ])
		ax.set_aspect('auto')
		ax.set_box_aspect((np.amax(positions[:,0]) - np.amin(positions[:,0]),
						   np.amax(positions[:,1]) - np.amin(positions[:,1]),
						   np.amax(positions[:,2]) - np.amin(positions[:,2])))
		plt.show()

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = Window()
	window.show()
	sys.exit(app.exec_())
