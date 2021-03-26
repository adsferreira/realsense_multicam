from datetime import datetime
from enum import IntEnum
import os.path

import PyQt5
from PyQt5.QtGui import QImage, QPixmap
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

from camera_realsense import RealSenseCamera

from PyQt5 import QtCore, Qt, QtGui
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QPushButton, QLineEdit, QLabel, QGridLayout, QWidget, QMainWindow, QTabWidget, \
    QFrame, QApplication, QVBoxLayout, QHBoxLayout, QMessageBox
from QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import numpy as np
import pyrealsense2 as rs
import sys
import vtk
import vtk.util.numpy_support as vtk_np


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def np_color_img_to_q_image(color_image):
    h, w, channel = color_image.shape
    q_image = QImage(color_image.data, w, h, channel * w, QImage.Format_RGB888)

    return q_image


# access to camera
# real_sense_cam = RealSenseCamera((640, 480), 15)
# real_sense_cam.config()
# real_sense_cam.start_pipeline()

# Use the device manager class to enable the devices and get the frames
# device_manager = DeviceManager(rs.context(), real_sense_cam.get_config())
# device_manager.enable_all_devices()
ctx = rs.context()
devices = ctx.query_devices()
nr_devices = len(devices)
serial_numbers = []
pipelines = []
depth_scales = []
visualizers = []
align_to = rs.stream.color
align = rs.align(align_to)
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

for device in devices:
    print("current device: ", device)
    pipeline = rs.pipeline(ctx)
    current_serial_number = device.get_info(rs.camera_info.serial_number)
    config.enable_device(current_serial_number)
    pipeline.start(config)
    # add current pipeline to the pipeline collection
    pipelines.append(pipeline)
    serial_numbers.append(current_serial_number)

flip_transform = [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]


class MainWindow(QMainWindow):
    def __init__(self, parent=None, path_prefix=None, farm_name=None, technician_name=None):
        QMainWindow.__init__(self, parent)
        date_str = datetime.now().strftime("%Y-%m-%d")
        path = path_prefix + "/" + farm_name + "/" + date_str + "/" + technician_name + "/"

        self.__counter = 0
        self.frame = QFrame()

        self.tabs_widget = CamTabsWidget(self, path)
        self.setCentralWidget(self.tabs_widget)

        self.setWindowTitle("SmartUS Image Collector")

        self.show()
        self.tabs_widget.get_txt_box_2d_idx().setFocus()
        self.tabs_widget.get_txt_box_3d_idx().setFocus()
        self.tabs_widget.get_iren().Initialize()
        self.tabs_widget.get_iren().Start()
        # self.tabs_widget.frames_ready.connect(self.tabs_widget.receive_frame)


class CamTabsWidget(QWidget):
    frames_ready = pyqtSignal(QImage, QImage, QImage)

    def __init__(self, parent, path):
        super(QWidget, self).__init__(parent)
        self.__path = path
        self.__counter = 0
        self.layout = QVBoxLayout(self)
        # initialize tab screen
        self.__tabs = QTabWidget()
        self.__tabs.blockSignals(True)  # just for not showing the initial message
        self.__tabs.currentChanged.connect(self.on_change)

        self.__tab3D = QWidget()
        self.__tab2D = QWidget()
        # add tabs
        self.__tabs.addTab(self.__tab3D, "3D")
        self.__tabs.addTab(self.__tab2D, "2D")
        # create the content and layout of the tab with 3D cameras
        self.__set_tab3d_layout()
        # create the content and layout of the tab with 2D cameras
        self.__set_tab2d_layout()
        # set all objects that support the pipeline to create 3D images
        self.__set_3d_supporting_objects()
        # capture the cameras' frames to create a movie
        self.layout.addWidget(self.__tabs)
        self.__run_movie()
        self.__tabs.blockSignals(False)

    def __set_3d_supporting_objects(self):
        self.pd_collection = []
        self.mapper_collection = []
        self.actor_collection = []
        self.np_array = []
        self.cells_npy = []
        self.timer_count = 0
        self._n_coordinates = 0
        self.align = rs.align(rs.stream.color)
        self.__iren = self.__vtkWidget.GetRenderWindow().GetInteractor()
        self.__iren.GetInteractorStyle().SetCurrentStyleToTrackballActor()
        self.__iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.timer_event)
        self.view_coordinates = [[0., .5, .5, 1.], [.5, .5, 1., 1.], [0., 0., .5, .5], [.5, 0., 1., .5]]

    def __set_tab3d_layout(self):
        frame = QFrame()
        # add a vtk-based window for interaction
        self.__vtkWidget = QVTKRenderWindowInteractor(frame)
        # add a label and a text box for input of the object id
        self.__lbl_3d_idx = QLabel('ID:', self)
        self.__txt_box_3d_idx = QLineEdit(self)
        # add a button to save 3D images
        self.__btn_save_3d_img = QPushButton('Save Image', self)
        self.__btn_save_3d_img.setToolTip('Save 3D image to the file.')
        self.__btn_save_3d_img.clicked.connect(self.save_3d_image)
        # set the layout of the tab which holds the 3D cameras
        self.__tab3D.setLayout(self.__create_3d_cams_grid_layout())

    def __set_tab2d_layout(self):
        self.__lbl_2d_cam_1 = QLabel()
        self.__lbl_2d_cam_2 = QLabel()
        self.__lbl_2d_cam_3 = QLabel()
        self.__lbl_2d_cam_4 = QLabel()
        # add a label and a text box for input of the object id
        self.__lbl_2d_idx = QLabel('ID:', self)
        self.__txt_box_2d_idx = QLineEdit(self)
        # add a button to save 3D images
        self.__btn_save_2d_img = QPushButton('Save Image', self)
        self.__btn_save_2d_img.setToolTip('Save 2D image to the file.')
        self.__btn_save_2d_img.clicked.connect(self.save_2d_image)
        # set the layout of the tab which holds the 2D cameras
        self.__tab2D.setLayout(self.__create_2d_cams_grid_layout())

    def get_txt_box_2d_idx(self):
        return self.__txt_box_2d_idx

    def get_txt_box_3d_idx(self):
        return self.__txt_box_3d_idx

    def __run_movie(self):
        cam_counter = 0

        for pipe in pipelines:
            frame_set = pipe.wait_for_frames()

            # Wait for a coherent color frame
            # frames = None  # real_sense_cam.get_pipeline().wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frame_set)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            color_image = color_image.reshape((color_image.shape[0] * color_image.shape[1], 3))

            # self._colors.SetNumberOfTuples(color_image.shape[0])
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            # colors.SetName("Colors")

            current_pd = vtk.vtkPolyData()
            self.pd_collection.append(current_pd)
            colors.SetArray(vtk_np.numpy_to_vtk(color_image), color_image.shape[0] * color_image.shape[1], 1)
            current_pd.GetPointData().SetScalars(colors)

            pc = rs.pointcloud()
            point_cloud = pc.calculate(depth_frame)
            pc.map_to(color_frame)
            v, t = point_cloud.get_vertices(), point_cloud.get_texture_coordinates()
            vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz

            self._n_coordinates = vertices.shape[0]

            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()

            points.SetData(vtk_np.numpy_to_vtk(vertices))
            cells_npy = np.vstack([np.ones(self._n_coordinates, dtype=np.int64),
                                   np.arange(self._n_coordinates, dtype=np.int64)]).T.flatten()
            cells.SetCells(self._n_coordinates, vtk_np.numpy_to_vtkIdTypeArray(cells_npy))
            self.pd_collection[cam_counter].SetPoints(points)
            self.pd_collection[cam_counter].SetVerts(cells)

            mapper = vtk.vtkPolyDataMapper()
            self.mapper_collection.append(mapper)
            self.mapper_collection[cam_counter].SetInputData(self.pd_collection[cam_counter])

            transform = vtk.vtkTransform()
            transform.SetMatrix(flip_transform)

            actor = vtk.vtkActor()
            self.actor_collection.append(actor)
            self.actor_collection[cam_counter].SetMapper(self.mapper_collection[cam_counter])
            self.actor_collection[cam_counter].GetProperty().SetRepresentationToPoints()
            self.actor_collection[cam_counter].SetUserTransform(transform)

            current_ren = vtk.vtkRenderer()
            current_ren.GetActiveCamera()

            # set viewports if the number of cams ara greater than one
            if len(pipelines) > 1:
                current_ren.SetViewport(self.view_coordinates[cam_counter])
            current_ren.AddActor(self.actor_collection[cam_counter])
            self.__vtkWidget.GetRenderWindow().AddRenderer(current_ren)
            cam_counter += 1

        self.__iren.AddObserver('TimerEvent', self.update_poly_data)

        dt = 30  # ms
        self.__iren.CreateRepeatingTimer(dt)

    def update_poly_data(self, obj=None, event=None):
        cam_counter = 0
        rgb_cam_images = []

        for pipe in pipelines:
            frame_set = pipe.wait_for_frames()
            # Wait for a coherent color frame
            # frames = real_sense_cam.get_pipeline().wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frame_set)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            pc = rs.pointcloud()

            point_cloud = pc.calculate(depth_frame)
            pc.map_to(color_frame)
            v, t = point_cloud.get_vertices(), point_cloud.get_texture_coordinates()
            vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz

            color_image = np.asanyarray(color_frame.get_data())
            rgb_cam_images.append(np_color_img_to_q_image(color_image))
            color_image = color_image.reshape((color_image.shape[0] * color_image.shape[1], 3))

            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)

            colors.SetArray(vtk_np.numpy_to_vtk(color_image), color_image.shape[0] * color_image.shape[1], 1)
            self.pd_collection[cam_counter].GetPointData().SetScalars(colors)

            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()

            points.SetData(vtk_np.numpy_to_vtk(vertices))
            cells_npy = np.vstack([np.ones(self._n_coordinates, dtype=np.int64),
                                   np.arange(self._n_coordinates, dtype=np.int64)]).T.flatten()
            cells.SetCells(self._n_coordinates, vtk_np.numpy_to_vtkIdTypeArray(cells_npy))

            self.pd_collection[cam_counter].SetPoints(points)
            self.pd_collection[cam_counter].SetVerts(cells)
            self.pd_collection[cam_counter].Modified()
            cam_counter += 1

        self.frames_ready.emit(rgb_cam_images[0],
                               rgb_cam_images[1],
                               rgb_cam_images[2])

        self.__iren.GetRenderWindow().Render()
        # print(self.timer_count)
        self.timer_count += 1

    def __create_2d_cams_grid_layout(self):
        layout = QGridLayout()
        layout_idx = QHBoxLayout()
        layout_idx.addWidget(self.__lbl_2d_idx)
        layout_idx.addWidget(self.__txt_box_2d_idx)
        # add all widgets to the layout
        # add four labels of images, one for each camera
        layout.addWidget(self.__lbl_2d_cam_1, 0, 0, 1, 1)
        layout.addWidget(self.__lbl_2d_cam_2, 0, 1, 1, 1)
        layout.addWidget(self.__lbl_2d_cam_3, 1, 0, 1, 1)
        layout.addWidget(self.__lbl_2d_cam_4, 1, 1, 1, 1)
        # add a label and a text box to input the object id
        # layout.addWidget(self.__lbl_2d_idx, 2, 0, 1, 1)
        # layout.addWidget(self.__txt_box_2d_idx, 2, 1, 1, 1)
        layout.addLayout(layout_idx, 2, 0, 1, 2)
        # add a save button
        layout.addWidget(self.__btn_save_2d_img, 3, 0, 1, 2)
        # could we create a shared button to save all images at once?

        return layout

    def __create_3d_cams_grid_layout(self):
        layout = QGridLayout()

        layout.addWidget(self.__vtkWidget, 0, 0, 1, 2)
        # add a label and a text box to input the object id
        layout.addWidget(self.__lbl_3d_idx, 1, 0, 1, 1)
        layout.addWidget(self.__txt_box_3d_idx, 1, 1, 1, 1)
        # add a save button
        layout.addWidget(self.__btn_save_3d_img, 2, 0, 1, 2)

        return layout

    def timer_event(self):
        self.__iren.TimerEvent()

    def create_timer(self, obj, evt):
        self._timer.start(0)

    def destroy_timer(self, obj, evt):
        self._timer.stop()
        return 1

    def get_iren(self):
        return self.__iren

    @QtCore.pyqtSlot(QImage, QImage, QImage)
    def receive_frame(self, rgb_1, rgb_2, rgb_3):
        ratio = 1.
        p_cam1 = QPixmap.fromImage(rgb_1)
        p_cam2 = QPixmap.fromImage(rgb_2)
        p_cam3 = QPixmap.fromImage(rgb_3)
        # p_cam4 = QPixmap.fromImage(rgb_4)
        w_cam1, h_cam1 = p_cam1.width() * ratio, p_cam1.height() * ratio
        w_cam2, h_cam2 = p_cam1.width() * ratio, p_cam1.height() * ratio
        w_cam3, h_cam3 = p_cam1.width() * ratio, p_cam1.height() * ratio
        # w_cam4 = self.__lbl_2d_cam_1.width()
        # h_cam4 = self.__lbl_2d_cam_1.height()
        self.__lbl_2d_cam_1.setPixmap(p_cam1.scaled(h_cam1, w_cam1, PyQt5.QtCore.Qt.KeepAspectRatio))
        self.__lbl_2d_cam_2.setPixmap(p_cam2.scaled(h_cam1, w_cam1, PyQt5.QtCore.Qt.KeepAspectRatio))
        self.__lbl_2d_cam_3.setPixmap(p_cam3.scaled(h_cam1, w_cam1, PyQt5.QtCore.Qt.KeepAspectRatio))
        # self.__lbl_2d_cam_4.setPixmap(QPixmap.fromImage(rgb_4))

    def on_change(self, i):
        if i == 0:  # 3d tab
            self.__txt_box_3d_idx.setText(self.__txt_box_2d_idx.text())
        else:
            self.__txt_box_2d_idx.setText(self.__txt_box_3d_idx.text())
        #  QMessageBox.information(self,
        #                          "Tab Index Changed!",
        #                          "Current Tab Index: %d" % i)  # changed!

    def save_2d_image(self):
        cam_1_pixmap = self.__lbl_2d_cam_1.pixmap()
        cam_2_pixmap = self.__lbl_2d_cam_2.pixmap()
        cam_3_pixmap = self.__lbl_2d_cam_3.pixmap()
        # cam_4_pixmap = self.__lbl_2d_cam_4.pixmap()

        # set the correct path to save the images files
        cam_1_pixmap.save("")
        cam_2_pixmap.save("")
        cam_3_pixmap.save("")
        # cam_4_pixmap.save("")

    def save_3d_image(self):
        for i in range(nr_devices):
            writer = vtkXMLPolyDataWriter()
            path_cam = self.__path + serial_numbers[i] + "/3D/"
            path_exists = os.path.isdir(path_cam)
            today_str = datetime.now().strftime("%H:%M:%S.%f")

            if not path_exists:
                os.makedirs(path_cam)

            f_name = path_cam + self.__txt_box_3d_idx.text() + "_img_" + str(self.__counter) + "_" + today_str + ".vtp"

            writer.SetFileName(f_name)
            writer.SetInputData(self.pd_collection[i])
            writer.Write()

        self.__counter += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    prefix = "/home/adriano/PycharmProjects/realsense_multicam/pics"
    farm = "FazendaCamparino"
    technician = "Adriano"
    window = MainWindow(path_prefix=prefix, farm_name=farm, technician_name=technician)
    window.tabs_widget.frames_ready.connect(window.tabs_widget.receive_frame)
    sys.exit(app.exec_())

# TODO: add depth cam
# TODO: decouple objects: create a devices manager, a vtk service object, etc.
# TODO: create a first window to set the files' prefix, farms' name, technicians' name, etc.
# TODO: try to implement a possible change where threads hold the cams' stream and visualization.
# TODO: verify why it is always necessary to downgrade pyqt5: pip3 install PyQt5==5.9.2
