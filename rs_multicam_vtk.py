from datetime import datetime
from enum import IntEnum

from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

from camera_realsense import RealSenseCamera

from PyQt5 import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QPushButton, QLineEdit, QLabel, QGridLayout
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
    config.enable_device(device.get_info(rs.camera_info.serial_number))
    pipeline.start(config)
    # add current pipeline to the pipeline collection
    pipelines.append(pipeline)

flip_transform = [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]


class MainWindow(Qt.QMainWindow):
    def __init__(self, parent=None):
        Qt.QMainWindow.__init__(self, parent)
        self.__counter = 0
        self.__prefix = "/home/adriano/PycharmProjects/realsense_multicam/pics/"
        self.frame = Qt.QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.pd_collection = []
        self.mapper_collection = []
        self.actor_collection = []
        self.np_array = []
        self.cells_npy = []
        self.timer_count = 0
        self._n_coordinates = 0
        self.align = rs.align(rs.stream.color)
        self._iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self._iren.GetInteractorStyle().SetCurrentStyleToTrackballActor()
        self._iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
        print(type(self._iren.GetInteractorStyle()))
        self._timer = QTimer(self)
        self.__label_idx = QLabel('ID:', self)
        self.__txt_box_idx = QLineEdit(self)
        self.__button = QPushButton('Save Image', self)
        self.__button.setToolTip('Save 3D image.')
        self.__button.clicked.connect(self.save_image)
        self._timer.timeout.connect(self.timer_event)
        self.view_coordinates = [[0., .5, .5, 1.], [.5, .5, 1., 1.], [0., 0., .5, .5], [.5, 0., 1., .5]]
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
            self.vtkWidget.GetRenderWindow().AddRenderer(current_ren)
            cam_counter += 1

        self._iren.AddObserver('TimerEvent', self.update_poly_data)

        dt = 30  # ms
        ide = self._iren.CreateRepeatingTimer(dt)

        self.frame.setLayout(self.__create_grid_layout())
        self.setCentralWidget(self.frame)
        self.setWindowTitle("SmartUS Image Collector")
        self.__txt_box_idx.setFocus()

        self.show()
        self._iren.Initialize()
        self._iren.Start()

    def __create_grid_layout(self):
        layout = QGridLayout()

        layout.addWidget(self.vtkWidget, 0, 0, 1, 2)
        layout.addWidget(self.__label_idx, 1, 0, 1, 1)
        layout.addWidget(self.__txt_box_idx, 1, 1, 1, 1)
        layout.addWidget(self.__button, 2, 0, 1, 2)

        return layout

    def get_txt_box_idx(self):
        return self.__txt_box_idx

    def update_poly_data(self, obj=None, event=None):
        cam_counter = 0
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

        self._iren.GetRenderWindow().Render()
        # print(self.timer_count)
        self.timer_count += 1

    def create_timer(self, obj, evt):
        self._timer.start(0)

    def destroy_timer(self, obj, evt):
        self._timer.stop()
        return 1

    def timer_event(self):
        self._iren.TimerEvent()

    def save_image(self):
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d_%H:%M:%S.%f")
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(self.__prefix +
                           "animal_" +
                           self.__txt_box_idx.text() +
                           "_img_" + str(self.__counter) +
                           "_3D_" +
                           today_str +
                           ".vtp")
        writer.SetInputData(self.pd_collection[0])
        writer.Write()
        self.__counter += 1


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    # window.create_scene()
    sys.exit(app.exec_())
