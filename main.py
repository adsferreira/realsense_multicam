import pyrealsense2 as rs
import numpy as np
import cv2
import vtk
import vtk.util.numpy_support as vtk_np

from realsense_device_manager import DeviceManager


colors = vtk.vtkUnsignedCharArray()
pd = vtk.vtkPolyData()
mapper = vtk.vtkDataSetMapper()
actor = vtk.vtkActor()
ren = vtk.vtkRenderer()
ren_win = vtk.vtkRenderWindow()
render_window_interaction = vtk.vtkRenderWindowInteractor()


def visualise_measurements_cv2(frames_devices):
    """
    Calculate the cumulative pointcloud from the multiple devices
    Parameters:
    -----------
    frames_devices : dict
    	The frames from the different devices
    	keys: str
    		Serial number of the device
    	values: [frame]
    		frame: rs.frame()
    			The frameset obtained over the active pipeline from the realsense device
    """
    for (device, frame) in frames_devices.items():
        color_image = np.asarray(frame[rs.stream.color].get_data())
        text_str = device
        cv2.putText(color_image, text_str, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
        # Visualise the results
        text_str = 'Color image from RealSense Device Nr: ' + device
        cv2.namedWindow(text_str)
        cv2.imshow(text_str, color_image)
        cv2.waitKey(1)


def visualise_vtk(frames_devices):
    for (device, frame) in frames_devices.items():
        depth_frame = frame[rs.stream.depth]
        color_frame = frame[rs.stream.color]

        pc = rs.pointcloud()
        point_cloud = pc.calculate(depth_frame)
        pc.map_to(color_frame)
        v, t = point_cloud.get_vertices(), point_cloud.get_texture_coordinates()
        vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)

        np_array = vertices

        color_image = np.asanyarray(frame[rs.stream.color].get_data())
        color_image = color_image.reshape((color_image.shape[0] * color_image.shape[1], 3))

        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()

        colors.SetArray(vtk_np.numpy_to_vtk(color_image), color_image.shape[0] * color_image.shape[1], 1)
        pd.GetPointData().SetScalars(colors)

        points.SetData(vtk_np.numpy_to_vtk(np_array))
        cells_npy = np.vstack([np.ones(np_array.shape[0], dtype=np.int64),
                               np.arange(np_array.shape[0], dtype=np.int64)]).T.flatten()
        cells.SetCells(np_array.shape[0], vtk_np.numpy_to_vtkIdTypeArray(cells_npy))

        pd.SetPoints(points)
        pd.SetVerts(cells)
        pd.Modified()

        mapper.SetInputDataObject(pd)
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToPoints()
        ren.AddActor(actor)
        ren_win.AddRenderer(ren)
        render_window_interaction.SetRenderWindow(ren_win)

    ren_win.Render()
    render_window_interaction.Initialize()
    render_window_interaction.Start()


# Define some constants 
resolution_width = 640  # pixels
resolution_height = 480  # pixels
frame_rate = 15  # fps
dispose_frames_for_stablisation = 30  # frames

try:
    # Enable the streams from all the intel realsense devices
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
    # rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
    rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

    # Use the device manager class to enable the devices and get the frames
    device_manager = DeviceManager(rs.context(), rs_config)
    device_manager.enable_all_devices()

    # Allow some frames for the auto-exposure controller to stablise
    while True:
        frames = device_manager.poll_frames()
        visualise_measurements_cv2(frames)

except KeyboardInterrupt:
    print("The program was interupted by the user. Closing the program...")

finally:
    device_manager.disable_streams()
    cv2.destroyAllWindows()
