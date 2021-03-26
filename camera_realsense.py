"""
Intel RealSense Camera: Implement CameraOpenCV with Intel RealSense Camera
"""

# __all__ = ('RealSenseCamera', 'RealSenseColorCamera', 'RealSenseDepthCamera', 'RealSense3DCamera')

# from enum import IntEnum
#
# from kivy.logger import Logger
# from kivy.clock import Clock
# from kivy.graphics.texture import Texture
# from kivy.core.camera.camera_opencv import CameraOpenCV

import cv2
import math
import pyrealsense2 as rs
import numpy as np


# class AppState:
#     def __init__(self, *args, **kwargs):
#         self.WIN_NAME = 'RealSense'
#         self.pitch, self.yaw = math.radians(-10), math.radians(-15)
#         self.translation = np.array([0, 0, -1], dtype=np.float32)
#         self.distance = 2
#         self.prev_mouse = 0, 0
#         self.mouse_buttons = [False, False, False]
#         self.paused = False
#         self.decimate = 1
#         self.scale = True
#         self.color = True
#
#     def reset(self):
#         self.pitch, self.yaw, self.distance = 0, 0, 2
#         self.translation[:] = 0, 0, -1
#
#     @property
#     def rotation(self):
#         rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
#         ry, _ = cv2.Rodrigues((0, self.yaw, 0))
#         return np.dot(ry, rx).astype(np.float32)
#
#     @property
#     def pivot(self):
#         return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


class RealSenseCamera:
    def __init__(self, resolution, frame_rate):
        # Configure depth and color streams
        self.__pipeline = rs.pipeline()
        self.__config = rs.config()
        self.__profile = None
        self.__resolution = resolution
        self.__frame_rate = frame_rate
        self.__clipping_distance = 0.

    def config(self):
        self.__config.enable_stream(
            rs.stream.depth, self.__resolution[0], self.__resolution[1], rs.format.z16, self.__frame_rate)
        self.__config.enable_stream(
                rs.stream.color, self.__resolution[0], self.__resolution[1], rs.format.rgb8, self.__frame_rate)

    def start_pipeline(self):
        self.__profile = self.__pipeline.start(self.__config)

    def set_clip_distance(self, clipping_distance_in_meters):
        # getting the depth sensor's depth scale (see rs-align example for explanation)
        if self.__profile:
            depth_sensor = self.__profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            self.__clipping_distance = clipping_distance_in_meters / depth_scale
        else:
            print("\nPipeline has not been started.\n")

    def get_pipeline(self):
        return self.__pipeline

    def get_config(self):
        return self.__config

    def get_frame_rate(self):
        return self.__frame_rate

    def get_clip_distance(self):
        return self.__clipping_distance


# class RealSenseColorCamera(CameraOpenCV):
#     """
#     Implementation of CameraBase using Intel Real Sense Depth Camera D435
#     """
#     _update_ev = None
#
#     def __init__(self, realsense_cam_config, **kwargs):
#         self._camera = None
#         self._pipeline = realsense_cam_config.get_pipeline()
#         self._config = realsense_cam_config.get_config()
#         self.fps = 1. / realsense_cam_config.get_frame_rate()
#         self._is_object_detected = False
#         super(RealSenseColorCamera, self).__init__(**kwargs)
#
#     def init_camera(self):
#         if self._camera is not None:
#             self._camera.close()
#
#         if not self.stopped:
#             self.start()
#
#     def object_detected(self, is_object_detected):
#         self._is_object_detected = is_object_detected
#
#     def _update(self, dt):
#         if self.stopped:
#             return
#
#         if self._texture is None:
#             # Create the texture
#             self._texture = Texture.create(self._resolution)
#             self._texture.flip_vertical()
#             self.dispatch('on_load')
#
#         try:
#             # Wait for a coherent color frame
#             frames = self._pipeline.wait_for_frames()
#
#             color_frame = frames.get_color_frame()
#             color_image = np.asanyarray(color_frame.get_data())
#
#             if self._is_object_detected:
#                 # call the cnn here to detect objects in the image
#                 detected_obj_img = detect_objects_in_image(color_image)
#                 self._buffer = detected_obj_img.reshape(-1)
#             else:
#                 self._buffer = color_image.reshape(-1)
#
#             self._copy_to_gpu()
#
#         except KeyboardInterrupt:
#             raise
#         except Exception:
#             Logger.exception("CameraRealSense: Could not get image from Camera.")
#
#     def start(self) -> object:
#         super(RealSenseColorCamera, self).start()
#         if self._update_ev is not None:
#             self._update_ev.cancel()
#         self._update_ev = Clock.schedule_interval(self._update, self.fps)
#
#     def stop(self):
#         super(RealSenseColorCamera, self).stop()
#         if self._update_ev is not None:
#             self._update_ev.cancel()
#             self._update_ev = None
#
#
# class RealSenseDepthCamera(CameraOpenCV):
#     """
#     Implementation of CameraBase using Intel Real Sense Depth Camera D435
#     """
#     _update_ev = None
#
#     def __init__(self, realsense_cam_config, **kwargs):
#         self._camera = None
#         self._pipeline = realsense_cam_config.get_pipeline()
#         self._config = realsense_cam_config.get_config()
#         self.fps = 1. / realsense_cam_config.get_frame_rate()
#         self._colorizer = rs.colorizer()
#         super(RealSenseDepthCamera, self).__init__(**kwargs)
#
#     def init_camera(self):
#         if self._camera is not None:
#             self._camera.close()
#
#         if not self.stopped:
#             # Start streaming
#             self.start()
#
#     def _update(self, dt):
#         if self.stopped:
#             return
#
#         if self._texture is None:
#             # Create the texture
#             self._texture = Texture.create(self._resolution)
#             self._texture.flip_vertical()
#             self.dispatch('on_load')
#
#         try:
#             # Wait for a coherent color frame
#             frames = self._pipeline.wait_for_frames()
#
#             depth_frame = frames.get_depth_frame()
#             # apply colormap on depth image (image must be converted to 8-bit per pixel first)
#             depth_colormap = np.asanyarray(self._colorizer.colorize(depth_frame).get_data())
#             # reshape color image to a 1d array
#             self._buffer = depth_colormap.reshape(-1)
#             self._copy_to_gpu()
#
#         except KeyboardInterrupt:
#             raise
#         except Exception:
#             Logger.exception("CameraRealSense: Could not get image from Camera.")
#
#     def start(self) -> object:
#         super(RealSenseDepthCamera, self).start()
#         if self._update_ev is not None:
#             self._update_ev.cancel()
#         self._update_ev = Clock.schedule_interval(self._update, self.fps)
#
#     def stop(self):
#         super(RealSenseDepthCamera, self).stop()
#         if self._update_ev is not None:
#             self._update_ev.cancel()
#             self._update_ev = None
#
#
# class RealSense3DCamera(CameraOpenCV):
#     """
#     Implementation of CameraBase using Intel Real Sense Depth Camera D435
#     """
#     _update_ev = None
#
#     def __init__(self, realsense_cam_config, **kwargs):
#         self._camera = None
#         self._pipeline = realsense_cam_config.get_pipeline()
#         self._config = realsense_cam_config.get_config()
#         self.fps = 1. / realsense_cam_config.get_frame_rate()
#         self._pc = rs.pointcloud()
#         self._w_intrinsics = 0
#         self._h_intrinsics = 0
#         self._decimate = rs.decimation_filter()
#         self._state = AppState()
#         super(RealSense3DCamera, self).__init__(**kwargs)
#
#     def init_camera(self):
#         if self._camera is not None:
#             self._camera.close()
#
#         if not self.stopped:
#             # Start streaming
#             profile = self._pipeline.get_active_profile()
#             depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
#             depth_intrinsics = depth_profile.get_intrinsics()
#             self._w_intrinsics, self._h_intrinsics = depth_intrinsics.width, depth_intrinsics.height
#             self._decimate.set_option(rs.option.filter_magnitude, 1. ** self._state.decimate)
#             self.start()
#
#     def _update(self, dt):
#         if self.stopped:
#             return
#
#         if self._texture is None:
#             # Create the texture
#             self._texture = Texture.create(self._resolution)
#             self._texture.flip_vertical()
#             self.dispatch('on_load')
#
#         try:
#             # Wait for a coherent color frame
#             frames = self._pipeline.wait_for_frames()
#
#             depth_frame = frames.get_depth_frame()
#             color_frame = frames.get_color_frame()
#             depth_frame = self._decimate.process(depth_frame)
#             color_image = np.asanyarray(color_frame.get_data())
#
#             points = self._pc.calculate(depth_frame)
#             self._pc.map_to(color_frame)
#             proj_2d_img_from_pc = self.__proj_point_cloud_to_2d(
#                 color_image, points, self._w_intrinsics, self._h_intrinsics)
#
#             # reshape color image to a 1d array
#             self._buffer = proj_2d_img_from_pc.reshape(-1)
#             self._copy_to_gpu()
#
#         except KeyboardInterrupt:
#             raise
#         except Exception:
#             Logger.exception("CameraRealSense: Could not get image from Camera.")
#
#     def __proj_point_cloud_to_2d(self, color_image, points, w, h, painter=True):
#         """draw point cloud with optional painter's algorithm"""
#         # point cloud data to arrays
#         proj_2d_img = np.zeros((h, w, 3), dtype=np.uint8)
#         v, t = points.get_vertices(), points.get_texture_coordinates()
#         vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
#         tex_coordinates = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
#         proj_2d_img.fill(255)
#
#         if painter:
#             # Painter's algo, sort points from back to front
#             # get reverse sorted indices by z (in view-space)
#             # https://gist.github.com/stevenvo/e3dad127598842459b68
#
#             v = self.__view(vertices)
#             s = v[:, 2].argsort()[::-1]
#             proj = self.__project(v[s])
#
#         else:
#             proj = self.__project(self.__view(vertices))
#
#         if self._state.scale:
#             proj *= 1. ** self._state.decimate
#
#         # proj now contains 2d image coordinates
#         j, i = proj.astype(np.uint32).T
#
#         # create a mask to ignore out-of-bound indices
#         im = (i >= 0) & (i < self._resolution[0])
#         jm = (j >= 0) & (j < self._resolution[1])
#         m = im & jm
#
#         cw, ch = color_image.shape[:2][::-1]
#
#         if painter:
#             v, u = (tex_coordinates[s] * (cw, ch) + 0.5).astype(np.uint32).T
#         else:
#             v, u = (tex_coordinates * (cw, ch) + 0.5).astype(np.uint32).T
#         # clip tex_coordinates to image
#         np.clip(u, 0, ch-1, out=u)
#         np.clip(v, 0, cw-1, out=v)
#
#         # perform uv-mapping
#         proj_2d_img[i[m], j[m]] = color_image[u[m], v[m]]
#
#         return proj_2d_img
#
#     def __project(self, vertices):
#         """project 3d vector array to 2d"""
#         w, h = self._resolution[0], self._resolution[1]
#         view_aspect = float(h) / w
#
#         # ignore divide by zero for invalid depth
#         with np.errstate(divide='ignore', invalid='ignore'):
#             proj = vertices[:, :-1] / vertices[:, -1, np.newaxis] * \
#                    (w * view_aspect, h) + (w / 2.0, h / 2.0)
#
#         # near clipping
#         z_near = 0.03
#         proj[z_near > vertices[:, 2]] = np.nan
#         return proj
#
#     def __view(self, vertices):
#         """apply view transformation on vector array"""
#         state = self._state
#         return np.dot(vertices - state.pivot, state.rotation) + state.pivot - state.translation
#
#     def start(self) -> object:
#         super(RealSense3DCamera, self).start()
#         if self._update_ev is not None:
#             self._update_ev.cancel()
#         self._update_ev = Clock.schedule_interval(self._update, self.fps)
#
#     def stop(self):
#         super(RealSense3DCamera, self).stop()
#         if self._update_ev is not None:
#             self._update_ev.cancel()
#             self._update_ev = None
#
#
# # Load Yolo
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# classes = []
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#     colors = np.random.uniform(0, 255, size=(len(classes), 3))
#
#
# def detect_objects_in_image(img):
#     # Loading image
#     # img = cv2.imread("cow1.jpg")
#     # img = cv2.resize(img, None, fx=0.4, fy=0.4)
#     height, width, channels = img.shape
#
#     # Detecting objects
#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
#
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#
#     # Showing information on the screen
#     class_ids = []
#     confidences = []
#     boxes = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#     font = cv2.FONT_HERSHEY_PLAIN
#
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             color = colors[class_ids[i]]
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
#
#     return img


# class Preset(IntEnum):
#     Custom = 0
#     Default = 1
#     Hand = 2
#     HighAccuracy = 3
#     HighDensity = 4
#     MediumDensity = 5


# def get_intrinsic_matrix(frame):
#     intrinsics = frame.profile.as_video_stream_profile().intrinsics
#     out = PinholeCameraIntrinsic(640, 480, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
#     return out

# class CameraRealSense3D(CameraOpenCV):
#     def __init__(self, **kwargs):
#         self._camera = None
#         self._pipeline = None
#         self._config = None
#         self._format = 'bgr'
#         self._frame_rate = kwargs.get('frame_rate', 30)
#         self.fps = 1. / self._frame_rate
#         self._align = None
#         self._pcd = PointCloud()
#         self._clipping_distance_in_meters = 1.5
#         self._clipping_distance = None
#         self._depth_scale = None
#         self._flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
#         super(CameraRealSense3D, self).__init__(**kwargs)
#
#     def init_camera(self):
#         if self._camera is not None:
#             self._camera.close()
#
#         # Create a pipeline
#         self._pipeline = rs.pipeline()
#
#         # Create a config and configure the pipeline to stream
#         # different resolutions of color and depth streams
#         self._config = rs.config()
#
#         self._config.enable_stream(rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, 30)
#         self._config.enable_stream(rs.stream.color, self.resolution[0], self.resolution[1], rs.format.rgb8, 30)
#
#         if not self.stopped:
#             # Start streaming
#             profile = self._pipeline.start(self._config)
#             self.start()
#             depth_sensor = profile.get_device().first_depth_sensor()
#
#             # Using preset HighAccuracy for recording
#             depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)
#
#             # Getting the depth sensor's depth scale (see rs-align example for explanation)
#             self._depth_scale = depth_sensor.get_depth_scale()
#
#             # We will not display the background of objects more than
#             #  clipping_distance_in_meters meters away
#             self._clipping_distance_in_meters = 1.5  # 1 meter
#             self._clipping_distance = self._clipping_distance_in_meters / self._depth_scale
#             # print(depth_scale)
#
#             # Create an align object
#             # rs.align allows us to perform alignment of depth frames to others frames
#             # The "align_to" is the stream type to which we plan to align depth frames.
#             align_to = rs.stream.color
#             self._align = rs.align(align_to)
#
#     def _update(self, dt):
#         if self.stopped:
#             return
#
#         if self._texture is None:
#             # Create the texture
#             self._texture = Texture.create(self._resolution)
#             self._texture.flip_vertical()
#             self.dispatch('on_load')
#
#         try:
#             # Get frame set of color and depth
#             frames = self._pipeline.wait_for_frames()
#
#             # Align the depth frame to color frame
#             aligned_frames = self._align.process(frames)
#
#             # Get aligned frames
#             aligned_depth_frame = aligned_frames.get_depth_frame()
#             color_frame = aligned_frames.get_color_frame()
#             intrinsic = get_intrinsic_matrix(color_frame)
#
#             depth_image = Image(np.array(aligned_depth_frame.get_data()))
#             color_image = Image(np.array(color_frame.get_data()))
#             vra = np.array(color_frame.get_data())
#             print("vra shape: ", vra.shape)
#             print("vra type: ", type(vra))
#             print("vra min element: ", np.min(vra))
#             print("vra max element: ", np.max(vra))
#
#             rgb_image = RGBDImage.create_from_color_and_depth(color_image,
#                                                               depth_image,
#                                                               depth_scale=1. / self._depth_scale,
#                                                               depth_trunc=self._clipping_distance_in_meters,
#                                                               convert_rgb_to_intensity=False)
#
#             temp = PointCloud.create_from_rgbd_image(rgb_image, intrinsic)
#             temp.transform(self._flip_transform)
#             self._pcd.points = temp.points
#             self._pcd.colors = temp.colors
#             points = np.asarray(self._pcd.points)
#             colors = np.asarray(self._pcd.colors)
#             print("point cloud data type: ", points[0])
#             print("color matrix data type: ", colors[0])
#
#             # reshape color image to a 1d array
#             self._buffer = vra.reshape(-1)
#             print("buffer shape: ", self._buffer.shape)
#             print("buffer data type: ", type(self._buffer[0]))
#             self._copy_to_gpu()
#         except KeyboardInterrupt:
#             raise
#         except Exception:
#             Logger.exception("CameraRealSense: Could not get image from Camera.")
#
#     def start(self) -> object:
#         super(CameraRealSense3D, self).start()
#         if self._update_ev is not None:
#             self._update_ev.cancel()
#         self._update_ev = Clock.schedule_interval(self._update, self.fps)
#
#     def stop(self):
#         super(CameraRealSense3D, self).stop()
#         if self._update_ev is not None:
#             self._update_ev.cancel()
#             self._update_ev = None
