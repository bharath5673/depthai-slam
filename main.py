from slam import process
from display import Display
from pointmap import PointMap
import cv2
import open3d as o3d
import depthai as dai

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.createColorCamera()

# camRgb.setPreviewSize(300, 300)
camRgb.setPreviewSize(600, 600)
# camRgb.setPreviewSize(900, 900)


camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)


# Create output
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)





pmap = PointMap()
display = Display()



pcd = o3d.geometry.PointCloud()
visualizer = o3d.visualization.Visualizer()
visualizer.create_window(window_name="3D plot", width=960, height=540)


# Connect to the device
with dai.Device() as device:
	# Print out available cameras
	print('Connected cameras: ', device.getConnectedCameras())
	# Start pipeline
	device.startPipeline(pipeline)

	# Output queue will be used to get the rgb frames from the output defined above
	# qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
	qRgb = device.getOutputQueue(name="rgb", maxSize=30, blocking=False)

	# while cap.isOpened():
	while True:
		# ret, frame = cap.read()
		frame = qRgb.get()
		frame = frame.getCvFrame()
		# cv2.imshow("bgr", frame.getCvFrame())

		# frame = cv2.resize(frame, (960, 540))
		img, tripoints, kpts, matches = process(frame)
		xyz = pmap.collect_points(tripoints)

		if kpts is not None or matches is not None:
			display.display_points2d(frame, kpts, matches)
		else:
			pass
		display.display_vid(frame)

		if xyz is not None:
			display.display_points3d(xyz, pcd, visualizer)
		else:
			pass
		if cv2.waitKey(1) == 27:
			break



	cv2.destroyAllWindows()
