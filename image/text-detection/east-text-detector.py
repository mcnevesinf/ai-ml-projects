
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression

import argparse

def main(args):
	image = cv2.imread("./"+args.image)

	orig = image.copy()
	(height, width) = image.shape[:2]

	#EAST model requires image dimensions (width, height) in multiple of 32
	(newWidth, newHeight) = (args.imgWidth, args.imgHeight)

	#Save scaling ratios
	rW = width / float(newWidth)
	rH = height / float(newHeight)

	image = cv2.resize(image, (newWidth, newHeight))

	#Preprocess the image
	blob = cv2.dnn.blobFromImage(image, 1.0, (newWidth, newHeight), 
			(123.68, 116.78, 103.94), swapRB=True, crop=False)

	#Import EAST model
	net = cv2.dnn.readNet("east-text-detection.pb")

	outputLayers = []
	outputLayers.append("feature_fusion/Conv_7/Sigmoid")
	outputLayers.append("feature_fusion/concat_3")

	net.setInput(blob)
	output = net.forward(outputLayers)

	scores = output[0] #Confidence score of the bounding boxes
	geometry = output[1] #Geometry of the bounding boxes around the text

	#Decode the positions of the text boxes along with their orientation
	#Filter out the best looking text-boxes

	rects = []
	confidences = []

	(numBoxRows, numBoxCols) = scores.shape[2:4]

	#Loop over the number of rows
	for y in range(0, numBoxRows):

		#Extract the scores (probabilities), followed by the geometrical data
		#used to derive box coordinates that surround text
		scoresData = scores[0,0,y]
		xData0 = geometry[0,0,y]
		xData1 = geometry[0,1,y]
		xData2 = geometry[0,2,y]
		xData3 = geometry[0,3,y]
		anglesData = geometry[0,4,y]

		#loop over the number of columns
		for x in range(0, numBoxCols):
		
			#If box score does not have sufficient probability, ignore it
			if scoresData[x] < args.confidence:
				continue

			#The EAST text detector reduces the image size (4x). Compute the 
			#offset factor to bring the coordinates back into respect of original image
			(offsetX, offsetY) = (x*4.0, y*4.0)

			#Extract the rotation angle then compute sin and cos
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			#Derive the bounding box coordinates for the text area
			#For RBOX geometry, the 5 channels represent four distances from the
			#pixel location (top, right, bottom and left boundaries) plus the rotation angle
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData2[x]

			#Compute x and y coordinates for the text prediction bounding box
			endX = int( offsetX + (cos * xData1[x]) + (sin * xData2[x]) )
			endY = int( offsetY + (sin * xData1[x]) + (cos * xData2[x]) )
			startX = int(endX - w)
			startY = int(endY - h)

			#Add the bounding box coordinates and probability score to the respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	#Apply non-maxima suppression to suppress weak, overlapping bounding boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	#Loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
	
		#Scale the coordinates back to the original image dimensions
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		#Draw the bounding box on the image
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
		
	#Show the processed image
	cv2.imshow("Processed image", orig)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Text detector')

	parser.add_argument('image', type=str, help='Input image')
	parser.add_argument('--width', type=int, dest='imgWidth', default=1920, help='Image width')
	parser.add_argument('--height', type=int, dest='imgHeight', default=1920, help='Image height')
	parser.add_argument('--confidence', type=float, dest='confidence', default=1e-4, help='Confidence threshold for text detector')

	args = parser.parse_args()

	main(args)














