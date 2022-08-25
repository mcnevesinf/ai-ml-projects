
#Adapted from: https://pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html

import argparse
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
import torchaudio


#Define the decoder
class GreedyCTCDecoder(torch.nn.Module):

	def __init__(self, labels, blank=0):
		super().__init__()
		self.labels = labels
		self.blank = blank
		
	def forward(self, emission : torch.Tensor):
		#Get most probable label
		indices = torch.argmax(emission, dim=-1)
		
		#Eliminate repeated labels
		indices = torch.unique_consecutive(indices, dim=-1)
		
		#Eliminate blank spaces
		indices = [i for i in indices if i != self.blank]
		
		return "".join([self.labels[i] for i in indices])


def main(args):
	matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

	torch.random.manual_seed(0)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	SPEECH_FILE = args.audio

	print(SPEECH_FILE)

	#Create the pipeline
	bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

	print("Sample rate: ", bundle.sample_rate)
	print("Labels: ", bundle.get_labels())

	model = bundle.get_model()

	#Load data
	waveform, sample_rate = torchaudio.load(SPEECH_FILE)
	waveform = waveform.to(device)

	#Resample audio in case the sample rate is different from what the pipeline expects
	if sample_rate != bundle.sample_rate:
		waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
	
	#Extract acoustic features and classify them
	with torch.inference_mode():
		emission, _ = model(waveform)
	
	#Visualize the logits
	plt.imshow(emission[0].cpu().T)
	plt.title("Classification result")
	plt.xlabel("Frame (time-axis)")
	plt.ylabel("Class")
	plt.show()
	
	#Instantiate the decoder
	decoder = GreedyCTCDecoder(labels=bundle.get_labels())

	#Decode the transcript
	transcript = decoder(emission[0])

	print(transcript)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Automatic Speech Recognition')

	parser.add_argument('audio', type=str, help='Input audio')

	args = parser.parse_args()

	main(args)
		























