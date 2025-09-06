Please create and implement a Machine-Learning based auto timing for subtitling project for me.

## Goal & Use case
In fansub community, people do subtitles for some foreign videos with manual timing -- listen to the audio, watch the video, and marking start and end timestamps for each line of subtitle. Instead of manually doing so, I would like to enhance it via ML.

Challanges:
- Subtitle lines may overlap, sometimes intensively
- We need to understand what they are talking about to split the lines properly

## Approach
We treat this problem as object detection for 1D (audio) data. The network is similar to CornerNet, works as follows:
- Takes audio signal $a(t)$ as input. Perhaps, spectrum.
- Process audio signal to features (time-seq $f(t)$) using some audio backbone.
- The model consists of "Starting network" and "Ending network".
- For each timestamp $t$, each (starting/ending) network outputs the followings:
	- A binary classification result / heatmap from 0~1.
		- For starting network, that is the confidence that a line of subtitle will start at $t$, and vice versa for ending net.
	- A feature with pre-defined dimensions.
	- Starting/Ending networks should be identical for now.
- A NPM filtering step to find local extreme values of starting/ending heatmap.
	- That is, if `f(t) == maxpool(f(t), time_window)`, it is considered as local extreme value.
- As a final step, a post-processor will find relavent pairs of starting/ending timings, by comparing their features. It rejects pairs with distant features.

To train, we use
- Focal loss, which compares the heatmap with the GT timing labels. A gaussian could be convolved with the GT label (discrete timings) to make soft labels.
- Feature loss, which pulls together GT start-end pair's feature and pushs away negative pair's feature.

Misc notes.
- Timestamps should be in 60FPS (i.e., 16ms).
## Pre-processing
Data preproc is the most tedious part and I'd like you to help me the most.

Starting from raw files:
- Audio files (`mp3`, `ogg`, `m4a`, etc.)
- Substation alpha subtitle files (`ass`, can be handled with the python package `ass`)

To construct dataset, we need to:
- Collect all files under a folder, identify audio-subtitle pairs
- Do necessary processing (e.g., extract spectrum, etc.)
- Cut the audio and corresponding subtitles into small blocks, say, 30sec. (idk what sizes are good)
	- If there are subtitle lines that goes outside the time window, keep the subtitle line but ignore out-of-bound starting / ending marks.
	- Remember to re-time the subtitles so it matches the trimmed audio timestamp.
	- It might be necessary to include paddings due to how conv nets work. For blocks within the audio, use the original audio for padding. For blocks at the border of audio, use silent padding.
- Make dataset files with those processed files
Above step should be easily done with a folder path. I want to create many versions of datasets.

Then train the network using the dataset. Do the validations also.

Finally, for infrerence:
- Take audio input
- Cut it into overlapping time windows as in the training step; pad if necessary
- Compute heatmaps, do post-processing, concatenate and obtain pairs
- Output ass file using the pairs obtained.

## Project Infra
I want a modular config file that lets me to edit config easily. e.g., overall config, containing model config, etc.

I have installed the following libraries:
```
librosa
torch==2.8.0
ass
```
You can check by `pip list`.

We work under the conda environment `torch`.

The project needs to be work under windows (`cuda`) and macos (`mps`). So, we need to be able to switch devices (or auto-detect).
