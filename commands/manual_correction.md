

```bash
# Extract frames from video (only for the labeled frames)
python scripts/manual_correction/extract_frames.py


python scripts/manual_correction/build_prediction.py


python scripts/manual_correction/launch_web.py


from modules.detector_ssdlite_model import detector_extraction_ssdlite
```


video -> frames -> predictions (keypoints + bounding box) -> web interface for manual correction


Pipeline:
- A function that take a video path, frame indices (or a single index), and output folder as input, and extract the corresponding frames from the video. Save the frames in output folder
- A function `detector_extraction_ssdlite` that take a list of image rgb, return the bounding box and score for each detected rat in the image. 
- A function `keypoint_extraction_rtmpose` that take a list of image rgb and bounding box, return the keypoints for each detected rat in the image.




- A function that take a frame path (or a folder contains frames), model, and detector. Return the prediction in a list format (or a dict if only one frame is given). Each element in the list is a dictionary containing the keypoints and bounding box for a detected rat in the frame.


