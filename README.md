
# Concurrent FairMOT with Simple Camera Motion Cancellation

![output](https://user-images.githubusercontent.com/37895847/111121868-8bbba180-85b0-11eb-8607-21330daca119.gif)


This repository contains the source code for our concurrent pipeline implementation for FairMOT. This implementation increases execution speed by up to 50%.

In addition, we apply a simple 4 parameter affine transformation-based frame registering method to cancel camera motion which can cause problems for tracking far away small targets.

The exact extents of the applicability of this method are unknown, however rough requirements are that there should be a dedicated plane of potential 2D motion (e.g., the ground), and any deterioration from this plane should be relatively small compared to the camera distance. Furthermore, the plane should be visible from a steep angle, to reduce errors from perspective distortions. A typical example, and our use-case for this, is a footage taken from helicopter at several hundred meters looking down at an urban environment at a steep angle.

We use CSRT method for tracking key-points. For this to work, we need to specify key-points on the first frame either by the input file or, if the points are omitted, through a dialog window. In the latter case a window manager is required, and if it is not present the program will crash. The key-point tracking has its own thread for every point. We used OpenCV, so it is a true separate thread, that can run on its own core. We found that CSRT runs at about 30 FPS on a 3.2 GHz core, therefore on such processors it is not hindering the real-time capabilities, given that there are enough CPU cores for every point. This also depends on the size of the track boxes (we used 40x40).

For the frame registration to work well, distinguishable objects (lamps, signs) close to the ground (plane of motion) must be selected, and they should be always visible. We found, that if a moving entity (person or car) passes over the object and occludes it, it will hijack the track box. Otherwise, the CSRT method proved to be very robust against target loss, and error accumulations since it is feature-based.

We have two more addition to the original code. Firstly, both training and inference can be done using CPU, which is good if you do not want to waste precious GPU resources for testing. This was an option originally as well, but it was not working. Note that you will need to compile the necessary DCNv2 library for the appropriate target. And secondly, we added an option for skipping frames which can basically emulate real-time execution, where processing time for one frame takes longer than the time between consecutive frames.

The root directory contains sample shell scripts to show how to execute.

One additional note is that the naming of the data handler module “dataset” is a little bit unfortunate, and in our case, it caused name collision with the “dataset” module of the slim package. There are a number of ways to resolve this. Simplest is of course, to not have slim installed at all. Or you can set up a new python environment without slim. We simply removed the slim package at the start of our scripts before importing the dataset “module”.

## Training
### Sample command
`python train.py --gpus 0 mot --data_cfg ./lib/cfg/TH2020_split1.json`

There are no big surprises here, this is essentially the same as the original [FairMOT](https://github.com/ifzhang/FairMOT) code.

## Tracking
### Input file
```json
[
    {
        "subdir" : "TH2020_MOT/eval_q",
        "seqs" :
        [
            {"seq":"01B_3", "kps": {"0":[678, 68, 40, 40], "1": [654, 232, 40, 40], "2": [540, 30, 40, 40], "3": [338, 218, 40, 40]}},
            {"seq":"01B_4", "kps": {"0":[588, 240, 40, 37], "1": [509, 346, 46, 39], "2": [194, 112, 39, 36], "3": [208, 333, 38, 38]}}
        ]
    }
]
```

There can be multiple sequences sets specified.

###### subdir - The set’s subdirectory relative to the data root (which is “<repo_root>/../Dataset”).

###### seqs – List of the sequences in the set.

###### seq – Directory name of the sequence. The directory structure must conform to Clear MOT format.

###### kps – Key-point center coordinates and box sizes as (center_x, center_y, w, h), can be empty.

### Sample command
`python track.py mot --load_model ../models/fairmot_th2020_dla34.pth --gpus 0 --val_seq_file TH2020_seqs_4p.json --cmc_on --cmc_type aff --mot_frame_skip 1`


###### val_seq_file – The input file containing the sequences for validation. It is searched for under the “<repo_root>/src/data” directory.

###### cmc_on – Turns on frame registration. If this is not enabled, key-points are ignored. If it is enabled, but there are no key-points, a dialog window will appear (if there is no window manager, e.g. executed from terminal, this will crash) that asks for key-points in the first frame.

###### cmc_type – Experimental. Possible values are “aff” and “proj”. By default, we use 4 parameter affine transformation, but we also implemented support for projective transformation. However, we did not test this thoroughly. 

###### mot_frame_skip – the number of frames that should be skipped during MOT calculations. Not setting this or setting it to 0 processes all frames. Otherwise, if it is set to N, out of every N+1 frames N will be ignored, starting from the second frame. E.g. if frame skip is 3 then, frames 1,5,9,13… will be processed. Note that this is only for MOT. The key-point tracking will still process all frames.

## Demo
### Input file
```json
[
    {
        "subdir" : "",
        "seqs" :
        [
            {"seq":"03-G.mp4", "kps": {"0":[445, 175, 35, 35], "1":[680, 210, 35, 35]}}
        ]
    }
]
```

Same as for tracking except for “seq” which in this case should refer to the video file. “subdir” in this case is relative to the input video root that is the “<repo_root>/videos” by default.

### Sample command
`python demo.py mot --input-root ../videos --demo_seq_file TH2020_demo_seqs.json --load_model ../models/fairmot_th2020_dla34.pth --gpus -1`

###### input-root -The directory under which the program should look for the subdirectories specified in the input file.

###### demo_seq_file – The input file containing the videos for demonstration. It is searched for under the “<repo_root>/src/data” directory.
The other extra parameters for tracking are working here in exactly the same way. Note that if frame skip is applied, the rendered output video will still contain all frames, but skipped frames will not have bounding boxes.

## TH2020 dataset
If you want to use our helicopter-based medium altitude dataset you download it below.

Images with bounding box annotation for training: [link](https://sekilab-students.s3-ap-northeast-1.amazonaws.com/Gergely/TH2020/TH2020_bb.zip)

Video sequences with ID-based annotation for evaluation:  [link](https://sekilab-students.s3-ap-northeast-1.amazonaws.com/Gergely/TH2020/TH2020_MOT_eval_seq.zip)

We also share a model pretrained on the above dataset: [link](https://sekilab-students.s3-ap-northeast-1.amazonaws.com/Gergely/TH2020/fairmot_th2020_dla34.pth)

For further models please see the original FairMOT page: https://github.com/ifzhang/FairMOT

We provided sample input files under “<repo_root>/src/data”,  and “src/lib/cfg/” which help in the organizing of the data structure

## Final comments and acknowledgements
The original code was borrowed from the FairMOT repo: https://github.com/ifzhang/FairMOT

It had several problems (probably still has), and in many cases we just removed parts that were unnecessary for us because it was simpler than fixing it. We found some dataset dependent hardcoded settings well hidden (e.g. max number of targets), and there could be some more. If you encounter problems with other datasets, we recommend you investigate these first.

Regardless we would like to thank the authors for publishing their work.

We also created similar implementations for [DeepSORT](https://github.com/nwojke/deep_sort) and [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT), however we believe that getting one of these into publishable shape should be enough. Regardless, we would like to thank the authors of these as well.

# Citation
If you find our source code, or dataset useful please cite our paper:
[Online real-time pedestrian tracking from medium altitude aerial footage with camera motion cancellation](https://doi.org/10.1016/j.cviu.2022.103386)

# License
Images on this dataset are available under the [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/) (CC BY-SA 4.0). The license and link to the legal document can be found next to every image on the service in the image information panel and contains the CC BY-SA 4.0 mark:
<br><a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/deed.en"><img alt="Creative Commons License" style="border-width:0" src="https://licensebuttons.net/l/by-sa/4.0/88x31.png" /></a><br />

