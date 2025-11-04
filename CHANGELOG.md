# Changelog

0.13.0 - (2025-11-03)
------------------

* Fixed high quality output mode to be more amenable to uploads to YouTube and other shorts sites.
* Added `--audio-directory-random` flag to the CLIs to be able to set a directory of files as a
source for audio for the videos. A random audio file will be picked from the directory.
* Added some more environment variable passing.


0.12.3 - (2025-10-30)
------------------

* Fixes color order bug.


0.12.2 - (2025-10-30)
------------------

* Fixes bug where if a single video was passed into the video concat function it would result in an 
invalid video.


0.12.1 - (2025-10-30)
------------------

* A few fixes following running the UT suite on another GPU server.


0.12.0 - (2025-10-28)
------------------

* Introduces a new concept to the `viderator` internal package, `html_on_image` used to draw text
and other graphics onto images using HTML/CSS. 
* Adds an initial simple thumbnail implementation and unit test.
* Fixes bug in crop score where entire input had to be re-read when selecting output frames.


0.11.0 - (2025-10-24)
------------------

* Moved a bunch of UI code into a library module so it can be consumed in other packages.
* Dropped vidgear as an ffmpeg interface.
* GPU resources are freed after timelapses are created.


0.10.0 - (2025-10-21)
------------------

* Added `--layout` to `content-cropped`, can now take the best regions of an image and crop to 
them, combining the output in a new video. This is for going from landscape -> portrait while
retaining more of the image.
* Internally did a lot of refactoring to support this change, broke more code into internal API.
* Was able to factor out the intermediate cropped video piece in favor of just re-cropping the
input video.
* Added `--best-frame-path` to content mode for generating thumbnails.


0.9.0 - (2025-10-18)
------------------

* Fixed bug in intermediate video file storage of cropped frames, opencv could only handle around
3 hours for some reason, switched to the ffmpeg wrapper already implemented in viderator.
* Added GPU arguments to both of the content aware CLIs to limit GPU hogging.


0.8.1 - (2025-09-17)
------------------

* Small fixes in `cropped-content`.
* Uses video file for storing cropped frames, don't love this.


0.8.0 - (2025-09-16)
------------------

* Introduced a new CLI command, `content-cropped` that uses POI analysis to pick the most 
interesting region of a given aspect ratio to crop the output to. Down-sampling on score also 
occurs.
* Reorganized project a bit to support this new mode.
* Fixed bug in buffer where frames would not flow if the buffer was larger than the input frames.
* All CLIs now support the `--audio` option to add audio files ot the videos.


0.7.0 - (2025-09-07)
------------------

* `--buffer-size` is now an independent argument from batch size, this is how many frames are
loaded into memory from disk prior to feeding any frames to the GPU.
* Input images are pre-shrunk to near their final size before sitting in memory to reduce the
overall memory footprint of the application.
* Determined that video decode speed is the key bottleneck to throughput. Started work on fix.


0.6.0 - (2025-08-25)
------------------

* Fixed a long-standing bug where images were cropped to 224x224 instead of scaled and padded.
* Implemented a VIT attention map processing/scoring backend.


0.5.0 - (2025-06-22)
------------------

* Broke CLI into multiple commands: `content` for content aware and `classic` for down-sampling.
* Implemented the down-sampling mode. 


0.4.0 - (2025-06-21)
------------------

* Added a CLI switch to go between multiple "vectorization backends". Right now VIT and CLIP.
* Switched to VIT CLS token analysis for VIT Mode.
* Added CLIP analysis as a backend.
* Implemented first pass at a radius deselection method to avoid clusters in the output videos.
* Added unit tests that actually verify interesting frames are selected over boring ones.
* Added sensible weights for the different vector score results.


0.3.0 - (2025-04-02)
------------------

* Fixed output parallelization, needed to switch format of vector file hdf5.
* Better defined the scoring process types and test.
* Tested with a bunch of different scoring weights but never found anything that worked 
exceptionally well. Going to move to other approaches in subsequent versions. 


0.2.0 - (2025-01-18)
------------------

* Increased data throughput to be able to work through a large amount of video.
* Parallelized hdf5 compression, better usage of multiple GPUs. 


0.1.0 - (2024-10-15)
------------------

* Works good enough to produce videos for twitter. Model is `vit_base_patch16_224` and scores
are based on variance and saliency.


0.0.1 - (2024-09-18)
------------------

* Project begins
