# Changelog

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
