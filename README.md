# OptonicsObjectReconstruction
[![Contributor](https://img.shields.io/badge/contributor-cracee-brightgreen.svg)](https://github.com/Cracee/OptonicsObjectReconstruction)
[![Project](https://img.shields.io/badge/Project%20under%20supervision%20of-OPTONIC-blue.svg)](https://github.com/Cracee/OptonicsObjectReconstruction)

# Table of Content
<ol>
  <li><a href='#intro'>Introduction</a></li>
  <li><a href='#sub'>Subfolder Explanation</a></li>
</ol>

# <span id='intro'>Introduction</span>

The repository of the Master Project of Cracee. Goal is to reconstruct 3D Objects via a top view in bins. You can get a lot of point cloud fragments by the camera system we are using. From these fragments it should be possible to re-puzzle the object. 

# <span id='sub'>Subfolder Explanation</span>

For this project, there are different subfolders:

In `Registration` you can find the 1. try, with manual ICP Algorithms, to puzzle together an object from different subparts.

In `nxlib_tutorials` you can find some Tutorials from the Ensenso SDK Website. They are used to test the environment, if camera communication works etc.

The [OMNet_Pytorch-main], [prnet-master] and [UTOPIC-main] folders contain DL approaches, which will get fitted to this project in the next days and weeks. 

<details><summary>Do not click to open</summary>
  <ul>
    <li>Got you!</li>
    <li>Now you probably feel ashamed</li>
    <li>But that is ok</li>
    <li>I would have clicked it too</li>
    <li>Don't worry, be happy!</li>
  </ul>
</details>

[OMNet_Pytorch-main]: https://github.com/hxwork/OMNet_Pytorch
[prnet-master]: https://github.com/WangYueFt/prnet
[UTOPIC-main]: https://github.com/ZhileiChen99/UTOPIC
