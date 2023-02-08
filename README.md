# OptonicsObjectReconstruction
[![Contributor](https://img.shields.io/badge/contributor-cracee-brightgreen.svg)](https://github.com/Cracee)
[![Project](https://img.shields.io/badge/Project%20under%20supervision%20of-OPTONIC-blue.svg)](https://www.optonic.com/)

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

The `OMNet_Pytorch-main` contains the Deep Learning implementation of [OMNet] from their official github [OMNet_Pytorch]. It is already transformed to fit our needs with new dataloaders that can load out own 3d models.

The `prnet-master` contains the implementation of [PRNet] from their official github [prnet-master]. It is not working with newer versions of CUDA and runs into NaN Bugs. Therefore we changed to using `learning3d`.

The `UTOPIC-main` folders contain another DL approach, which I haven't worked on yet. It is the official implementation of [UTOPIC] from their github [UTOPIC-main]. Soon to come. 

In `learning3d` there is a whole library of DL Nets to use, but we only need the [PRNet] implementation, that is way more cleaner and doesen't throw errors like crazy. It comes from the official [learning3d] github repo.

<details><summary>Do not click to open</summary>
  <ul>
    <li>Got you!</li>
    <li>Now you probably feel ashamed</li>
    <li>But that is ok</li>
    <li>I would have clicked it too</li>
    <li>Don't worry, be happy!</li>
  </ul>
</details>

[OMNet_Pytorch]: https://github.com/hxwork/OMNet_Pytorch
[OMNet]: https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_OMNet_Learning_Overlapping_Mask_for_Partial-to-Partial_Point_Cloud_Registration_ICCV_2021_paper.pdf
[prnet-master]: https://github.com/WangYueFt/prnet
[PRNet]: https://arxiv.org/pdf/1910.12240.pdf
[UTOPIC-main]: https://github.com/ZhileiChen99/UTOPIC
[learning3d]: https://github.com/vinits5/learning3d
[UTOPIC]: https://arxiv.org/pdf/2208.02712.pdf
