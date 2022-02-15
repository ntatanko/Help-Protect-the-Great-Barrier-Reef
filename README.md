# TensorFlow - Help Protect the Great Barrier Reef
https://www.kaggle.com/c/tensorflow-great-barrier-reef/overview   

## Goal of the Competition
The goal of this competition is to accurately identify starfish in real-time by building an object detection model trained on underwater videos of coral reefs.  

Your work will help researchers identify species that are threatening Australia's Great Barrier Reef and take well-informed action to protect the reef for future generations.  

### Context
Australia's stunningly beautiful Great Barrier Reef is the world’s largest coral reef and home to 1,500 species of fish, 400 species of corals, 130 species of sharks, rays, and a massive variety of other sea life.

Unfortunately, the reef is under threat, in part because of the overpopulation of one particular starfish – the coral-eating crown-of-thorns starfish (or COTS for short). Scientists, tourism operators and reef managers established a large-scale intervention program to control COTS outbreaks to ecologically sustainable levels.

To know where the COTS are, a traditional reef survey method, called "Manta Tow", is performed by a snorkel diver. While towed by a boat, they visually assess the reef, stopping to record variables observed every 200m. While generally effective, this method faces clear limitations, including operational scalability, data resolution, reliability, and traceability.  

The Great Barrier Reef Foundation established an innovation program to develop new survey and intervention methods to provide a step change in COTS Control. Underwater cameras will collect thousands of reef images and AI technology could drastically improve the efficiency and scale at which reef managers detect and control COTS outbreaks.  

To scale up video-based surveying systems, Australia’s national science agency, CSIRO has teamed up with Google to develop innovative machine learning technology that can analyse large image datasets accurately, efficiently, and in near real-time.   

## Solution

Solution includes ensemble of YOLOv5 models trained on different data with high resolution with [Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) and tracking with [norfair library](https://github.com/tryolabs/norfair). Different hyperparameter tuning with albumentation tuning, different train test split approaches, norfair tuning.  
During experiments several models [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4) were trained, but the result was worse.   
Also was tried approach with classification for 'starfish' and 'no starfish' using [efiicientnet pytorch](https://github.com/lukemelas/EfficientNet-PyTorch), but it didn't affect the result.  
Measure f2_score during training (with changing yolov5 training code) didn't affect the result.  
Since there was data disbalance (images with starfishes were 4.77 times less than images with), one of the approaches includes choosing validation dataset with balance of images close to train dataset, including number of labeled images and total number of starfishes. The goal was to make ensemble, that works well not only on public test, but on the whole test set. Since the result on the public test is close to the result on the hidden test, this was achieved.



<details> 
    <summary>
        <b>Acknowledgements</b>
    </summary>

https://github.com/ultralytics/yolov5  
    
https://github.com/tryolabs/norfair   

https://github.com/ZFTurbo/Weighted-Boxes-Fusion  
    
    
```
@article{solovyev2021weighted,
  title={Weighted boxes fusion: Ensembling boxes from different object detection models},
  author={Solovyev, Roman and Wang, Weimin and Gabruseva, Tatiana},
  journal={Image and Vision Computing},
  pages={1-6},
  year={2021},
  publisher={Elsevier}
}
```
    
    
https://github.com/WongKinYiu/ScaledYOLOv4   
```
    @InProceedings{Wang_2021_CVPR,  
    author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},  
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},  
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
    month     = {June},  
    year      = {2021},  
    pages     = {13029-13038}  
}
 ```  

    
https://github.com/lukemelas/EfficientNet-PyTorch  



</details>

