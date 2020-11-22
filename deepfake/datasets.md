# 数据集

| dataset | publish date | conference | author |
| - | - | - | - |
| [FF++(FaceForensics++)](https://github.com/ondyari/FaceForensics) | 25 Jan 2019 | iccv19 | Andreas R¨ossler(Technical University of Munich) | 
| [Celeb-DF](http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html) | 27 Sep 2019 | cpvr20 | Yuezun Li(University at Albany) |
| [DeeperForensics-1.0](https://github.com/EndlessSora/DeeperForensics-1.0) | 9 Jan 2020 | cvpr20 | Liming Jiang(NTU Singapore), wuwenyan(sensetime) |
| [DFDC](https://ai.facebook.com/datasets/dfdc) | 12 Jun 2020 |  | Brian Dolhansky(Facebook AI) |

## FF++

[benchmark results for the Binary Classification](http://kaldir.vc.in.tum.de/faceforensics_benchmark/)

```
root-dir:
    manipulated_sequences:
        DeepFakeDetection
            c23/videos # 3068个假视频，文件名包含了替换方式
                <target actor>_<source actor>__<sequence name>__<8 charactor long experiment id>.mp4
            mask/videos: # 和videos相对应
        Deepfakes|Face2Face|FaceSwap|NeuralTextures
            c23/videos # 1000个视频，一对之间相互交换，四种变换均相同
                <target sequence>_<source sequence>.mp4 # source 提供人脸，target提供背景
            mask/videos: # 和videos相对应
    original_sequences:
        actors/c23/videos # 16个不同场景中28位付费演员的363个原始序列
            <actor number>__<scene name>.mp4
        youtube/c23/videos # 1000个视频
            000.mp4~999.mp4
```

## Celeb-DF

```
root-dir:
    YouTube-real:
        00000.mp4~00299.mp4 # 共300个视频，额外的youtube下载真视频
    Celeb-real:
        id*.mp4 # youtube收集的590个原始视频
    Celeb-synthesis:
        *.mp4 # Celeb-real中的对应的5639个假视频
    List_of_testing_videos.txt # 518个视频路径
```

## DeeperForensics-1.0

[DeeperForensics Challenge 2020 @ ECCV SenseHuman Workshop](https://competitions.codalab.org/competitions/25228)

```
DeeperForensics-1.0
|--lists
   |--manipulated_videos_distortions_meta
      <several meta files of distortion information in the manipulated videos>
   |--manipulated_videos_lists
      <several lists of the manipulated videos>
   |--source_videos_lists
      <several lists of the source videos>
   |--splits
      <train, val, test data splits for your reference>
|--manipulated_videos
   <11 folders named by the type of variants for the manipulated videos>
|--source_videos
   <100 folders named by the ID of the actors for the source videos, and their subfolders named by the source video information>
|--terms_of_use.pdf
```

## DFDC

```
root-dir:
    part0:
        metadata.json # 保存真假视频对应信息
        *.mp4 # 10s的视频
    part1:
    ……
    part49:
```