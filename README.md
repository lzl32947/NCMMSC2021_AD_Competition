# NCMMSC2021

## Introduction
This is the repo for NCMMSC2021 competition.

Notice that the ```master``` branch is for development, for stable release, please switch to the ```stable``` branch.

## Project Structure
```
NCMMSC2021
├─bin               # Contains the runnable scripts
├─configs           # Contains the configiurations
├─dataset           # Contains the dataset
│  ├─merge          # Concat all the audios from one person
│  │  ├─AD
│  │  ├─HC
│  │  └─MCI
│  ├─raw            # Raw audios 
│  │  ├─AD
│  │  ├─HC
│  │  └─MCI
│  ├─merge_vad      # Perform unsupervised VAD on the separated audios and concat the results
│  │  ├─AD
│  │  ├─HC
│  │  └─MCI
│  └─raw_vad        # Perform unsupervised VAD on raw audios
│      ├─AD
│      ├─HC
│      └─MCI
├─log               # Contains the log files
├─model             # Contains the main model
│  ├─models         # Contains all the model
│  └─modules        # Contains all the modules
├─weight            # Contains the weight files
└─util              # Contains the util files
   ├─log_util       # Utils for log
   ├─tool           # Useful tools for drawing and files
   ├─train_util     # Dataloader and trainer
   └─model_util     # Utils for networks
```

## Target Approach

There are two given tasks, predicting on 5 seconds audio and on 30 seconds audio separately

* For both, extract features (MFCC, Spectrogram and MelSpectrogram) from the audio and treat them with the Image-based Classification methods.
* LSTM is introduced into the model, however, not performing well.
* Other fusion methods like [Feature Fusion](model/modules/cam.py) are also tested but not work well in feature fusion than concat.

## Model Performance

| ID | Sample Seconds | Model | Use Feature | K-fold |Accuracy | Remark |
| :----: | :----: | :----: |:----: |:----: |:---- |:----: |
|[20210923_230628](log/20210903_230628)| 5s|SpecificTrainModel|MFCC|4|[75.91%](weight/20210903_230628/MFCC/fold0_4-epoch20-loss0.06089490287020003-acc0.7590725806451613.pth),[63.10%](weight/20210903_230628/MFCC/fold1_4-epoch9-loss0.09261391252603701-acc0.6310483870967742.pth),[76.21%](weight/20210903_230628/MFCC/fold2_4-epoch9-loss0.12377177163252352-acc0.7620967741935484.pth),[68.23%](weight/20210903_230628/MFCC/fold3_4-epoch3-loss0.3631502442001151-acc0.6822916666666666.pth)| |
|[20210923_230628](log/20210903_230628)| 5s|SpecificTrainModel|SPECS|4|[71.47%](weight/20210903_230628/Spectrogram/fold0_4-epoch3-loss0.18052627741075333-acc0.7147177419354839.pth),[59.78%](weight/20210903_230628/Spectrogram/fold1_4-epoch15-loss0.022810556306768516-acc0.5977822580645161.pth),[77.42%](weight/20210903_230628/Spectrogram/fold2_4-epoch15-loss0.0477966607046631-acc0.7741935483870968.pth),[62.50%](weight/20210903_230628/Spectrogram/fold3_4-epoch6-loss0.13587625618548332-acc0.625.pth)| |
|[20210923_230628](log/20210903_230628)| 5s|SpecificTrainModel|MELSPEC|4|[71.77%](weight/20210903_230628/MelSpectrogram/fold0_4-epoch17-loss0.032558338473584865-acc0.717741935483871.pth),[54.74%](weight/20210903_230628/MelSpectrogram/fold1_4-epoch19-loss0.021754477738892733-acc0.5473790322580645.pth),[78.73%](weight/20210903_230628/MelSpectrogram/fold2_4-epoch9-loss0.0651434302929813-acc0.7872983870967742.pth),[64.69%](weight/20210903_230628/MelSpectrogram/fold3_4-epoch19-loss0.029134586646059887-acc0.646875.pth)| |
|[20210904_141710](log/20210904_141710)| 5s|MSMJointConcatFineTuneModel|General|4|[75.60%](weight/20210904_141710/General/fold0_4-epoch8-loss0.1915400112553945-acc0.7560483870967742.pth),[69.15%](weight/20210904_141710/General/fold1_4-epoch10-loss0.10834520640175628-acc0.6915322580645161.pth),[77.22%](weight/20210904_141710/General/fold2_4-epoch19-loss0.04884094702909984-acc0.7721774193548387.pth),[73.96%](weight/20210904_141710/General/fold3_4-epoch17-loss0.06354183974044939-acc0.7395833333333334.pth)| MFCC,SPECS,MELSPEC for training |
|[20210904_141710](log/20210904_141710)| 5s|MSMJointConcatFineTuneModel|Fine-tune|4|[78.53%](weight/20210904_141710/Fine_tune/fold0_4-epoch14-loss0.015200369094521233-acc0.7852822580645161.pth),[68.25%](weight/20210904_141710/Fine_tune/fold1_4-epoch14-loss0.013524920946684173-acc0.6824596774193549.pth),[78.63%](weight/20210904_141710/Fine_tune/fold2_4-epoch19-loss0.004208964913864886-acc0.7862903225806451.pth),[75.00%](weight/20210904_141710/Fine_tune/fold3_4-epoch16-loss0.007014893440207997-acc0.75.pth)| MFCC,SPECS,MELSPEC for training |
|[20210904_150739](log/20210904_150739)| 5s|SpecificTrainResNetModel| MELSPEC| 4| [67.64%](weight/20210904_150739/MelSpectrogram/fold0_4-epoch15-loss0.0042680471195066984-acc0.6764112903225806.pth),[70.06%](weight/20210904_150739/MelSpectrogram/fold1_4-epoch12-loss0.019796290293312348-acc0.7006048387096774.pth),[72.18%](weight/20210904_150739/MelSpectrogram/fold2_4-epoch3-loss0.23754373382088606-acc0.7217741935483871.pth),[68.23%](weight/20210904_150739/MelSpectrogram/fold3_4-epoch9-loss0.03088850547685989-acc0.6822916666666666.pth)| |
|[20210904_215820](log/20210904_215820)| 25s|SpecificTrainResNetLongLSTMModel|MELSPEC|4|[65.32%](weight/20210904_215820/MelSpectrogram/fold0_4-epoch17-loss0.017942649561598006-acc0.6532258064516129.pth),[57.46%](weight/20210904_215820/MelSpectrogram/fold1_4-epoch19-loss0.0057612667840895365-acc0.5745967741935484.pth),[65.73%](weight/20210904_215820/MelSpectrogram/fold2_4-epoch19-loss0.03588582380198995-acc0.657258064516129.pth),[72.29%](weight/20210904_215820/MelSpectrogram/fold3_4-epoch19-loss0.054043575335213895-acc0.7229166666666667.pth)| |
|[20210904_234029](log/20210904_234029)| 25s|SpecificTrainResNetLongModel|MELSPEC|4|[77.62%](weight/20210904_234029/MelSpectrogram/fold0_4-epoch15-loss0.0005298890865880733-acc0.7762096774193549.pth),[59.07%](weight/20210904_234029/MelSpectrogram/fold1_4-epoch16-loss7.291974601726466e-05-acc0.5907258064516129.pth),[64.52%](weight/20210904_234029/MelSpectrogram/fold2_4-epoch13-loss0.0773518512467233-acc0.6451612903225806.pth),[72.50%](weight/20210904_234029/MelSpectrogram/fold3_4-epoch18-loss0.04882786973404128-acc0.725.pth)| |
|[20210905_151007](log/20210905_151007)| 25s|SpecificTrainLongLSTMModel|MELSPEC|4|[73.49%](weight/20210905_151007/MelSpectrogram/fold0_4-epoch14-loss0.11516540292043077-acc0.7348790322580645.pth),[61.09%](weight/20210905_151007/MelSpectrogram/fold1_4-epoch11-loss0.2968559519428274-acc0.6108870967741935.pth),[75.40%](weight/20210905_151007/MelSpectrogram/fold2_4-epoch13-loss0.14352692384272814-acc0.7540322580645161.pth),[65.10%](weight/20210905_151007/MelSpectrogram/fold3_4-epoch13-loss0.19081749549756447-acc0.6510416666666666.pth) | |


