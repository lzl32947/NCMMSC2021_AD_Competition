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

| ID | Sample Seconds | Model | Use Feature | K-fold |Accuracy |Train Average Acc| Remark | Evaluation|
| :----: | :----: | :----: |:----: |:----: |:---- |:---- |:----: |:----: |
|[20210903_230628](log/20210903_230628)| 5s|SpecificTrainModel|MFCC|4|[75.91%](weight/20210903_230628/MFCC/fold0_4-epoch20-loss0.06089490287020003-acc0.7590725806451613.pth),[63.10%](weight/20210903_230628/MFCC/fold1_4-epoch9-loss0.09261391252603701-acc0.6310483870967742.pth),[76.21%](weight/20210903_230628/MFCC/fold2_4-epoch9-loss0.12377177163252352-acc0.7620967741935484.pth),[68.23%](weight/20210903_230628/MFCC/fold3_4-epoch3-loss0.3631502442001151-acc0.6822916666666666.pth)| 68.36%||
|[20210903_230628](log/20210903_230628)| 5s|SpecificTrainModel|SPECS|4|[71.47%](weight/20210903_230628/Spectrogram/fold0_4-epoch3-loss0.18052627741075333-acc0.7147177419354839.pth),[59.78%](weight/20210903_230628/Spectrogram/fold1_4-epoch15-loss0.022810556306768516-acc0.5977822580645161.pth),[77.42%](weight/20210903_230628/Spectrogram/fold2_4-epoch15-loss0.0477966607046631-acc0.7741935483870968.pth),[62.50%](weight/20210903_230628/Spectrogram/fold3_4-epoch6-loss0.13587625618548332-acc0.625.pth)|67.79%| | |
|[20210903_230628](log/20210903_230628)| 5s|SpecificTrainModel|MELSPEC|4|[71.77%](weight/20210903_230628/MelSpectrogram/fold0_4-epoch17-loss0.032558338473584865-acc0.717741935483871.pth),[54.74%](weight/20210903_230628/MelSpectrogram/fold1_4-epoch19-loss0.021754477738892733-acc0.5473790322580645.pth),[78.73%](weight/20210903_230628/MelSpectrogram/fold2_4-epoch9-loss0.0651434302929813-acc0.7872983870967742.pth),[64.69%](weight/20210903_230628/MelSpectrogram/fold3_4-epoch19-loss0.029134586646059887-acc0.646875.pth)| 67.48%||
|[20210904_141710](log/20210904_141710)| 5s|MSMJointConcatFineTuneModel|General|4|[75.60%](weight/20210904_141710/General/fold0_4-epoch8-loss0.1915400112553945-acc0.7560483870967742.pth),[69.15%](weight/20210904_141710/General/fold1_4-epoch10-loss0.10834520640175628-acc0.6915322580645161.pth),[77.22%](weight/20210904_141710/General/fold2_4-epoch19-loss0.04884094702909984-acc0.7721774193548387.pth),[73.96%](weight/20210904_141710/General/fold3_4-epoch17-loss0.06354183974044939-acc0.7395833333333334.pth)| 71.48%|MFCC,SPECS,MELSPEC for training |
|[20210904_141710](log/20210904_141710)| 5s|MSMJointConcatFineTuneModel|Fine-tune|4|[78.53%](weight/20210904_141710/Fine_tune/fold0_4-epoch14-loss0.015200369094521233-acc0.7852822580645161.pth),[68.25%](weight/20210904_141710/Fine_tune/fold1_4-epoch14-loss0.013524920946684173-acc0.6824596774193549.pth),[78.63%](weight/20210904_141710/Fine_tune/fold2_4-epoch19-loss0.004208964913864886-acc0.7862903225806451.pth),[75.00%](weight/20210904_141710/Fine_tune/fold3_4-epoch16-loss0.007014893440207997-acc0.75.pth)| 75.10%|MFCC,SPECS,MELSPEC for training |
|[20210904_150739](log/20210904_150739)| 5s|SpecificTrainResNetModel| MELSPEC| 4| [67.64%](weight/20210904_150739/MelSpectrogram/fold0_4-epoch15-loss0.0042680471195066984-acc0.6764112903225806.pth),[70.06%](weight/20210904_150739/MelSpectrogram/fold1_4-epoch12-loss0.019796290293312348-acc0.7006048387096774.pth),[72.18%](weight/20210904_150739/MelSpectrogram/fold2_4-epoch3-loss0.23754373382088606-acc0.7217741935483871.pth),[68.23%](weight/20210904_150739/MelSpectrogram/fold3_4-epoch9-loss0.03088850547685989-acc0.6822916666666666.pth)| 69.53%||
|[20210915_093218](log/20210915_093218)| 5s|CompetitionSpecificTrainVggNet19BNBackboneModel| SPEC| 4|[70.36%](weight/20210915_093218/Spectrogram/fold0_4-epoch1-loss0.9654736254850159-acc0.7036290322580645.pth),[80.85%](weight/20210915_093218/Spectrogram/fold1_4-epoch17-loss0.003880349351826626-acc0.8084677419354839.pth),[83.67%](weight/20210915_093218/Spectrogram/fold2_4-epoch10-loss0.007162975656742871-acc0.8366935483870968.pth),[68.85%](weight/20210915_093218/Spectrogram/fold3_4-epoch13-loss0.007478766414307404-acc0.6885416666666667.pth)| 75.93%||
|[20210915_012356](log/20210915_012356)| 5s|CompetitionSpecificTrainVggNet19BNBackboneModel| MFCC| 4|[75.50%](weight/20210915_012356/MFCC/fold0_4-epoch3-loss0.13448772518440028-acc0.7550403225806451.pth),[63.41%](weight/20210915_012356/MFCC/fold1_4-epoch6-loss0.08252490379242733-acc0.6340725806451613.pth),[81.15%](weight/20210915_012356/MFCC/fold2_4-epoch10-loss0.02826608310354696-acc0.811491935483871.pth),[74.90%](weight/20210915_012356/MFCC/fold3_4-epoch9-loss0.011612757636553837-acc0.7489583333333333.pth)| 73.74%||
|[20210914_221835](log/20210914_221835)| 5s|CompetitionSpecificTrainVggNet19BNBackboneModel| MELSPEC| 4|[79.23%](weight/20210914_221835/MelSpectrogram/fold0_4-epoch8-loss0.0689652904411955-acc0.7923387096774194.pth),[75.40%](weight/20210914_221835/MelSpectrogram/fold1_4-epoch8-loss0.018455197155108377-acc0.7540322580645161.pth),[85.69%](weight/20210914_221835/MelSpectrogram/fold2_4-epoch10-loss0.019249525609251104-acc0.8568548387096774.pth),[62.81%](weight/20210914_221835/MelSpectrogram/fold3_4-epoch20-loss0.0007100844070611211-acc0.628125.pth)| 75.78%||
|[20210916_144512](log/20210916_144512)| 5s|CompetitionSpecificTrainResNet18BackboneModel| MFCC| 4|[69.96%](weight/20210916_144512/MFCC/fold0_4-epoch4-loss0.179108698037453-acc0.6995967741935484.pth),[72.08%](weight/20210916_144512/MFCC/fold1_4-epoch15-loss0.0018856231989977223-acc0.7207661290322581.pth),[76.71%](weight/20210916_144512/MFCC/fold2_4-epoch14-loss0.0013061160717054167-acc0.7671370967741935.pth),[61.04%](weight/20210916_144512/MFCC/fold3_4-epoch4-loss0.13942710617532134-acc0.6104166666666667.pth)| 69.92%||
|[20210917_154750](log/20210917_154750)| 5s|CompetitionSpecificTrainWideResNet| MELSPEC| 4|[77.52%](weight/20210917_155141/MelSpectrogram/fold0_4-epoch9-loss0.06866901246425898-acc0.7752016129032258.pth),[74.80%](weight/20210917_155141/MelSpectrogram/fold1_4-epoch4-loss0.5764459543254065-acc0.7479838709677419.pth),[78.02%](weight/20210917_155141/MelSpectrogram/fold2_4-epoch7-loss0.19566268546749715-acc0.780241935483871.pth),[55.73%](weight/20210917_155141/MelSpectrogram/fold3_4-epoch9-loss0.07832202158107232-acc0.5572916666666666.pth)| 71.51%||
|[20210917_154750](log/20210917_154750)| 5s|CompetitionSpecificTrainVggNet16BNBackboneModel| MELSPEC| 4|[76.81%](weight/20210917_154750/MelSpectrogram/fold0_4-epoch15-loss0.007421083951621323-acc0.7681451612903226.pth),[79.94%](weight/20210917_154750/MelSpectrogram/fold1_4-epoch8-loss0.0483754780209932-acc0.7993951612903226.pth),[79.64%](weight/20210917_154750/MelSpectrogram/fold2_4-epoch8-loss0.023448249723890553-acc0.7963709677419355.pth),[63.12%](weight/20210917_154750/MelSpectrogram/fold3_4-epoch13-loss0.0006954660123982091-acc0.63125.pth)| 74.87%||
|[20210904_215820](log/20210904_215820)| 25s|SpecificTrainResNetLongLSTMModel|MELSPEC|4|[65.32%](weight/20210904_215820/MelSpectrogram/fold0_4-epoch17-loss0.017942649561598006-acc0.6532258064516129.pth),[57.46%](weight/20210904_215820/MelSpectrogram/fold1_4-epoch19-loss0.0057612667840895365-acc0.5745967741935484.pth),[65.73%](weight/20210904_215820/MelSpectrogram/fold2_4-epoch19-loss0.03588582380198995-acc0.657258064516129.pth),[72.29%](weight/20210904_215820/MelSpectrogram/fold3_4-epoch19-loss0.054043575335213895-acc0.7229166666666667.pth)|65.20%| |[Detail](#longdetail1) [General](#longgeneral1)|
|[20210904_234029](log/20210904_234029)| 25s|SpecificTrainResNetLongModel|MELSPEC|4|[77.62%](weight/20210904_234029/MelSpectrogram/fold0_4-epoch15-loss0.0005298890865880733-acc0.7762096774193549.pth),[59.07%](weight/20210904_234029/MelSpectrogram/fold1_4-epoch16-loss7.291974601726466e-05-acc0.5907258064516129.pth),[64.52%](weight/20210904_234029/MelSpectrogram/fold2_4-epoch13-loss0.0773518512467233-acc0.6451612903225806.pth),[72.50%](weight/20210904_234029/MelSpectrogram/fold3_4-epoch18-loss0.04882786973404128-acc0.725.pth)| 68.43%| |[Detail](#longdetail2) [General](#longgeneral2)|
|[20210905_151007](log/20210905_151007)| 25s|SpecificTrainLongLSTMModel|MELSPEC|4|[73.49%](weight/20210905_151007/MelSpectrogram/fold0_4-epoch14-loss0.11516540292043077-acc0.7348790322580645.pth),[61.09%](weight/20210905_151007/MelSpectrogram/fold1_4-epoch11-loss0.2968559519428274-acc0.6108870967741935.pth),[75.40%](weight/20210905_151007/MelSpectrogram/fold2_4-epoch13-loss0.14352692384272814-acc0.7540322580645161.pth),[65.10%](weight/20210905_151007/MelSpectrogram/fold3_4-epoch13-loss0.19081749549756447-acc0.6510416666666666.pth) |68.77%| |[Detail](#longdetail3) [General](#longgeneral3)|
|[20210905_130825](log/20210905_130825)| 25s|SpecificTrainLongModel|MELSPEC|4|[78.23%](weight/20210905_130825/MelSpectrogram/fold0_4-epoch4-loss0.04840035374661017-acc0.782258064516129.pth),[59.98%](weight/20210905_130825/MelSpectrogram/fold1_4-epoch2-loss0.09746426020485713-acc0.5997983870967742.pth),[78.63%](weight/20210905_130825/MelSpectrogram/fold2_4-epoch17-loss0.0036692070889725787-acc0.7862903225806451.pth),[66.35%](weight/20210905_130825/MelSpectrogram/fold3_4-epoch2-loss0.14088858466755638-acc0.6635416666666667.pth)|70.79%| | [Detail](#longdetail4) [General](#longgeneral4)|
|[20210905_133648](log/20210905_133648)| 25s|SpecificTrainLongModel|SPECS|4|[70.97%](weight/20210905_133648/Spectrogram/fold0_4-epoch17-loss0.005109133508401852-acc0.7096774193548387.pth),[58.17%](weight/20210905_133648/Spectrogram/fold1_4-epoch7-loss0.009974943350560194-acc0.5816532258064516.pth),[76.41%](weight/20210905_133648/Spectrogram/fold2_4-epoch2-loss0.14389855253672146-acc0.7641129032258065.pth),[66.88%](weight/20210905_133648/Spectrogram/fold3_4-epoch5-loss0.0316563960589138-acc0.66875.pth)| 68.11%| |[Detail](#longdetail5) [General](#longgeneral5)|
|[20210905_133648](log/20210905_133648)| 25s|SpecificTrainLongModel|MFCC|4|[73.19%](weight/20210905_133648/MFCC/fold0_4-epoch1-loss0.6708392670944981-acc0.7318548387096774.pth),[66.94%](weight/20210905_133648/MFCC/fold1_4-epoch18-loss0.011163503149399057-acc0.6693548387096774.pth),[76.41%](weight/20210905_133648/MFCC/fold2_4-epoch17-loss0.0059203855958596405-acc0.7641129032258065.pth),[70.21%](weight/20210905_133648/MFCC/fold3_4-epoch13-loss0.006423260154288953-acc0.7020833333333333.pth)| 71.68%| |[Detail](#longdetail6) [General](#longgeneral6)|
|[20210905_133648](log/20210905_133648)| 25s|SpecificTrainLongModel|MELSPEC|4|[78.23%](weight/20210905_133648/MelSpectrogram/fold0_4-epoch1-loss0.5207754814916331-acc0.782258064516129.pth),[59.17%](weight/20210905_133648/MelSpectrogram/fold1_4-epoch11-loss0.003643666341304197-acc0.5917338709677419.pth),[75.60%](weight/20210905_133648/MelSpectrogram/fold2_4-epoch10-loss0.007353401618393432-acc0.7560483870967742.pth),[63.75%](weight/20210905_133648/MelSpectrogram/fold3_4-epoch1-loss0.45186451201637584-acc0.6375.pth)| 68.19%| |[Detail](#longdetail7) [General](#longgeneral7)|
|[20210905_133648](log/20210905_133648)| 25s|MSMJointConcatFineTuneLongModel|General|4|[71.27%](weight/20210905_133648/General/fold0_4-epoch9-loss0.014396540212897968-acc0.7127016129032258.pth),[72.38%](weight/20210905_133648/General/fold1_4-epoch13-loss0.007122711696865736-acc0.7237903225806451.pth),[79.64%](weight/20210905_133648/General/fold2_4-epoch11-loss0.006662264470081857-acc0.7963709677419355.pth),[72.40%](weight/20210905_133648/General/fold3_4-epoch6-loss0.054713346807646654-acc0.7239583333333334.pth)| 73.92%| MFCC,SPECS,MELSPEC for training|[Detail](#longdetail8) [General](#longgeneral8)|
|[20210905_133648](log/20210905_133648)| 25s|MSMJointConcatFineTuneLongModel|Fine-tune|4|[73.29%](weight/20210905_133648/Fine_tune/fold0_4-epoch3-loss0.006180769617260566-acc0.7328629032258065.pth),[64.21%](weight/20210905_133648/Fine_tune/fold1_4-epoch2-loss0.012040591682307422-acc0.6421370967741935.pth),[79.94%](weight/20210905_133648/Fine_tune/fold2_4-epoch12-loss0.0006443048127948714-acc0.7993951612903226.pth),[74.79%](weight/20210905_133648/Fine_tune/fold3_4-epoch14-loss0.000780794843008788-acc0.7479166666666667.pth)| 73.06%| MFCC,SPECS,MELSPEC for training|[Detail](#longdetail9) [General](#longgeneral9)|
|[20210906_215527](log/20210906_215527)| 25s|SpecificTrainLongModel|MELSPEC_VAD|4|[68.45%](weight/20210906_215527/MelSpectrogram_VAD/fold0_4-epoch20-loss0.005048008708004288-acc0.6844758064516129.pth),[66.13%](weight/20210906_215527/MelSpectrogram_VAD/fold1_4-epoch4-loss0.11396426730789244-acc0.6612903225806451.pth),[68.85%](weight/20210906_215527/MelSpectrogram_VAD/fold2_4-epoch17-loss0.016430959921189755-acc0.688508064516129.pth),[73.12%](weight/20210906_215527/MelSpectrogram_VAD/fold3_4-epoch20-loss0.011851182989591497-acc0.73125.pth)| 69.14%| | [Detail](#longdetail10) [General](#longgeneral10)|
|[20210906_185221](log/20210906_185221)| 25s|SpecificTrainLongTransformerEncoderModel|MELSPEC|4|[67.94%](weight/20210906_185221/MelSpectrogram/fold0_4-epoch3-loss0.15268928052872702-acc0.6794354838709677.pth),[65.02%](weight/20210906_185221/MelSpectrogram/fold1_4-epoch10-loss0.030885360165800946-acc0.6502016129032258.pth),[74.40%](weight/20210906_185221/MelSpectrogram/fold2_4-epoch11-loss0.000771018819744748-acc0.7439516129032258.pth),[69.06%](weight/20210906_185221/MelSpectrogram/fold3_4-epoch8-loss0.05979630712814684-acc0.690625.pth)| 69.11%| | [Detail](#longdetail11) [General](#longgeneral11)|
|[20210908_121607](log/20210908_121607)| 25s|SpecificTrainResNet18BackboneLongModel|MELSPEC_VAD|4|[70.46%](weight/20210908_121607/MelSpectrogram_VAD/fold0_4-epoch8-loss0.0007008382593322376-acc0.7046370967741935.pth),[65.83%](weight/20210908_121607/MelSpectrogram_VAD/fold1_4-epoch4-loss0.009402812518680508-acc0.6582661290322581.pth),[79.54%](weight/20210908_121607/MelSpectrogram_VAD/fold2_4-epoch2-loss0.10287831783669231-acc0.7953629032258065.pth),[64.79%](weight/20210908_121607/MelSpectrogram_VAD/fold3_4-epoch6-loss0.002103926696035678-acc0.6479166666666667.pth)|73.77%| |[Detail](#longdetail12) [General](#longgeneral12)|
|[20210907_230640](log/20210907_230640)| 25s|MSMJointConcatFineTuneLongModel|General|4|[80.04%](weight/20210907_230640/General/fold0_4-epoch17-loss0.004639386917282853-acc0.8004032258064516.pth),[63.61%](weight/20210907_230640/General/fold1_4-epoch6-loss0.05025758121527084-acc0.6360887096774194.pth),[76.51%](weight/20210907_230640/General/fold2_4-epoch8-loss0.020449309752186073-acc0.7651209677419355.pth),[74.90%](weight/20210907_230640/General/fold3_4-epoch5-loss0.10975683429929357-acc0.7489583333333333.pth) | 73.92%| MFCC,SPECS,MELSPEC for training|[Detail](#longdetail13) [General](#longgeneral13)|
|[20210907_230640](log/20210907_230640)| 25s|MSMJointConcatFineTuneLongModel|Fine-tune|4|[77.42%](weight/20210907_230640/Fine_tune/fold0_4-epoch16-loss0.00031449106786171995-acc0.7741935483870968.pth),[65.12%](weight/20210907_230640/Fine_tune/fold1_4-epoch5-loss0.0026477884619690367-acc0.6512096774193549.pth),[76.11%](weight/20210907_230640/Fine_tune/fold2_4-epoch14-loss0.0035135104489318505-acc0.7610887096774194.pth),[74.79%](weight/20210907_230640/Fine_tune/fold3_4-epoch4-loss0.007959074192112111-acc0.7479166666666667.pth) | 73.36%| MFCC,SPECS,MELSPEC for training|[Detail](#longdetail4) [General](#longgeneral14)|
|[20210907_230704](log/20210907_230704)| 25s|SpecificTrainLongModel|MELSPEC_VAD|4|[68.15%](weight/20210907_230704/MelSpectrogram_VAD/fold0_4-epoch10-loss0.03768929171138038-acc0.6814516129032258.pth),[64.01%](weight/20210907_230704/MelSpectrogram_VAD/fold1_4-epoch15-loss0.007877903156954942-acc0.6401209677419355.pth),[69.15%](weight/20210907_230704/MelSpectrogram_VAD/fold2_4-epoch11-loss0.013715556814175849-acc0.6915322580645161.pth),[70.21%](weight/20210907_230704/MelSpectrogram_VAD/fold3_4-epoch10-loss0.028328277380956758-acc0.7020833333333333.pth)| 67.88%| | [Detail](#longdetail5) [General](#longgeneral15)|
|[20210907_230704](log/20210907_230704)| 25s|SpecificTrainLongModel|SPECS_VAD|4|[70.87%](weight/20210907_230704/Spectrogram_VAD/fold0_4-epoch2-loss0.3224336070453991-acc0.7086693548387096.pth),[68.65%](weight/20210907_230704/Spectrogram_VAD/fold1_4-epoch6-loss0.056907686666818336-acc0.686491935483871.pth),[64.82%](weight/20210907_230704/Spectrogram_VAD/fold2_4-epoch20-loss0.007898257525242947-acc0.6481854838709677.pth),[71.25%](weight/20210907_230704/Spectrogram_VAD/fold3_4-epoch19-loss0.008848778591478763-acc0.7125.pth)| 68.90%| | [Detail](#longdetail16) [General](#longgeneral16)|
|[20210907_230704](log/20210907_230704)| 25s|SpecificTrainLongModel|MFCC_VAD|4|[67.94%](weight/20210907_230704/MFCC_VAD/fold0_4-epoch13-loss0.0171057377238353-acc0.6794354838709677.pth),[63.00%](weight/20210907_230704/MFCC_VAD/fold1_4-epoch3-loss0.2565434914406227-acc0.6300403225806451.pth),[69.15%](weight/20210907_230704/MFCC_VAD/fold2_4-epoch9-loss0.034415889523781676-acc0.6915322580645161.pth),[64.27%](weight/20210907_230704/MFCC_VAD/fold3_4-epoch6-loss0.10468367105149613-acc0.6427083333333333.pth) |66.09% | | [Detail](#longdetail17) [General](#longgeneral17)|
|[20210907_230704](log/20210907_230704)| 25s|MSMJointConcatFineTuneLongModel|General|4|[71.37%](weight/20210907_230704/General/fold0_4-epoch17-loss0.021520300520933233-acc0.7137096774193549.pth),[62.50%](weight/20210907_230704/General/fold1_4-epoch14-loss0.03671111866930479-acc0.625.pth),[67.04%](weight/20210907_230704/General/fold2_4-epoch20-loss0.018662136247376507-acc0.6703629032258065.pth),[64.90%](weight/20210907_230704/General/fold3_4-epoch14-loss0.05942454429403428-acc0.6489583333333333.pth)| 66.45%| MFCC_VAD, SPECS_VAD and MELSPEC_VAD for training|[Detail](#longdetail18) [General](#longgeneral18)|
|[20210907_230704](log/20210907_230704)| 25s|MSMJointConcatFineTuneLongModel|Fine-tune|4|[67.04%](weight/20210907_230704/Fine_tune/fold0_4-epoch19-loss0.0006361513321142253-acc0.6703629032258065.pth),[66.73%](weight/20210907_230704/Fine_tune/fold1_4-epoch12-loss0.018077083243003035-acc0.6673387096774194.pth),[69.15%](weight/20210907_230704/Fine_tune/fold2_4-epoch4-loss0.009594945272945788-acc0.6915322580645161.pth),[66.77%](weight/20210907_230704/Fine_tune/fold3_4-epoch9-loss0.012511203791917225-acc0.6677083333333333.pth)| 67.42%| MFCC_VAD, SPECS_VAD and MELSPEC_VAD for training|[Detail](#longdetail19) [General](#longgeneral19)|


## Evaluation

### Details

* [20210904_215820](log/20210904_215820)  <span id="longdetail1">SpecificTrainResNetLongLSTMModel</span> with MELSPEC

    ![SpecificTrainResNetLongLSTMModel](image/20210907_184826/SpecificTrainResNetLongLSTMModel_0-4_Fold_Results_Accuracy_60.50_Percent.png)

    ![SpecificTrainResNetLongLSTMModel](image/20210907_184826/SpecificTrainResNetLongLSTMModel_1-4_Fold_Results_Accuracy_51.90_Percent.png)

    ![SpecificTrainResNetLongLSTMModel](image/20210907_184826/SpecificTrainResNetLongLSTMModel_2-4_Fold_Results_Accuracy_65.46_Percent.png)

    ![SpecificTrainResNetLongLSTMModel](image/20210907_184826/SpecificTrainResNetLongLSTMModel_3-4_Fold_Results_Accuracy_57.75_Percent.png)



* [20210904_234029](log/20210904_234029) <span id="longdetail2">SpecificTrainResNetLongModel</span> with MELSPEC

    ![SpecificTrainResNetLongModel](image/20210907_192320/SpecificTrainResNetLongModel_0-4_Fold_Results_Accuracy_76.85_Percent.png)

    ![SpecificTrainResNetLongModel](image/20210907_192320/SpecificTrainResNetLongModel_1-4_Fold_Results_Accuracy_58.41_Percent.png)

    ![SpecificTrainResNetLongModel](image/20210907_192320/SpecificTrainResNetLongModel_2-4_Fold_Results_Accuracy_61.85_Percent.png)

    ![SpecificTrainResNetLongModel](image/20210907_192320/SpecificTrainResNetLongModel_3-4_Fold_Results_Accuracy_70.69_Percent.png)



* [20210905_151007](log/20210905_151007)  <span id="longdetail3">SpecificTrainLongLSTMModel</span> with MELSPEC

    ![SpecificTrainLongLSTMModel](image/20210907_212203/SpecificTrainLongLSTMModel_0-4_Fold_Results_Accuracy_73.53_Percent.png)

    ![SpecificTrainLongLSTMModel](image/20210907_212203/SpecificTrainLongLSTMModel_1-4_Fold_Results_Accuracy_59.94_Percent.png)

    ![SpecificTrainLongLSTMModel](image/20210907_212203/SpecificTrainLongLSTMModel_2-4_Fold_Results_Accuracy_75.36_Percent.png)

    ![SpecificTrainLongLSTMModel](image/20210907_212203/SpecificTrainLongLSTMModel_3-4_Fold_Results_Accuracy_64.58_Percent.png)


* [20210905_130825](log/20210905_130825)  <span id="longdetail4">SpecificTrainLongModel</span> with MELSPEC

    ![SpecificTrainLongModel](image/20210907_202132/SpecificTrainLongModel_0-4_Fold_Results_Accuracy_78.51_Percent.png)

    ![SpecificTrainLongModel](image/20210907_202132/SpecificTrainLongModel_1-4_Fold_Results_Accuracy_61.05_Percent.png)

    ![SpecificTrainLongModel](image/20210907_202132/SpecificTrainLongModel_2-4_Fold_Results_Accuracy_77.78_Percent.png)

    ![SpecificTrainLongModel](image/20210907_202132/SpecificTrainLongModel_3-4_Fold_Results_Accuracy_65.50_Percent.png)


* [20210905_133648](log/20210905_133648)  <span id="longdetail5">SpecificTrainLongModel</span> with SPEC

    ![SpecificTrainLongModel](image/20210907_183657/SpecificTrainLongModel_0-4_Fold_Results_Accuracy_70.91_Percent.png)

    ![SpecificTrainLongModel](image/20210907_183657/SpecificTrainLongModel_1-4_Fold_Results_Accuracy_57.42_Percent.png)

    ![SpecificTrainLongModel](image/20210907_183657/SpecificTrainLongModel_2-4_Fold_Results_Accuracy_75.52_Percent.png)

    ![SpecificTrainLongModel](image/20210907_183657/SpecificTrainLongModel_3-4_Fold_Results_Accuracy_66.56_Percent.png)


* [20210905_133648](log/20210905_133648)  <span id="longdetail6">SpecificTrainLongModel</span> with MFCC

    ![SpecificTrainLongModel](image/20210907_184705/SpecificTrainLongModel_0-4_Fold_Results_Accuracy_73.12_Percent.png)

    ![SpecificTrainLongModel](image/20210907_184705/SpecificTrainLongModel_1-4_Fold_Results_Accuracy_66.71_Percent.png)

    ![SpecificTrainLongModel](image/20210907_184705/SpecificTrainLongModel_2-4_Fold_Results_Accuracy_75.71_Percent.png)

    ![SpecificTrainLongModel](image/20210907_184705/SpecificTrainLongModel_3-4_Fold_Results_Accuracy_69.52_Percent.png)


* [20210905_133648](log/20210905_133648)  <span id="longdetail7">SpecificTrainLongModel</span> with MELSPEC

    ![SpecificTrainLongModel](image/20210907_192416/SpecificTrainLongModel_0-4_Fold_Results_Accuracy_77.36_Percent.png)

    ![SpecificTrainLongModel](image/20210907_192416/SpecificTrainLongModel_1-4_Fold_Results_Accuracy_58.71_Percent.png)

    ![SpecificTrainLongModel](image/20210907_192416/SpecificTrainLongModel_2-4_Fold_Results_Accuracy_74.90_Percent.png)

    ![SpecificTrainLongModel](image/20210907_192416/SpecificTrainLongModel_3-4_Fold_Results_Accuracy_63.06_Percent.png)


* [20210905_133648](log/20210905_133648)  <span id="longdetail8">MSMJointConcatFineTuneLongModel</span> with General

    ![MSMJointConcatFineTuneLongModel](image/20210907_192859/MSMJointConcatFineTuneLongModel_0-4_Fold_Results_Accuracy_70.83_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210907_192859/MSMJointConcatFineTuneLongModel_1-4_Fold_Results_Accuracy_71.63_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210907_192859/MSMJointConcatFineTuneLongModel_2-4_Fold_Results_Accuracy_78.61_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210907_192859/MSMJointConcatFineTuneLongModel_3-4_Fold_Results_Accuracy_72.56_Percent.png)


* [20210905_133648](log/20210905_133648)  <span id="longdetail9">MSMJointConcatFineTuneLongModel</span> with Fine-tune

    ![MSMJointConcatFineTuneLongModel](image/20210907_202227/MSMJointConcatFineTuneLongModel_0-4_Fold_Results_Accuracy_72.16_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210907_202227/MSMJointConcatFineTuneLongModel_1-4_Fold_Results_Accuracy_64.29_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210907_202227/MSMJointConcatFineTuneLongModel_2-4_Fold_Results_Accuracy_78.57_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210907_202227/MSMJointConcatFineTuneLongModel_3-4_Fold_Results_Accuracy_73.92_Percent.png)


* [20210906_215527](log/20210906_215527)  <span id="longdetail10">SpecificTrainLongModel</span> with MelSpectrogram_VAD

    ![SpecificTrainLongModel](image/20210907_175242/SpecificTrainLongModel_0-4_Fold_Results_with_Accuracy_67.88_Percent.png)

    ![SpecificTrainLongModel](image/20210907_175242/SpecificTrainLongModel_1-4_Fold_Results_with_Accuracy_66.23_Percent.png)

    ![SpecificTrainLongModel](image/20210907_175242/SpecificTrainLongModel_2-4_Fold_Results_with_Accuracy_67.78_Percent.png)

    ![SpecificTrainLongModel](image/20210907_175242/SpecificTrainLongModel_3-4_Fold_Results_with_Accuracy_73.38_Percent.png)


* [20210906_185221](log/20210906_185221)  <span id="longdetail11">SpecificTrainLongTransformerEncoderModel</span> with MELSPEC

    ![SpecificTrainLongTransformerEncoderModel](image/20210907_192510/SpecificTrainLongTransformerEncoderModel_0-4_Fold_Results_Accuracy_68.39_Percent.png)

    ![SpecificTrainLongTransformerEncoderModel](image/20210907_192510/SpecificTrainLongTransformerEncoderModel_1-4_Fold_Results_Accuracy_64.80_Percent.png)

    ![SpecificTrainLongTransformerEncoderModel](image/20210907_192510/SpecificTrainLongTransformerEncoderModel_2-4_Fold_Results_Accuracy_74.21_Percent.png)

    ![SpecificTrainLongTransformerEncoderModel](image/20210907_192510/SpecificTrainLongTransformerEncoderModel_3-4_Fold_Results_Accuracy_68.54_Percent.png)

* [20210908_121607](log/20210908_121607)  <span id="longdetail12">SpecificTrainResNet18BackboneLongModel</span> with MELSPEC_VAD

    ![SpecificTrainResNet18BackboneLongModel](image/20210908_155528/SpecificTrainResNet18BackboneLongModel_0-4_Fold_Results_Accuracy_69.42_Percent.png)

    ![SpecificTrainResNet18BackboneLongModel](image/20210908_155528/SpecificTrainResNet18BackboneLongModel_1-4_Fold_Results_Accuracy_66.11_Percent.png)

    ![SpecificTrainResNet18BackboneLongModel](image/20210908_155528/SpecificTrainResNet18BackboneLongModel_2-4_Fold_Results_Accuracy_77.56_Percent.png)

    ![SpecificTrainResNet18BackboneLongModel](image/20210908_155528/SpecificTrainResNet18BackboneLongModel_3-4_Fold_Results_Accuracy_64.73_Percent.png)

* [20210907_230640](log/20210907_230640)  <span id="longdetail13">MSMJointConcatFineTuneLongModel</span> with General

    ![MSMJointConcatFineTuneLongModel](image/20210908_235454/MSMJointConcatFineTuneLongModel_0-4_Fold_Results_Accuracy_78.91_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210908_235454/MSMJointConcatFineTuneLongModel_1-4_Fold_Results_Accuracy_62.52_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210908_235454/MSMJointConcatFineTuneLongModel_2-4_Fold_Results_Accuracy_76.11_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210908_235454/MSMJointConcatFineTuneLongModel_3-4_Fold_Results_Accuracy_75.19_Percent.png)

* [20210907_230640](log/20210907_230640)  <span id="longdetail14">MSMJointConcatFineTuneLongModel</span> with Fine-tune

    ![MSMJointConcatFineTuneLongModel](image/20210909_103922/MSMJointConcatFineTuneLongModel_0-4_Fold_Results_Accuracy_76.23_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210909_103922/MSMJointConcatFineTuneLongModel_1-4_Fold_Results_Accuracy_65.69_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210909_103922/MSMJointConcatFineTuneLongModel_2-4_Fold_Results_Accuracy_75.36_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210909_103922/MSMJointConcatFineTuneLongModel_3-4_Fold_Results_Accuracy_74.00_Percent.png)

* [20210907_230704](log/20210907_230704)  <span id="longdetail15">SpecificTrainLongModel</span> with MELSPEC_VAD

    ![SpecificTrainLongModel](image/20210909_121950/SpecificTrainLongModel_0-4_Fold_Results_Accuracy_67.18_Percent.png)

    ![SpecificTrainLongModel](image/20210909_121950/SpecificTrainLongModel_1-4_Fold_Results_Accuracy_64.70_Percent.png)

    ![SpecificTrainLongModel](image/20210909_121950/SpecificTrainLongModel_2-4_Fold_Results_Accuracy_68.93_Percent.png)

    ![SpecificTrainLongModel](image/20210909_121950/SpecificTrainLongModel_3-4_Fold_Results_Accuracy_69.96_Percent.png)

* [20210907_230704](log/20210907_230704)  <span id="longdetail16">SpecificTrainLongModel</span> with SPECS_VAD

    ![SpecificTrainLongModel](image/20210909_124246/SpecificTrainLongModel_0-4_Fold_Results_Accuracy_70.56_Percent.png)

    ![SpecificTrainLongModel](image/20210909_124246/SpecificTrainLongModel_1-4_Fold_Results_Accuracy_67.90_Percent.png)

    ![SpecificTrainLongModel](image/20210909_124246/SpecificTrainLongModel_2-4_Fold_Results_Accuracy_64.13_Percent.png)

    ![SpecificTrainLongModel](image/20210909_124246/SpecificTrainLongModel_3-4_Fold_Results_Accuracy_70.69_Percent.png)

* [20210907_230704](log/20210907_230704)  <span id="longdetail17">SpecificTrainLongModel</span> with MFCC_VAD

    ![SpecificTrainLongModel](image/20210909_130955/SpecificTrainLongModel_0-4_Fold_Results_Accuracy_67.26_Percent.png)

    ![SpecificTrainLongModel](image/20210909_130955/SpecificTrainLongModel_1-4_Fold_Results_Accuracy_62.96_Percent.png)

    ![SpecificTrainLongModel](image/20210909_130955/SpecificTrainLongModel_2-4_Fold_Results_Accuracy_68.65_Percent.png)

    ![SpecificTrainLongModel](image/20210909_130955/SpecificTrainLongModel_3-4_Fold_Results_Accuracy_64.44_Percent.png)

* [20210907_230704](log/20210907_230704)  <span id="longdetail18">MSMJointConcatFineTuneLongModel</span> with General VAD

    ![MSMJointConcatFineTuneLongModel](image/20210909_142350/MSMJointConcatFineTuneLongModel_0-4_Fold_Results_Accuracy_70.48_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210909_142350/MSMJointConcatFineTuneLongModel_1-4_Fold_Results_Accuracy_61.19_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210909_142350/MSMJointConcatFineTuneLongModel_2-4_Fold_Results_Accuracy_67.18_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210909_142350/MSMJointConcatFineTuneLongModel_3-4_Fold_Results_Accuracy_63.04_Percent.png)

* [20210907_230704](log/20210907_230704)  <span id="longdetail19">MSMJointConcatFineTuneLongModel</span> with Fine tune VAD

    ![MSMJointConcatFineTuneLongModel](image/20210909_152527/MSMJointConcatFineTuneLongModel_0-4_Fold_Results_Accuracy_66.57_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210909_152527/MSMJointConcatFineTuneLongModel_1-4_Fold_Results_Accuracy_66.83_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210909_152527/MSMJointConcatFineTuneLongModel_2-4_Fold_Results_Accuracy_68.97_Percent.png)

    ![MSMJointConcatFineTuneLongModel](image/20210909_152527/MSMJointConcatFineTuneLongModel_3-4_Fold_Results_Accuracy_66.23_Percent.png)

### General

* [20210904_215820](log/20210904_215820)  <span id="longgeneral1">SpecificTrainResNetLongLSTMModel</span> with MELSPEC

    ![SpecificTrainResNetLongLSTMModel](image/20210907_184826/SpecificTrainResNetLongLSTMModel_Results_Accuracy_58.91_Percent.png)


* [20210904_234029](log/20210904_234029) <span id="longgeneral2">SpecificTrainResNetLongModel</span> with MELSPEC

    ![SpecificTrainResNetLongModel](image/20210907_192320/SpecificTrainResNetLongModel_Results_Accuracy_66.92_Percent.png)


* [20210905_151007](log/20210905_151007)  <span id="longgeneral3">SpecificTrainLongLSTMModel</span> with MELSPEC

    ![SpecificTrainLongLSTMModel](image/20210907_212203/SpecificTrainLongLSTMModel_Results_Accuracy_68.38_Percent.png)

* [20210905_130825](log/20210905_130825)  <span id="longgeneral4">SpecificTrainLongModel</span> with MELSPEC

    ![SpecificTrainLongModel](image/20210907_202132/SpecificTrainLongModel_Results_Accuracy_70.75_Percent.png)



* [20210905_133648](log/20210905_133648)  <span id="longgeneral5">SpecificTrainLongModel</span> with SPEC

    ![SpecificTrainLongModel](image/20210907_183657/SpecificTrainLongModel_Results_Accuracy_67.61_Percent.png)


* [20210905_133648](log/20210905_133648)  <span id="longgeneral6">SpecificTrainLongModel</span> with MFCC

    ![SpecificTrainLongModel](image/20210907_184705/SpecificTrainLongModel_Results_Accuracy_71.28_Percent.png)

* [20210905_133648](log/20210905_133648)  <span id="longgeneral7">SpecificTrainLongModel</span> with MELSPEC

    ![SpecificTrainLongModel](image/20210907_192416/SpecificTrainLongModel_Results_Accuracy_68.55_Percent.png)


* [20210905_133648](log/20210905_133648)  <span id="longgeneral8">MSMJointConcatFineTuneLongModel</span> with General

    ![MSMJointConcatFineTuneLongModel](image/20210907_192859/MSMJointConcatFineTuneLongModel_Results_Accuracy_73.41_Percent.png)


* [20210905_133648](log/20210905_133648)  <span id="longgeneral9">MSMJointConcatFineTuneLongModel</span> with Fine-tune

    ![MSMJointConcatFineTuneLongModel](image/20210907_202227/MSMJointConcatFineTuneLongModel_Results_Accuracy_72.22_Percent.png)


* [20210906_215527](log/20210906_215527)  <span id="longgeneral10">SpecificTrainLongModel</span> with MelSpectrogram_VAD

    ![SpecificTrainLongModel](image/20210907_175242/SpecificTrainLongModel_Results_with_Accuracy_68.78_Percent.png)


* [20210906_185221](log/20210906_185221)  <span id="longgeneral11">SpecificTrainLongTransformerEncoderModel</span> with MELSPEC

    ![SpecificTrainLongTransformerEncoderModel](image/20210907_192510/SpecificTrainLongTransformerEncoderModel_Results_Accuracy_68.99_Percent.png)


* [20210908_121607](log/20210908_121607)  <span id="longgeneral12">SpecificTrainResNet18BackboneLongModel</span> with MELSPEC_VAD

    ![SpecificTrainResNet18BackboneLongModel](image/20210908_155528/SpecificTrainResNet18BackboneLongModel_Results_Accuracy_69.49_Percent.png)

* [20210907_230640](log/20210907_230640)  <span id="longgeneral13">MSMJointConcatFineTuneLongModel</span> with General

    ![MSMJointConcatFineTuneLongModel](image/20210908_235454/MSMJointConcatFineTuneLongModel_Results_Accuracy_73.17_Percent.png)

* [20210907_230640](log/20210907_230640)  <span id="longgeneral14">MSMJointConcatFineTuneLongModel</span> with Fine-tune

    ![MSMJointConcatFineTuneLongModel](image/20210909_103922/MSMJointConcatFineTuneLongModel_Results_Accuracy_72.81_Percent.png)

* [20210907_230704](log/20210907_230704)  <span id="longgeneral15">SpecificTrainLongModel</span> with MELSPEC_VAD

    ![SpecificTrainLongModel](image/20210909_121950/SpecificTrainLongModel_Results_Accuracy_67.67_Percent.png)

* [20210907_230704](log/20210907_230704)  <span id="longgeneral16">SpecificTrainLongModel</span> with SPECS_VAD

    ![SpecificTrainLongModel](image/20210909_124246/SpecificTrainLongModel_Results_Accuracy_68.30_Percent.png)

* [20210907_230704](log/20210907_230704)  <span id="longgeneral17">SpecificTrainLongModel</span> with MFCC_VAD

    ![SpecificTrainLongModel](image/20210909_130955/SpecificTrainLongModel_Results_Accuracy_65.84_Percent.png)

* [20210907_230704](log/20210907_230704)  <span id="longgeneral18">MSMJointConcatFineTuneLongModel</span> with Gereral VAD

    ![MSMJointConcatFineTuneLongModel](image/20210909_142350/MSMJointConcatFineTuneLongModel_Results_Accuracy_65.49_Percent.png)

* [20210907_230704](log/20210907_230704)  <span id="longgeneral19">MSMJointConcatFineTuneLongModel</span> with Fine tune VAD

    ![MSMJointConcatFineTuneLongModel](image/20210909_152527/MSMJointConcatFineTuneLongModel_Results_Accuracy_67.16_Percent.png)