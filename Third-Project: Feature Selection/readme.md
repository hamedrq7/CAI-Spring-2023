download the data from [https://drive.google.com/drive/folders/1TYnhCSMU1rQcAWqXYzsn-HBKYy-8IORY?usp=sharing](https://drive.google.com/drive/folders/1MmhfNndQr-1WVP93wVQSxKLeK-dACJCd?usp=sharing) and store it under the `./data/` directory


- to genereate a different dataset from the original Digit-Five dataset, download the Digit-Five dataset from [here](https://drive.google.com/open?id=1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm) and modify `store_digit_five()`
- effect of Triplet loss on digits is shown in the below figure:
<img src="https://github.com/hamedrq7/CAI-Spring-2023/blob/main/Final-Project%3A%20Triplet%20Loss%20and%20fusion/TripletLoss%20validation%20-%20Triplet%20applied%20on%20classes%20of%20mnist%20.jpg" width="500"/>

- results of different fusion models and different architectures are stored in `exp1.X...` folders, best results are in `fusion_mode_concat-beta_0.0-margin_0.001-bn_on_features_True-dropout_True-embed_space_256-model_name_fusion_model-lr_0.01`:


<img src="https://github.com/hamedrq7/CAI-Spring-2023/blob/main/Final-Project%3A%20Triplet%20Loss%20and%20fusion/exp1.4_%20effect%20of%20both%20dropout%20and%20BN%20on%20baseline%20models/imgs/fusion_mode_concat-beta_0.0-margin_0.001-bn_on_features_True-dropout_True-embed_space_256-model_name_fusion_model-lr_0.01/acc_loss.jpg" width="650"/>

|
<img src="https://github.com/hamedrq7/CAI-Spring-2023/blob/main/Final-Project%3A%20Triplet%20Loss%20and%20fusion/exp1.4_%20effect%20of%20both%20dropout%20and%20BN%20on%20baseline%20models/imgs/fusion_mode_concat-beta_0.0-margin_0.001-bn_on_features_True-dropout_True-embed_space_256-model_name_fusion_model-lr_0.01/loss_cross%20entropy%20loss.jpg" width="300"/>
|
<img src="https://github.com/hamedrq7/CAI-Spring-2023/blob/main/Final-Project%3A%20Triplet%20Loss%20and%20fusion/exp1.4_%20effect%20of%20both%20dropout%20and%20BN%20on%20baseline%20models/imgs/fusion_mode_concat-beta_0.0-margin_0.001-bn_on_features_True-dropout_True-embed_space_256-model_name_fusion_model-lr_0.01/loss_triplet%20loss.jpg" width="300"/>
|
 
