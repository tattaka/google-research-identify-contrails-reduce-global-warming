# まとめ
* 2Dモデル最新版
  * exp044
* 2.5Dモデル最新版
  * exp043

# TODO
* エラー分析
* foldの切り方は後々考えたい
  * 画像サイズはたまに256じゃないやつもあるのでresize入れる
* binary classification modelを作る
  * 過去コンペ参考に
  * 分類パートでは分類ペット付きセマセグモデルと分類モデルどっちも試す
    * 前者ではセマセグヘッドはポジティブなやつでしか計算しない
  * pos/neg検出がどれだけ簡単か
  * pos/negで間違うことでどれだけ損失があるか
    * そこをリカバーするためにどうするべきか
* 各種SegModel option
  * deep supervision
  * attention
  * center
  * etc...

# DONE
* まずはtraining/validationで単純にやってみる
* ベースライン作成
  * UNet+BCE
* binary classification model
  * 過去コンペ参考に
  * 分類パートでは分類ペット付きセマセグモデルと分類モデルどっちも試す
    * 前者ではセマセグヘッドはポジティブなやつでしか計算しない
  * pos/neg検出がどれだけ簡単か
  * pos/negで間違うことでどれだけ損失があるか
    * そこをリカバーするためにどうするべきか
  * classification->0.8強、segmentation->0.64位
    * 悪くはないんだが......
* pipeline考える
  * stage1_clsとstage1_unetのtopkでフィルタリング
    * 雲コンペとか鉄コンペとか
  * stage1_unetとstage2_unetでアンサンブル
  * v2 pilelineで精度向上
    * stage1_clsいらないかも？
    * モデルとして異なる性質持ってる気がするので欲しいっちゃ欲しい？
* 週末のsubで確認すること
  * 現状とaugを強くしたやつを投げてみる
  * valに過学習してない？
  * 現状で高くなるならtestとvalがある程度分布が同じ
  * そうでないならaug強の方が強くなるはず

# IDEA
* hrnet(modified_stide)
* CSPDarkNet53
* swinv2_base_window12to24_192to384
* ashデータセットのclipをなくす？
* deep supervisionでmax pooling
* coat_lite_medium
* fpn
  * https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/fpn/decoder.py
* リークをどうにか
    * https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/419994
    * 時間みる
    * パッケージ使う
* pseudo label
  * trainにつける
  * 2.5Dモデルも同じdataset使えばいけそう
* 本命特徴と3D特徴のconcat
* 9ch + 2diff
* softmax cross entropy
* 3DCNN
* 10pixel以下削除
* freeze backbone, large size training
* indivisual label使う
* 全データ学習？
* フレーム間でpseudo label？
* Optical flowも使いたい？
  * optical flowはtwo streamで使う？
  * raftとか考えてみる
  * 可視化してみたけどあんまり......
* labelers modelsを作る
  * labelerの数が一緒じゃない->トレーニングセットでラベラーが共通してない？
* data loadingにmemmap使ってみる
  * https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/414549
* RGBスキームもスタッキングする
* 無と無で挟まれてるやつを後処理でどうにかするとか（いらないかも）
* mask_scoring_head
  * https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/discussion/107757
  * https://github.com/zjhuang22/maskscoring_rcnn
* bandってどういう意味？
  * https://ja.wikipedia.org/wiki/GOES
* 1024x1024入力の256出力モデル
  * deeplabv3+とか
  * FastFCNとか
* pointrend

# WORK
* epoch数を伸ばす
* unet内のnearestとbilinearを比較する
    * align cornerに関しても（nearestとbilinearのtrueを比較する）
    * nearestの方が微妙に良く高速だった
* model size up
  * 微量ではあるので要検討
* stage1において、classification > segmentation
* image size up
  * 上がるは上がるがそこまで
    * segで0.01, clsでほぼ変わらない  
* dice lossはあって良さそうだけどなくても最終的にはそこまで
  * 閾値が0.5付近になる
* FastFCNImprove
  * 悪くなさそう 
* swin transformer
  * clsでは悪くない
  * segも悪くない
  * 収束が早いのでもう少しハイパラ調整した方が良いかも？
* GroupNorm
  * 性能は悪くない
  * 大きいバッチで良くなる？
  * exp010
  * https://github.com/pudae/kaggle-understanding-clouds/blob/master/kvt/models/segmentations/unet.py
* train dir統計
  * 収束は早くなる
  * 精度向上はせず
* 3Dモデル検討
  * 火山コンペで使ったやつがいい感じ
  * それでも2DモデルでもっとCV上げて適用した方が良いかも？
  * num_3d_layerはclassificationは6, segは3で良さそう
    * exp16
* 使う枚数考える
  * 現在8枚
  * 3, 5, 7とかで精度見る
  * 7で良さそうな気配ある
* flipとrotationのpを調整する
  * stage1_clsはそのまま+cutout入れる
  * stage2_segは0.25まで下げる
  * stage1はflipとrotateを切る
  * 2.5Dモデルで適切なaug変わりそうでアレだけど精度は上がる（2Dモデルに比べたら改善は小さい）


# NOT WORK
* この辺は後でも試す
  * blurやnoise系統のaug
  * mixup
  * cutmix
* qfocalloss
* efficientnetv2_m
* fpa・CBAM
* label smoothing
  * 0.1
  * 収束は早くなる？
* deep supervision
* bceのpos_weight=5で良さげだけどaug変えたから説ある
  * distortion取ったからですね......
  * pos_weight
* segでのgrid distortion
* decoderのparam減らす+3d layerを6にする
* droppath変える
* 逆再生
  * 微妙？
* 論理和mixup
* tta
  * なぜ？
* label smoothing
* 画像wiseの正規化
* 5枚で最後の特徴だけ使う
* copy and paste
  * positive only model学習時にnegative modelにpositveなcopy貼り付ける
  * all image学習の時にsegmentation mask部分を平均値とかで埋める
* 後ろフレーム使わない方が良い？
  * 論文参照
  * これがうまくいけば推論も早くなる
  * そうなるとLSTMとかの方が良いかも？
* stage2でmaskが空にならないように厳密にみる
  * 多少の改善？
  * 誤差っぽい
* multi scale augmenation
* 2.5Dでflip周り修正したあと、逆再生も追加する？
  * 誤差
* edge effectの対処？
  * 大きめの画像で周りをrefrect paddingして切り取り
  * あんまり.....
* 地理情報でサンプリング？
  * trainとvalidの分布をみる
  * ~~それでも結構乖離ありそうなので、頻度の逆数で学習中にサンプリングするのは良いかも~~
    * ある程度相関してそうだけど、少数クラスな場所を学習するという意味では試してみても良いかも？
    * 精度だいぶ下がる......
      * valがある程度trainと地理的に一致しているのと、逆に多数クラスの学習が進んでない
      * valに過学習してない？その割にはtestでも相関取れてそうなのが謎
* 他フレームにpseudo label
  * 改善が微妙すぎる・計算時間少し増える
* mean teacher?
  * lossをmseとbceどちらも試す 
* 320sizeで学習して中心切り取る
  * exp043の320と比較する
  * 立ち上がりは良いけど最終的にはそこまで変わらなくなりそうだし、計算コスト的にボツ
* 最初のstride取るやつ
  * exp050のmodel name変える
* finetune
  * 2Dで学習して、encoderをfreezeして/しないで学習
    * しないで学習するのが良さげ
    * 多少CVは上がる？
  * 2.5Dで学習してencoderを解像度上げて学習
  * encoder/decoderのbatchnorm凍結する
  * 学習が不安定すぎる割にはゲイン少ない
* random brightness and contrast
* swinやconvnextの1st featureをupsampling+convしてUNetやFastFCNに食わせる
  * 結局convは使わずにupsamplingしてUNetに食わせる
    * 精度は若干落ちるが誤差の範囲
  * proc: featの数について考えなくても良くなる・FastFCNを使える
  * cons: 計算量増加・なんか精度低い(実装の都合上？)
  * 結局精度向上ならず・計算量も増えるのでボツ

# 状況整理

## データ
* ash形式に変換
* train/valで検証
* モデルに入力前にtrainデータセットの統計値で正規化
  * 後述の2.5Dモデルと前処理を共通化するために8フレーム使う
    * (train_data, 256, 256, 3, 8) -> (3)

## pipeline
1. binary classification model (all data)
2. segmentation model (all data)
3. segmentation model (positive only)
モデル1とモデル2のtopk logits(k=100)を用いて、飛行機雲が含まれないデータを除外(best dice=0.87)  
positiveと判定されたデータに対して、モデル2(alldata: dice=0.665)とモデル3でmaskを推定 (positiveのみ：dice=0.688, 最終: dice=0.67)    
こんな多段のpipelineなくていい気がする......

## モデル詳細

* 2Dモデル
  * CNN or Swin backbone + UNet (or FastFCN(+Upsample CNN))
* 2.5Dモデル
  * CNN or Swin backbone + 3D layers + UNet (or FastFCN(Upsample CNN))  

SwinやConvNeXtは収束が早いがResNet50RSより精度良くない  
実験では2.5Dモデル(7フレームのみ使用)が良いが学習に時間がかかる  

## その他
* augmentationを強くするとCV悪くなる  
  * 他正則化(mixup, label smoothing, cutmix)も同じ
  * 分類モデルではその傾向は多少マシに
  * flip・rotateさえも精度悪化の原因になる
    * 全く入れないとすぐ過学習するので少し入れているが、モデルが大きくなると過学習傾向
    * もっとパラメータ詰めると精度上がるかも？
  * 仮説：地理的な特徴を学習していて、回転やFlipに弱い/また時系列的にも近いものが各setにまたがっている
* EfficientNet学習できない（いつもの）
* classification head他Aux Headの追加
* all dataではBCE, positive onlyではBCE+dice

## TODO
* 別フレームのpseudo labelを作ってtrainに加える
  * 単純にデータ量が増える
  * valはtrainに近いデータ多いのでCVは上がりそう(testは謎)
* 256x256で学習後、encoderをfreezeして大きいサイズでfinetune
* 3DCNN
