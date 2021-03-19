# Multi-Mode MAML and Optimization

## Acknowledgments
Our work is heavily based on the open source code provided by Shaohua Sun et al. in @GitHub https://github.com/shaohua0116/MMAML-Classification. Here we sincerely thanks to their wonderful job.

Based on the their framework, we use their dataset processing approaches and API designed in main(). We mainly implement our own tasknet and task_encoder which can be viewed in ./maml/model/tasknet.py and ./maml/model/embedding_modulation_net.py. 

Besides, thanks for Antreas Antoniou et al. 's work in @GitHub https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch for reference. Based on that, we rewrite the ./maml/metalearner.py and ./maml/trainer.py in the original repository https://github.com/shaohua0116/MMAML-Classification.

Finally, also thanks to the original MAML's codes provided by Finn et al. in @GitHub https://github.com/cbfinn/maml for reference, which helps us to make a deeper understand toward the insight of MAML.

## Datasets download
You can simply get the datasets cifar miniimagenet aircraft by running the command

```bash
python download.py --dataset aircraft cifar miniimagenet
```
The omniglot is provided in totch and can be automatically downloaded while using.

In theory, the bird dataset could also be downloaded via the command above, but it didn't work from our side(wget doesn't work). So you can firstly download it in http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz and then put it in the main folder and running the following command:
```bash
python get_dataset_script/proc_bird.py
rm -rf CUB_200_2011 CUB_200_2011.tgz
```

## Enviorment setting
Here we use anaconda to manage our environment. 
Firstly, you can create a new environment using command：
```bash
conda create --name mmaml_final python=3.6
```
Then we need to activate it:
```bash
conda activate mmaml_final
```
After that we will install several packages by typing:

```bash
conda install pytorch==1.2.0 torchvision==0.4.0

conda install ipython imageio scikit-image tqdm Pillow numpy scipy requests 

pip install tensorboardX
```

## Experiments 
We mainly did three experiments scenario: MAML in multi-task, MMAML in multi-task, MMAML+ in multi-task
You can test them by typing the command below:
### Training
#### MAML
```bash
python main.py --dataset multimodal_few_shot --multimodal_few_shot omniglot  miniimagenet  --maml-model True  --num-batches 150000 --output-folder maml_2modes_5w1s
python main.py --dataset multimodal_few_shot --multimodal_few_shot omniglot  miniimagenet cifar  --maml-model True  --num-batches 150000 --output-folder maml_3modes_5w1s
python main.py --dataset multimodal_few_shot --multimodal_few_shot omniglot  miniimagenet cifar bird --maml-model True  --num-batches 150000 --output-folder maml_4modes_5w1s
python main.py --dataset multimodal_few_shot --multimodal_few_shot omniglot  miniimagenet cifar bird aircraft --maml-model True  --num-batches 150000 --output-folder maml_5modes_5w1s
```
#### MMAML
```bash
python main.py --dataset multimodal_few_shot --multimodal_few_shot omniglot  miniimagenet  --mmaml-model True  --num-batches 150000 --output-folder mmaml_2modes_5w1s
python main.py --dataset multimodal_few_shot --multimodal_few_shot omniglot  miniimagenet cifar  --mmaml-model True  --num-batches 150000 --output-folder mmaml_3modes_5w1s
python main.py --dataset multimodal_few_shot --multimodal_few_shot omniglot  miniimagenet cifar bird --mmaml-model True  --num-batches 150000 --output-folder mmaml_4modes_5w1s
python main.py --dataset multimodal_few_shot --multimodal_few_shot omniglot  miniimagenet cifar bird aircraft --mmaml-model True  --num-batches 150000 --output-folder mmaml_5modes_5w1s
```
#### MMAML+(which might need choose a gpu with big memory, 11 G is enough in our experiments)
```bash
python main.py --dataset multimodal_few_shot --multimodal_few_shot omniglot  miniimagenet  --mmaml-model True --stabilize True --num-batches 150000 --output-folder mmaml+_2modes_5w1s
python main.py --dataset multimodal_few_shot --multimodal_few_shot omniglot  miniimagenet cifar  --mmaml-model True --stabilize True --num-batches 150000 --output-folder mmaml+_3modes_5w1s
python main.py --dataset multimodal_few_shot --multimodal_few_shot omniglot  miniimagenet cifar bird --mmaml-model True --stabilize True --num-batches 150000 --output-folder mmaml+_4modes_5w1s
python main.py --dataset multimodal_few_shot --multimodal_few_shot omniglot  miniimagenet cifar bird aircraft --mmaml-model True --stabilize True --num-batches 150000 --output-folder mmaml+_5modes_5w1s
```
### Evaluation
Here we take the second scenario as an example for evaluation：
```bash
python main.py --dataset multimodal_few_shot --multimodal_few_shot omniglot  miniimagenet  --mmaml-model True  --num-batches 150000 --output-folder mmaml_2modes_5w1s --checkpoint ./train_dir/mmaml_2modes_5w1s/maml_tasknet_150000.pt --eval True
```

- Arguments description 
    - --output-folder: a nickname for the training
    - --dataset: which dataset to use
    - --multimodal_few_shot: combination of different datasets
    - Checkpoints: specify the path to a pre-trained checkpoint
        - --checkpoint: load all the parameters (e.g. `./train_dir/mmaml_2modes_5w1s/maml_tasknet_150000.pt`).
    - Hyperparameters
        - --num-batches: total number of iteration
        - --meta-batch-size: number of tasks per batch
        - --slow-lr: learning rate for the global update of MAML
        - --fast-lr: learning rate for the adapted models
        - --num-updates: how many update steps in the inner loop
        - --num-classes-per-batch: how many classes per task (`N`-way)
        - --num-samples-per-class: how many samples per class for training (`K`-shot)
        - --num-val-samples: how many samples per class for validation
    - Logging
        - --log-interval: number of batches between tensorboard writes
        - --save-interval: number of batches between model saves
    - Model
        - maml-model: set to `True` to train a MAML model
        - mmaml-model: set to `True` to train a MMAML (our) model

## Reference 
Shaohua Sun et al. https://github.com/shaohua0116/MMAML-Classification.

Antreas Antoniou et al. https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch.

Finn et al. in GitHub https://github.com/cbfinn/maml.
