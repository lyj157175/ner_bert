# NER-BERT

### **利用BERT模型进行中文命名实体识别任务**

使用数据集: MSRA dataset



## 用法

1. 将预训练好的bert模型放到bert-model文件夹下，包括`bert_config.josn`, `pytorch_model.bin`, `vovab_txt` 三个文件。我这里主要使用的是

https://github.com/ymcui/Chinese-BERT-wwm提供的预训练模型，可以根据需要下载。

2. 准备数据和标签

```shell
python build_msra_dataset_tags.py
```

3. 开始训练

```python
python train.py
```



## 结果

|  Dataset  | F1_score |
| :-------: | :------: |
| train set |  99.88   |
| valid set |  95.90   |
| test set  |  94.62   |

test set上的具体结果

| NE Types | Precison | Recall | F1_score |
| :------: | -------- | ------ | -------- |
|   PER    | 96.36    | 96.43  | 96.39    |
|   ORG    | 89.64    | 92.07  | 90.84    |
|   LOC    | 95.92    | 95.13  | 95.52    |



## 参考

- Devlin et al. BERT: Pre-training of Deep Bidirectional Trasnsformers for Language Understanding (2018) [[paper]](https://arxiv.org/pdf/1810.04805.pdf)
- huggingface/pytorch-pretrained-BERT [[github]](https://github.com/huggingface/pytorch-pretrained-BERT)
