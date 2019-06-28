# DiversityMetrics
This  is the implementation of self-CIDEr and LSA-based diversity metrics (only for python 2.7). If you think this is helpful for your work, please cite the paper: [Qingzhong Wang and Antoni Chan. Describing like humans: on diversity in image captioning. CVPR, 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Describing_Like_Humans_On_Diversity_in_Image_Captioning_CVPR_2019_paper.html)

## Note ##
To compute the CIDEr score, TF-IDF file is required. In our paper, the TF-IDF is obtained from MSCOCO training dataset. And to compute the diversity, multiple captions for each image should be generated and the format must be the same as the file ./results/merge_results.json.

## Evaluation ##
1. Generating multiple captions for each image, for example 10 for each.
2. Put the json file in ./results and make sure that the format is the same as that of merge_results.json.
3. Download the TF-IDF file from [this link](https://drive.google.com/open?id=1jo2rdMZd9nGAz1CU-qk5ZG3W05CYmr4P) and put the file in ./data. Dowonload MSCOCO validation annotation file and put it in ./annotations.
4. Fill the information in the params.json.
5. Run accuracy_evalscript.py or diversity_evalscript.py to obtain the accuracy or diversity.

## References ##

- [Microsoft COCO Captions: Data Collection and Evaluation Server](http://arxiv.org/abs/1504.00325)
- PTBTokenizer: We use the [Stanford Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml) which is included in [Stanford CoreNLP 3.4.1](http://nlp.stanford.edu/software/corenlp.shtml).
- BLEU: [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf)
- Meteor: [Project page](http://www.cs.cmu.edu/~alavie/METEOR/) with related publications. We use the latest version (1.5) of the [Code](https://github.com/mjdenkowski/meteor). Changes have been made to the source code to properly aggreate the statistics for the entire corpus.
- Rouge-L: [ROUGE: A Package for Automatic Evaluation of Summaries](http://anthology.aclweb.org/W/W04/W04-1013.pdf)
- CIDEr: [CIDEr: Consensus-based Image Description Evaluation] (http://arxiv.org/pdf/1411.5726.pdf)

## Acknowledgement ##
- Ramakrishna Vedantam (Virgina Tech)
- MSCOCO Caption Evaluation Team (Xinlei Chen (CMU), Hao Fang (University of Washington), Tsung-Yi Lin (Cornell))

## Citation ##
If this is helpful for your work, please cite our paper as:

```
@InProceedings{Wang_2019_CVPR,
  author = {Wang, Qingzhong and Chan, Antoni B.},
  title = {Describing Like Humans: On Diversity in Image Captioning},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
  }
```
