# Neural Logic Inductive Learning
This is the implementation of the Neural Logic Inductive Learning model (NLIL) proposed in the ICLR 2020 paper:
[Learn to Explain Efficiently via Neural Logic Inductive Learning](https://openreview.net/forum?id=SJlh8CEYDB). The
Transformer implementation is based on this [repo](https://github.com/jadore801120/attention-is-all-you-need-pytorch).

### Requirements
- python 3.6+
- pytorch 1.1.0+
- numpy
- tqdm

### Knowledge completion on WN18 and FB15K

You can run knowledge completion task on WN18 and FB15K with provided scripts

    bash run_wn.sh
    bash run_fb.sh

### Object classification on Visual Genome

First, download the scene-graph dataset from the official site (click "Download Scene Graphs")

    https://cs.stanford.edu/people/dorarad/gqa/download.html

Extract the files, and run the following script to generate the dataset 

    bash preprocess.sh path/to/the/sgraph/folder
    
Now you can run object classification with

    bash run_gqa.sh

### Reference 

    @inproceedings{
        yang2020learn,
        title={Learn to Explain Efficiently via Neural Logic Inductive Learning},
        author={Yuan Yang and Le Song},
        booktitle={International Conference on Learning Representations},
        year={2020},
        url={https://openreview.net/forum?id=SJlh8CEYDB}
    }