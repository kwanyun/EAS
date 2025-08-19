# StyleMM: Stylized 3D Morphable Face Model via Text-Driven Aligned Image Translation (PG 2025 Journal Track)

### [PG2025] Official repository of EAS of StyleMM [[Project Page](https://kwanyun.github.io/stylemm_page/)] 
<img width="1793" height="656" alt="i2i_comp" src="https://github.com/user-attachments/assets/9b018b75-babc-40d6-a180-7b28e4286729" />

#### This repository is code for EAS, not full Pipeline. Full pipeline will be released in september


### Getting Started
* install dependency
```bash
bash set.sh
```
* Put EAM weight and image folders to change the style
  
#### [EAM Weight and Realistic Facial Images](https://drive.google.com/file/d/1Y5vc1yGKbyiX4NblQzGwYSjsR18htBsR/view?usp=drive_link)


### How to inference Scripts
```bash
python inference.py --num_styles 3
```

#### To change style, check align_utils.py


<img width="1024" height="529" alt="EAS_rev" src="https://github.com/user-attachments/assets/ce99ef6f-de36-42e7-bb97-8c35a2e64b3b" />



### Acknowledgments
our codes were based on Diffusers library, ControlNet, and SDXL.

### If found EAS or StyleMM useful for your research, please cite our paper:
```bash
```


