model处理相关：
/src/main.py
增加load weight相关处理部分：prepare_checkpoint_path，处理pretrain-model的选择

数据处理相关：
step 1: 在dataset和experiment下新建yaml。
step 2: src/dataset下新建数据读取py
step 3: src/dataset/__init__.py下补充DATASETS和DatasetCfgWrapper字段，src/dataset/data_module.py下补充多数据训练时选择数据集概率。

src/dataset/view_sampler/view_sampler_all.py 返回的target views的个数，可选修改。
config/dataset/view_sampler/all.yaml中需要补充几个参数，因为在src/dataset/data_sampler.py中MixedBatchSampler初始化会实例化DynamicBatchSampler，需要对应几个参数。

src/dataset/data_sampler.py 中，处理图像的高度相关random_ps_h；以及重写MixedBatchSampler，目的是处理中validation时，val数据的数量限制
src/dataset/data_module.py中，val_dataloader实例化MixedBatchSampler，需要显示指定val的数据数量。

训练过程相关：
src/model/model_wrapper.py 中，图像颜色输出；宽视野图像render；视频合成。
src/model/model/anysplat.py 中修改内参，得到宽视野图像。
src/model/model_wrapper.py configure_optimizers函数中，所有参数都作为pretrained params

用伪DepthMap和伪pose，生成GT pose下的伪DepthMap，作为和GT pose配套的DepthMap监督信号。