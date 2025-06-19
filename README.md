# LIBai
# 基于魔搭工具箱的 “李白” 数字分身模型开源说明

## 一、项目概述

本项目依托魔搭社区 ModelScope 平台，使用数据处理工具 Data-Juicer、模型训练工具 SWIFT，对国产开源大模型 [具体选用模型，如通义千问 Qwen 等] 进行微调训练，打造能拟合 “诗仙” 李白对话风格与知识领域的数字分身模型，可用于诗词文化传播、古代文学互动问答等场景，助力用户感受盛唐文化魅力与李白的诗词风采。

## 二、模型与工具

### （一）国产开源大模型

选用 [模型名称，如通义千问 Qwen - 7B]，模型在魔搭社区链接：[对应模型在魔搭社区的链接，需按实际填写] ，该模型具备良好的语言理解与生成基础，适配中文场景微调需求。

### （二）魔搭工具

1. **Data-Juicer**：用于对收集的与李白相关数据进行清洗、预处理，过滤低质量内容，规范数据格式，提升训练数据质量，工具链接：https://github.com/modelscope/data-juicer 。
2. **SWIFT**：基于高效微调技术（如 LoRA），对大模型进行针对性训练，降低训练成本，快速让模型拟合李白的语言风格，工具链接：https://github.com/modelscope/swift 。

## 三、数据相关

### （一）数据来源与生成流程

本项目数据为自建数据集，构建流程如下：

1. **数据采集**：从古诗文网、《全唐诗》等权威文献平台，爬取李白诗词原文、注释、赏析，以及历史文献中关于李白生平、轶事的文字记载；同时，整理文化类纪录片、学术论文里对李白语言风格、思想内涵分析的内容 。使用 Python 的`requests`库结合`BeautifulSoup`工具进行网络文本抓取，示例代码片段：

运行


```python
import requests
from bs4 import BeautifulSoup

url = "古诗文网李白相关页面URL"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
# 按需提取诗词、文本内容，如：
poetry_content = soup.find("div", class_="poetry-content").get_text()
```

1. **数据结构化处理**：将采集到的非结构化文本，按照`{"instruction": 问题/指令, "input": "", "output": 李白风格回答}`格式进行整理。例如，针对 “请以李白风格写一首咏月的诗” 指令，对应输出符合李白浪漫主义风格、运用夸张想象等手法的诗词文本 。借助 Python 脚本批量转换，核心代码逻辑：

```python
import json

raw_data = [...]  # 采集的原始文本数据列表
structured_data = []
for data in raw_data:
    instruction, output = process_data(data)  # 自定义函数处理得到指令和回答
    structured_data.append({"instruction": instruction, "input": "", "output": output})

with open("li_bai_dataset.json", "w", encoding="utf-8") as f:
    json.dump(structured_data, f, ensure_ascii=False, indent=4)
```

1. **数据清洗（Data-Juicer）**：配置 Data-Juicer 清洗规则，去除重复、低质量（如文本长度过短、含无关广告内容等）数据，保留优质训练样本。配置文件示例（`data_juicer_config.yaml`）：


```yaml
ops:
  - DocumentLengthFilter:
      min_len: 10
      max_len: 2000
  - DuplicatedExampleFilter:
  - TextQualityFilter:
      min_score: 0.8
```

运行清洗命令：


```bash
dj -c data_juicer_config.yaml -i input_data_dir -o output_data_dir
```

### （二）数据格式与规模

数据格式为 JSON，单条数据包含`instruction`（指令 / 问题）、`input`（选填，本项目多为空）、`output`（李白风格回答）字段。数据集规模为 [X] 条有效数据，涵盖诗词创作、生平问答、思想探讨等多种场景，总文本量约 [X] 字 。

## 四、训练过程

### （一）微调脚本（关键代码及注释）

以下为使用 SWIFT 进行模型微调的核心脚本（`finetune.py`）：

运行
```python
import os
from swift import Swift, LoRAConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型与数据路径配置
model_name_or_path = "[选用的大模型名称，如Qwen/Qwen-7B]"  # 魔搭社区对应模型路径
dataset_path = "output_data_dir"  # Data-Juicer清洗后的数据目录
output_dir = "./li_bai_model"  # 微调后模型保存路径

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, 
    device_map="auto", 
    trust_remote_code=True
)

# 配置LoRA微调参数，针对大模型高效微调
lora_config = LoRAConfig(
    r=8,  # LoRA秩，控制模型复杂度与训练成本
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 需根据选用模型架构调整，适配模型权重更新
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA技术到模型
model = Swift.prepare_model(model, lora_config)

# 训练参数设置
training_args = {
    "output_dir": output_dir,
    "per_device_train_batch_size": 4,  # 单设备训练批次大小
    "gradient_accumulation_steps": 4,  # 梯度累积步数，解决显存不足问题
    "learning_rate": 5e-5,  # 学习率，控制参数更新幅度
    "num_train_epochs": 3,  # 训练轮数
    "save_strategy": "epoch",  # 按轮次保存模型
    "logging_steps": 10,  # 日志记录间隔步数
    "fp16": True,  # 半精度训练加速
}

# 初始化训练器，开始微调
trainer = Swift.get_trainer(
    model=model,
    train_dataset=dataset_path,
    args=training_args,
    tokenizer=tokenizer
)
trainer.train()

# 保存微调后的模型
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

### （二）训练流程说明

1. **环境准备**：在魔搭社区 Notebook 中，选择支持的 GPU 环境（如 A10、A100），配置好 Python 环境（参考前文实验环境配置依赖），确保`modelscope`、`transformers`等库正确安装 。
2. **数据准备**：完成数据采集、结构化处理、清洗后，将数据集放置到指定目录，供训练脚本读取 。
3. **训练执行**：运行上述`finetune.py`脚本，训练过程中，可通过魔搭社区提供的 SwanLab 工具（若集成使用）跟踪训练指标（如损失值变化、训练速度等） 。训练时长约 [X] 小时（因模型规模、数据量、GPU 性能而异），迭代 [X] 次，最终模型在验证集上达到预期的角色拟合度 。

## 五、推理过程

### （一）推理脚本（`inference.py`及环境说明）

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载微调后的模型
model_path = "./li_bai_model"  # 微调后模型保存路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.float16,  # 与训练时精度匹配，加速推理
    trust_remote_code=True
)

# 推理函数定义
def generate_li_bai_response(prompt, max_length=512):
    """
    生成李白风格的回答
    :param prompt: 用户输入的指令/问题，如“请模仿李白写一首边塞诗”
    :param max_length: 生成回答的最大长度
    :return: 李白风格的文本回答
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,  # 控制生成文本的随机性，0.7为适中值
            top_p=0.9,  # 采样策略参数
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 示例调用
if __name__ == "__main__":
    user_prompt = "请以‘夜宿山寺’为意，续写一段李白风格的感悟"
    result = generate_li_bai_response(user_prompt)
    print(result)
```

**实验环境**：

- 操作系统：Linux（魔搭社区 Notebook 默认环境）
- Python 版本：3.9
- GPU：NVIDIA A10（推理时占用显存约 [X] GB，因模型和输入长度波动 ）

### （二）推理步骤说明

1. **模型加载**：确保微调后的模型文件已正确保存，脚本通过`AutoModelForCausalLM`和`AutoTokenizer`加载模型及对应的分词器 。
2. **输入处理**：用户输入问题 / 指令，经分词器转换为模型可识别的张量形式，并移动到 GPU 设备（若有） 。
3. **生成回答**：调用`model.generate`方法，设置生成参数（如`temperature`控制随机性、`max_length`控制文本长度），得到模型输出 。
4. **结果解码**：将模型输出的张量转换为可阅读的文本，去除特殊 token，返回给用户 。

## 六、推理效果

### （一）微调前后对比（文字示例）

| 场景 / 问题                | 微调前模型回答（大模型基础回答）                             | 微调后模型回答（李白数字分身）                               |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 写一首咏酒的诗             | 酒是一种常见饮品，适量饮用可愉悦心情，以下是一首关于酒的诗...（常规描述，缺乏风格） | 《醉饮人间》 金樽对月影凌乱，玉液倾杯意万千。 醉里不知天地老，狂歌笑问酒中仙。（模仿李白浪漫、豪放风格，运用夸张、想象手法 ） |
| 解释 “天生我材必有用” 含义 | 这句话表达了一种积极的自我认知，意思是上天赋予每个人才能必然有其用处...（常规解读） | “天生我材必有用”，此乃吾对自身之才的坚信！天地生吾，必赋吾纵横之才、不羁之性，纵遭坎坷，才亦有用武之地，如宝剑在鞘，终有出鞘饮血、光耀人间之时！（融入李白的自信、狂放语气，结合其生平心境解读 ） |

### （二）评估指标与方法

1. **角色拟合度**：邀请古代文学领域学者、李白诗词爱好者组成评估小组，对模型回答进行人工打分（1 - 5 分，5 分为最契合），随机抽取 [X] 条回答，平均得分达 [X] 分，表明模型能较好拟合李白风格 。
2. **文本质量**：使用 BLEU、ROUGE 等自然语言生成评估指标，对比微调后模型回答与真实李白诗词、言论在词汇、语义层面的相似度，BLEU 值达到 [X]，说明文本生成质量较高 。

## 七、模型部署（魔搭社区创空间）

### （一）部署流程

1. **模型开源后操作**：模型在魔搭社区审核通过开源后，进入模型详情页，确认模型支持 SwingDeploy 快速部署 。

2. 配置部署信息

   ：在模型库列表页面，过滤出支持 SwingDeploy 的模型，点击进入本模型详情页，点击右上角 “部署” 按钮 。在部署配置页面，设置：

   - 模型版本：选择微调后发布的模型版本
   - 部署地域：根据使用需求选择（如靠近目标用户区域）
   - 部署卡型：推荐与训练、推理环境适配的 GPU 卡型（如 A10 ）

3. **一键部署**：确认配置无误后，点击 “一键部署”，魔搭社区后台自动完成模型部署流程，部署完成后获取体验链接 。

### （二）体验说明

1. **操作指南**：用户通过体验链接进入创空间对话界面，在输入框输入问题（如 “请写一首送别友人的诗，仿李白风格” ），点击发送按钮，模型生成回答并展示 。

2. 应用场景示例

   ：

   - **文化传播**：输入 “讲解《将进酒》的创作背景和思想”，模型以李白的视角、语言风格解读，传递盛唐文化与诗歌内涵 。
   - **创意互动**：输入 “假设李白穿越到现代，写一首对城市夜景的感慨”，模型生成融合古今元素、保留李白风格的诗词文本，增强文化趣味性互动 。

## 八、模型属性设置

1. **开源许可**：选择 [具体开源协议，如 Apache License 2.0]，确保符合开源规范，允许其他开发者合理使用、修改模型 。
2. **适用场景**：标注为 “古代文学互动、诗词文化传播、历史人物模拟对话” 等，方便用户快速识别模型用途 。
3. **依赖软件包**：明确列出依赖的`modelscope`、`transformers`、`torch`等库的版本，如`modelscope==1.9.5`、`transformers==4.35.2`、`torch==2.0.1+cu118` ，便于其他开发者复现环境 。

## 九、提交审核与开源

1. **资料整理**：将训练好的模型文件、微调脚本、推理脚本、数据集（或数据生成流程说明）、README 文档等，按魔搭社区模型仓库要求整理齐全 。
2. **创建模型仓库**：登录魔搭社区账号，进入个人中心 - 模型管理，点击 “创建新模型仓库”，填写模型名称（如 “LiBai-DigitalAvatar” ）、简介（简述模型功能、特色 ）、所属领域（“自然语言处理 - 对话系统 - 历史人物模拟” ）等基本信息 。
3. **文件上传**：在模型仓库中，依次上传模型文件（含权重、配置等）、代码文件（`finetune.py`、`inference.py` ）、数据集（或数据生成流程文档 ）、README 文档，确保文件路径正确、可访问 。
4. **属性设置与提交审核**：在仓库设置中，配置模型属性（开源许可、适用场景、依赖等，见上文 “模型属性设置” ），确认无误后提交审核 。魔搭社区会对模型的合规性（知识产权、内容质量等）进行检查，一般 [X] 个工作日内完成审核（因模型复杂度可能调整 ）。
5. **审核通过与开源**：审核通过后，模型正式在魔搭社区开源，其他开发者可访问模型仓库，下载使用模型、查看代码和文档，参与模型优化与交流 。
