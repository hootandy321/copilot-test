# 需求：在现有InfiniCore-infer框架内适配的jiuge模型的基础上对Qwen3-1.7B模型进行适配，要求使用InfiniCore框架的结构适配（可利用现有jiuge模型适配好的结构）
## 现有参考：
InfiniCore-infer框架内适配的jiuge模型
  路径位置：InfiniCore-Infer-main/src/models/jiuge
InfiniCore所实现的框架
  路径位置：InfiniCore-main
jiuge模型py实现代码
  路径位置：9g/modeling_fm9g.py.py
需要适配的Qwen3模型实现代码
  路径位置：qwen3
## 需要重点关注的：
  jiuge和qwen3的模型框架较为相似（主要不同在于多头注意力的实现部分和其他小细节），因此可以根据源代码对比模型，相同的部分李咏在现有jiuge适配好的内容，着重在于不同的实现
## 要求：
 能够在算力平台跑通运行，遇到的、解决的和未解决的问题都需要在readme中更新。需要实现推理服务功能，并适配OpenAI标准流式推理请求接口；修改在InfiniCore-infer（如果对InfiniCore有修改也需要标出）
