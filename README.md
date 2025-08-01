# 需求：在现有Qwen3的基础上修复bug，现有bug是python和c++两端传参(input等）有问题，导致不管输入prompt是什么，最后转的张量都是0,导致胡言乱语复读。
## 现有参考：
### InfiniCore-infer框架内适配的jiuge模型
  路径位置：InfiniCore-Infer-main/src/models/jiuge
### InfiniCore所实现的框架
  路径位置：InfiniCore-main
### 需要适配的Qwen3模型实现代码（参考）
  路径位置：qwen3
