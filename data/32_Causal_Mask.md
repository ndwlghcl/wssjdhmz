# 🎭 Causal Mask

> 路径：AI / 机器学习学习进度 / 📚 知识点复习索引
>
> https://app.notion.com/p/3bf776baec49814e9240f99d4d3404ef?pvs=204

## 重点是什么
- 自回归语言模型生成当前位置 token 时，不能偷看未来 token。
- Causal Mask 通过遮挡未来位置，保证预测符合因果方向。
## 当前理解
🟡 **接触过，尚未正式验收**
## 还缺什么
- mask 在 attention score 上具体如何实现。
- 训练时为什么可以一次并行处理整段序列，同时仍保持因果约束。