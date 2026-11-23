# Midterm 讲稿（14 页）

- **怎么用**：英文部分照着说。"意思"是中文对照，帮你理解自己在说什么，不用念。
- **时长**：完整说一遍约 7 分 40 秒（按每分钟 125 个词算）。去掉方括号 [ ] 里的句子约 7 分钟，12 分钟里还能留 4 分钟以上给问答。先计时完整练一遍：如果超过 8 分钟，正式讲时就固定去掉方括号里的句子。
- **计时**：标题后的时间是这一页大约开始的时刻。如果第 7 页晚于 3:15 才开始，或第 11 页晚于 5:50 才开始，就跳过后面所有方括号里的句子。
- **备注**：英文部分也写进了 PPTX 每页的备注，演讲者视图（Presenter View）里能看到。"If asked" 那一行只在被问到时用。
- **读法**：EEG 读 E-E-G；EEGNet 读 E-E-G net；INT8 读 int eight；FP32 读 F-P thirty-two；mRMR 逐个字母读；μ 读 mu（发音像 "mew"）；0.74 读 zero point seven four。
- **第 12 页**：只说"到今天全部完成"。如果被问到某一段具体是什么时候做的，照实回答。
- **维护**：这份讲稿是 PPTX 备注和 Word 版的来源；改了以后把两者都重新生成（命令见 README 第 11 节）。

## 1 · Cover（0:00）

> Hi everyone, I'm Peiye Li. My project is about brain-computer interfaces that read EEG. The question is how many electrodes we can remove, and how much we can shrink the model, while it still works on a different day.

意思：大家好，我是 Peiye Li。我的项目研究读取脑电（EEG）的脑机接口。问题是：能去掉多少个电极、能把模型缩小多少，同时换到另一天还能用？

## 2 · Why Fewer Electrodes?（0:20）

> Why fewer electrodes? Each electrode has two kinds of cost. One is hardware: a contact, an amplifier channel, and setup time. The other is computation: the decoder, which turns EEG into commands, processes every electrode separately in its first layer. So removing electrodes saves both, while pruning the network only saves computation. [That's why I start with electrodes.]

意思：为什么要少用电极？每个电极都有两种成本。一种是硬件：一个接触点、一个放大器通道，还有佩戴调试的时间。另一种是计算：解码器（把脑电翻译成指令的模型）的第一层要把每个电极分别处理一遍。所以减少电极两种成本都省，而剪枝只省计算。[所以我先从电极入手。]

If asked: Electrodes are the sensors on the scalp, while filters are learned features inside the network. Pruning removes filters, so it never reduces the number of electrodes.

（补充：电极是头皮上的传感器，滤波器是网络内部学到的特征。剪枝删的是滤波器，所以不会减少电极数量。）

## 3 · Related Work and Novelty（0:50）

> All of these pieces already exist. EEGNet is the decoder I use. Q-EEGNet already ran an 8-bit EEGNet on a microcontroller. MI-BMInet picked electrodes using network weights, then used 8 bits. My rule comes from mRMR: pick features that are relevant but not redundant. What's new is a fair comparison: I change one thing at a time, I kept the test day locked until the plan was final, and I explain why my idea didn't help.

意思：这些部分都已经有人做过。EEGNet 是我用的解码器。Q-EEGNet 已经把 8 位的 EEGNet 跑在单片机上。MI-BMInet 根据网络权重挑电极，然后改用 8 位。我的规则来自 mRMR：挑有用但不重复的特征。新的地方在于公平的比较：我一次只改一个因素；在计划定下来之前，测试那天的数据一直锁着；我还解释了自己的想法为什么没有起作用。

If asked: The weight-based rule trains EEGNet on all 22 electrodes, then ranks each electrode by the size of its learned spatial weights.

（补充：按权重挑选的规则先用全部 22 个电极训练 EEGNet，再按每个电极学到的空间权重大小排序。）

## 4 · Research Question（1:25）

> My question is this: with the same number of electrodes, does a rule that avoids redundancy beat two simpler rules on a new day? One simpler rule uses relevance alone, and the other uses network weights. And does the answer change with 8-bit numbers? I expected task-based choices to beat random ones, but similar electrodes might still be useful.

意思：我的问题是：电极数量相同时，一个避开重复的规则，能不能在新的一天赢过两种更简单的规则？一种只看有用程度，另一种看网络权重。换成 8 位数字以后，答案会不会变？我预期按任务挑选会比随机好，但相似的电极也许仍然有用。

## 5 · Dataset and Evaluation（1:55）

> I use a public dataset: nine people, each recorded on two days with 22 electrodes while imagining four movements. I made every decision using day 1 only. Then I froze the plan on November 3; until then, my code couldn't even open day 2. After that, I tested on day 2 once. [That's how a real system works: you calibrate it one day and use it the next.]

意思：我用的是公开数据集：9 个人，每人在两天各录一次，用 22 个电极，想象四种动作。所有决定都只用第一天的数据做。然后我在 11 月 3 日冻结了计划；在那之前，我的代码连第二天的数据都打不开。之后我只在第二天上测试了一次。[实际系统就是这样用的：一天校准，第二天使用。]

If asked: Each person has 288 trials per day, and I use the 2 seconds after each cue. Trials marked as artifacts are left out of the main analysis.

（补充：每人每天 288 次试次，取每次提示后的 2 秒。被标记为伪迹的试次不进主分析。）

## 6 · Systems View: Model Size and Compute（2:30）

> Now for the systems view. The model is tiny: 2,420 parameters, about 9 kilobytes. But each prediction takes 5.9 million multiply-add operations, and 96 percent of them are in the first layer, which runs once per electrode. So computation grows with the number of electrodes: going from 22 to 8 cuts it by almost two-thirds. [Shrinking the weights alone doesn't save much.]

意思：现在从系统角度看。模型很小：2,420 个参数，大约 9 KB。但每做一次预测要 590 万次乘加运算，其中 96% 在第一层，而第一层对每个电极各运行一次。所以计算量随电极数量增长：从 22 个减到 8 个，计算少了将近三分之二。[只压缩权重省不了多少。]

If asked: Going from 22 to 16 electrodes cuts computation by 27 percent, and going to 8 cuts it by 63 percent. The time per prediction drops by almost as much.

（补充：从 22 个减到 16 个电极，计算少 27%；减到 8 个，少 63%。每次预测的时间也下降了差不多的比例。）

## 7 · Method: Electrode Selection（3:00）

> To pick electrodes, my rule uses two scores. Relevance is how well an electrode's mu and beta power distinguish the four movements. [These rhythms drop when you imagine moving.] Redundancy is how much it overlaps with the electrodes already chosen. The rule adds one electrode at a time, always choosing the one with the best relevance minus half its redundancy. I compare it with relevance alone, which is the same rule without the penalty, a weight-based rule like MI-BMInet, three random sets, and all 22. [Every rule favored the electrodes over the hand areas.]

意思：挑电极时，我的规则用两个分数。有用程度（relevance）：一个电极的 μ 波和 β 波能量能多好地区分四种动作。[想象动作时，这两种节律会减弱。] 重复程度（redundancy）：它和已经选中的电极有多少重叠。规则一次加一个电极，每次选"有用程度减去一半重复程度"最高的那个。我把它和这些对照比较：只看有用程度（也就是去掉惩罚项的同一规则）、类似 MI-BMInet 的按权重挑选、三组随机，以及 22 个全用。[所有规则都偏爱控制手的脑区上方的电极。]

If asked: Relevance is a Fisher score on log power in the mu band, 8 to 13 hertz, and the beta band, 13 to 30 hertz, scaled from 0 to 1. Redundancy is the average absolute Spearman correlation with the electrodes already chosen.

（补充：有用程度是 μ 频段（8–13 Hz）和 β 频段（13–30 Hz）对数能量的 Fisher 分数，缩放到 0 到 1；重复程度是和已选电极的 Spearman 相关系数取绝对值后的平均。）

## 8 · Result 1: Accuracy vs. Electrode Count（3:45）

> Here are the day-2 results. All 22 electrodes give 65 percent, where chance is 25. With 16 electrodes chosen by any task-based rule, accuracy is 66 to 67 percent, so basically the same. With only 8, those rules lose 3 to 5 points, and random sets lose even more. INT8 changed these numbers by half a point or less.

意思：这是第二天的结果。22 个电极全用是 65%，瞎猜是 25%。用任何一种按任务挑选的规则选 16 个电极，准确率是 66% 到 67%，基本一样。只剩 8 个时，这些规则掉 3 到 5 个点，随机挑的掉得更多。换成 INT8 后，这些数字的变化都不超过半个点。

If asked: Compared with all 22 electrodes, at 8 electrodes my rule loses 4.4 points, relevance alone 5.1, and the weight-based rule 2.9. At 16 electrodes, every confidence interval includes zero.

（补充：和 22 个全用相比，8 个电极时我的规则掉 4.4 个点，只看有用程度掉 5.1，按权重掉 2.9；16 个电极时，所有置信区间都包含 0。）

## 9 · Result 2: Comparing Selection Rules（4:15）

> This answers my question. Each bar is my rule minus one control. It beats random sets by 2 to 3 points, but it gains less than one point over relevance alone, which is within noise, and it doesn't beat the weight-based rule. Why so little? These rhythms rise and fall across the whole scalp together, with an average correlation of 0.74. So every electrode looks equally redundant, and for 6 or 7 of the 9 people, both rules picked the same electrodes.

意思：这页回答了我的问题。每根柱子是"我的规则减去一个对照"。它比随机组合高 2 到 3 个点，但比只看有用程度只高不到 1 个点，在误差范围内；也没有赢过按权重挑选的规则。为什么差别这么小？这些节律在整个头皮上一起增强或减弱，电极之间平均相关系数是 0.74。所以每个电极看起来都一样重复；9 个人里有 6 到 7 个，两种规则选出的电极完全相同。

If asked: My rule is 0.6 points above relevance alone at 16 electrodes and 0.7 at 8. Against the weight-based rule, it ties at 16 and is 1.5 points lower at 8. All four confidence intervals include zero, and INT8 doesn't change this. Against random sets, my rule was better for 8 of the 9 people.

（补充：我的规则比只看有用程度，16 个电极时高 0.6、8 个时高 0.7；和按权重比，16 个时打平、8 个时低 1.5。这四个置信区间都包含 0，换成 INT8 也一样。和随机比，9 个人里有 8 个是我的规则更好。）

## 10 · Result 3: INT8 Quantization（4:55）

> Next, compression. INT8 stores each number in 8 bits instead of 32. Accuracy changed by half a point or less, and the weights became almost three times smaller. But on my Mac's CPU, each prediction took 28 percent longer. The chart shows why: the first layer got faster, but the 8-bit grouped spatial convolution was 14 times slower. So smaller doesn't automatically mean faster. [You have to measure on the hardware you'll actually use.]

意思：接下来是压缩。INT8 就是每个数字用 8 位存，而不是 32 位。准确率变化不超过半个点，权重小了将近三倍。但在我 Mac 的 CPU 上，每次预测反而多花 28% 的时间。图上能看到原因：第一层变快了，但 8 位的分组空间卷积慢了 14 倍。所以模型变小不等于变快。[必须在实际要用的硬件上测。]

If asked: The first layer got 38 percent faster. A microcontroller with fast integer kernels could behave differently; I only measured on a desktop CPU. My proposal listed slow quantized kernels as a risk.

（补充：第一层快了 38%。有快速整数运算核的单片机结果可能不同，我只在电脑 CPU 上测过。proposal 里已经把"量化算子慢"列为风险。）

## 11 · Result 4: Pruning（5:30）

> For pruning, I removed half the filters in the last convolution layer, then trained 100 more epochs. My frozen final choice, 16 electrodes with pruning and INT8, scored 65 percent, the same as all 22. But only the pruned models got that extra training. After seeing the test results, I reran the comparison on a second Mac with equal training. This time, pruning cost 1.56 points, and it saved only 0.4 percent of the computation.

意思：剪枝方面，我删掉了最后一个卷积层一半的滤波器，然后多训练了 100 轮。冻结的最终方案（16 个电极、剪枝加 INT8）得了 65%，和 22 个全用一样。但只有剪过的模型得到了这些额外训练。看到测试结果之后，我在另一台 Mac 上用相同的训练量重新做了比较。这一次，剪枝损失了 1.56 个点，而只省了 0.4% 的计算。

If asked: The 95 percent confidence interval is −2.86 to −0.42 points, entirely below zero. With the frozen choice, 4 of the 9 people lost more than 2 points. This extra comparison doesn't change the frozen choice.

（补充：95% 置信区间是 −2.86 到 −0.42 个点，整个在 0 以下。冻结方案下，9 个人里有 4 个掉了 2 个点以上。这个额外比较不改变冻结的方案。）

## 12 · Progress vs. Schedule（6:10）

> Compared with my proposal's schedule, all the experiment phases are done, and I froze the plan on November 3, before looking at day 2. What's left is the report and the final presentation, which are on schedule. [I also put the code and all the models on GitHub.]

意思：对照 proposal 里的时间表，所有实验阶段都完成了；我在 11 月 3 日冻结了计划，那时还没看第二天的数据。剩下的是报告和期末展示，按计划进行。[我还把代码和所有模型放上了 GitHub。]

If asked: My proposal's backup plan was quantization-aware training, but I didn't need it, because INT8 lost at most half a point.

（补充：proposal 里的备用方案是量化感知训练，但没用上，因为 INT8 最多只掉半个点。）

## 13 · Next Steps（6:35）

> For next steps, I'll first finish the report and slides using today's feedback. [The frozen results will stay my main claims.] The first follow-up asks why INT8 was slower. I'll keep only the slow layer in 32-bit and time all three versions on the same Mac, with no retraining. The second, if time allows, is a better redundancy measure that first removes the activity all electrodes share. [Ideally, I'd confirm it on a second two-day dataset.]

意思：下一步，我会先根据今天的反馈完成报告和幻灯片。[主要结论仍以冻结计划的结果为准。] 第一个后续问题是 INT8 为什么更慢：我会只把那个慢的层留成 32 位，在同一台 Mac 上测三个版本的时间，不用重新训练。第二个，如果时间允许，是更好的重复程度度量：先去掉所有电极共有的那部分活动。[最好再在另一个也录了两天的数据集上确认。]

If asked: Both follow-ups come after the frozen plan, so I'll report their results as exploratory.

（补充：两个后续都在冻结计划之后，所以我会把结果当作探索性的来报告。）

## 14 · Takeaways（7:10）

> To sum up: first, how many electrodes you use, and which ones, matters more than how the weights are stored. Second, 16 well-chosen electrodes work as well as all 22, but avoiding redundancy didn't beat relevance alone. Third, INT8 saves memory but not time on my Mac, and with equal training, pruning costs accuracy. Thank you, I'm happy to take questions.

意思：总结一下：第一，用多少个电极、用哪几个，比权重怎么存更重要。第二，挑选过的 16 个电极和 22 个一样好，但避开重复没有赢过只看有用程度。第三，INT8 省了内存，但在我的 Mac 上没省时间；训练量相同时，剪枝会损失准确率。谢谢，欢迎提问。
