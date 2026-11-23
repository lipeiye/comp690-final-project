# Final 问答预案

- **怎么用**：每个问题先给出照着说的两三句英文（引号块），再是中文意思、用到的数字和它们出自哪个文件。回答问题占汇报成绩的 20%，被问到时先用第一句直接回答，再补数字。
- **数字**：这里的数字是手写的，出处写在每题最后；结果更新后要和出处核对（和讲稿一样）。11-22 21:00 起所有实验都已完成，每题的数字都已按当时的结果写好；以后结果文件再变，按每题最后的出处重新核对。
- **读法**同讲稿：EEG 读 E-E-G；INT8 读 int eight；VNNI 逐个字母读；Riemannian 读 ree-MAH-nee-un。

## 两分钟说清这个项目（中文白话，自己先熟悉，也可以在被问“你到底做了什么”时用英文版）

脑机接口要在很小的芯片上读脑电，每个电极、每一位数字、每一次乘法都要花电。我用一个很小的神经网络（EEGNet），在 9 个人、两天的标准数据集上问了三件事：去掉一些电极会不会掉准确率，8 位整数够不够用、会不会更快，这些结论换一批人还成不成立。所有选择只用第一天的数据，第二天只打一次分。

结果：保留最相关的 16 个电极（共 22 个）不掉准确率；在另一个 54 人的数据集上，我在打分之前写下预测，再检验，结论成立。原因是网络每多一个输入都要付出代价：代价来自这个输入里其他电极预测不了的那部分，网络会拿这部分去记训练试验。被去掉的电极大部分能被留下的电极预测；在第二个数据集上直接量，预测不了的那部分代价约 4 个点，带来的信息也约 4 个点，正好抵消。只有当能用很多别人的数据预训练时（第二个数据集上 45 个人），全部电极才比 16 个好。一起考虑电极组合的黎曼距离选法比一个一个打分好，也在 54 人上事先登记后证实。8 位整数几乎不掉准确率，但导出的 INT8 反而比浮点慢，原因是运行库缺了一个分组卷积的内核；我把两个缺的内核补上，又调换了前两层，最后比原来的浮点快约 5.7 倍。同一个 INT8 文件在没有 VNNI 的 x86 CPU 上会掉约 4 个百分点，用 7 位权重可以解决。在模拟的单片机上，我又补了一个时间卷积的内核，指令少 1.5–2.8 倍，输出逐位不变。

> In one sentence: I made a small EEG decoder cheaper in electrodes, bits and time, checked every saving on the hardware, and tested the main findings again on 54 new people with the predictions written down before scoring.

## Q1 · Why can 16 electrodes do as well as 22?（期中问题 1）

> They don't do better, they do about as well: retrained on the same hardware with ten seeds, 16 electrodes are 0.3 points above 22, and the interval includes zero. The dropped electrodes are mostly predictable from the kept ones. I measured both sides on 54 new people: their unpredictable part adds about 4 points, and as an input it costs about 4, mostly because the network uses it to memorize its training trials. The two cancel, as registered before scoring.

意思：不是更好，是差不多：同一硬件重训 10 个种子，16 个比 22 个高 0.3 个百分点，区间包含 0。被去掉的电极大部分能被留下的电极预测。我在 54 个新的人身上把两边都直接量了：它们预测不了的那部分带来约 4 个点的信息；作为输入，这部分也要付出约 4 个点的代价，主要是因为网络拿它去记训练试验。两者抵消，和打分前登记的预测一致。

数字：+0.32 [−0.42, +1.04]（`results/supplementary/overfit_study/contrasts.csv`）；被去掉的电极只有 3.7%（2a）和 29%（Lee 2019）的方差不能被留下的 16 个线性预测（`input_kind_study/analysis.json` 的 predictability）。WP1b `residual_surrogate_study/summary.md`：把每次试验预测不了的那部分换成别的试验的，代价 −4.12 [−5.31, −3.02]（S2 成立），真实的那部分多带来 +4.30 [+3.09, +5.50]（S1 成立），两者之和 +0.18；只给能预测的部分 +0.27（S3 成立）。2a（探索性）：代价 −1.50 [−2.29, −0.70]，信息第二天 +0.34 [−0.53, +1.18]、当天留出组 +2.20。有预测不了的部分时训练试验 99%，没有时 96%（Lee 2019）。被追问“那每个人都这样吗”：不是。S4 没有成立：54 人之间，一个人被去掉的电极有多难预测，和这个人的代价不相关（rho −0.08 [−0.34, +0.20]）；每个人的代价本身在两半种子之间可重复（rho +0.49），所以不是量不出来。“记训练试验”的证据来自 WP3b（Q15）。被问到“原来不是说信息 6 个点吗”：那是拿同位置的噪声比，把“比噪声容易预测、所以代价小”也算成了信息，高估了；线性插值的粗估是代价 1.7 / 4.5，和实测的 1.5 / 4.1 接近。Lee 2019 H1 +0.29 [−0.83, +1.46]、H1b +0.18 [−1.41, +1.80]（`replication_study/hypotheses.csv`）。

## Q2 · How exactly did you choose the electrodes?（期中问题 2）

> On the training day only, I scored each electrode by how well its log power in the mu and beta bands separates the classes, a Fisher score, and kept the top 16 for each person. The redundancy-aware version subtracts half of the electrode's average rank correlation with those already chosen, but that changed almost nothing. The best method we found later looks at electrodes together, using the Riemannian distance between class covariance matrices.

意思：只用训练日：每个电极算它在 mu（8–13 Hz）和 beta（13–30 Hz）频带的对数功率能多好地区分各类（Fisher 分数），每人保留最高的 16 个。考虑冗余的版本会减去它和已选电极平均等级相关的一半，但几乎没有改变结果。后来找到的最好方法是把电极放在一起看：用各类协方差矩阵之间的黎曼距离。

数字：冗余对只看相关性 +0.28 [−0.42, +1.06]（10 个种子，`electrode_count/analysis.json`）；黎曼 8 个电极 64.3% 对 59.9%（`selection_study/accuracy.csv`）。代码：`src/mibci/selection.py`。

## Q3 · Why was INT8 slower, and is INT16 better?（期中问题 3）

> INT8 was slower because ONNX Runtime has no fast INT8 path for EEGNet's spatial layer, eight groups of two filters across every electrode, so it falls to a general path that is about 15 times slower per multiply. The chip is fine: the temporal layer ran faster in INT8. INT16 bought nothing in accuracy and was slower than float on every CPU. With the two missing kernels written and two layers swapped, INT8 takes 47 microseconds instead of 273 for float.

意思：INT8 慢是因为 ONNX Runtime 对 EEGNet 的空间层（8 组、每组 2 个滤波器、覆盖全部电极）没有快速的 INT8 路径，只能走通用路径，每次乘加慢约 15 倍。芯片没问题：时间卷积在 INT8 下更快。INT16 没有提高准确率，而且在每个 CPU 上都比浮点慢。补上两个缺的内核、调换前两层后，INT8 只要 47 微秒，原来的浮点要 273 微秒。

数字：`ort_kernel_study/latency.csv`（22 个电极：FP32 273、INT8 357、加内核 180、调换后加内核 47 微秒）；INT16 −0.02、INT8 −0.10 pp（`quant_study/ptq_grid.csv`）；合成模型的快慢地图（`int8_paths_study/summary.md`：M4 上 EEGNet 空间层形状 INT8 是 FP32 的 15.6 倍；四种 x86 CPU 上 38 种卷积形状里 37–38 种 INT8 都比 FP32 慢，所以“调换后 INT8 比浮点快”只在 M4 上成立，x86 上调换后的 FP32 最快）。

## Q4 · Isn't this just overfitting?

> Partly, and we measured it: the networks fit 93 to 98 percent of their training trials but only 51 to 62 percent of held-out runs from the same day, so most of the drop to the next day happens within the same day. Useless inputs cost as much on held-out runs of the same day as on the next day. That is why the cost is about generalizing to new trials, not about the day-to-day shift.

意思：部分是，而且量过：网络在训练试验上 93%–98%，但同一天留出的组只有 51%–62%，所以到第二天的大部分落差在同一天之内就发生了。没用的输入在同一天留出组上的代价和第二天一样大。所以代价是“推广到新试验”的问题，不是跨天变化。

数字：`overfit_study/levels.csv`；`input_cost_study/costs.csv`（2a 4 组训练、加 16 个：当天 −14.6、第二天 −13.4）。被追问“噪声输入的代价是不是就是过拟合”：是，而且直接量过（WP3b）：录下来的噪声每轮都一样，模型配原来的噪声训练试验 100% 答对，换一份没见过的噪声只剩 61.5%；噪声改成每轮重抽，代价从 −13.3 降到 −1.1（`fresh_noise_study/summary.md`）。

## Q5 · Why not just train longer?

> I tried it: four times longer, 1,900 epochs, did not shrink the cost of noise inputs or let all 62 electrodes beat 16 on the second dataset, as registered. The models with 46 noise inputs fitted their training trials fastest but stayed near chance on held-out runs of the same day the whole time, so it is a generalization problem, not too little training. On the first dataset, longer training raised every condition by about 2 points without changing the comparisons.

意思：试过了：训练 4 倍长（1,900 轮），在第二个数据集上既没有让噪声输入的代价变小，也没有让全部 62 个电极胜过 16 个，和事先登记的预测相反。加 46 个噪声输入的模型在训练试验上拟合得最快，但在同一天留出的组上从头到尾都在随机水平附近，所以这是推广能力的问题，不是训练不够。在第一个数据集上，训练更久让每种条件都多了约 2 个点，但比较的结果不变。

数字：`long_training_study/summary.md`（L1 +0.52 [−0.40, +1.49]，L2 +0.68 [−0.61, +1.94]，L3 +0.19 [−0.48, +0.85]，都不成立）；`long_training_study/curves.csv`（Lee 2019 加 46 个噪声：留出组第 100、475、1,900 轮 50.3%、51.6%、51.7%，第 475 轮训练损失 0.03，不加噪声 0.14）；2a 1,900 对 475 轮：relevance-16 +1.68、加 6 个噪声 +2.48、全部 22 个 +2.04。被追问“为什么加噪声的模型拟合得最快”：它们靠每轮都一样的噪声记住训练试验（WP3b：换一份没见过的噪声，训练试验从 100% 掉到 61.5%）。

## Q6 · Why not pretrain on other people?

> I did, on the second dataset: pretraining on 45 other people and fine-tuning on the person's first day beat training from scratch by 7.7 points with all 62 electrodes. It also cut the cost of 46 noise inputs from 12 points to 2, and then all 62 electrodes beat a shared set of 16 by 4.5 points; all three were registered before scoring. On the first dataset, eight other people were too few: pretraining lowered accuracy by 9 points.

意思：做了，在第二个数据集上：用另外 45 个人的数据预训练，再用本人第一天微调，全部 62 个电极时比从头训练高 7.7 个点；46 个噪声输入的代价从 12 个点降到 2 个点；这时全部 62 个电极比共用的 16 个高 4.5 个点。三条都是打分前登记的预测。在第一个数据集上只有 8 个别人，太少了，预训练让准确率低了 9 个点。

数字：`pretrain_study/summary.md`（Lee 2019：T1 +7.65 [+5.39, +9.94]，74.55% 对 66.90%；T2 +9.91，噪声代价 −12.08 → −2.17；T3 +4.53 [+2.84, +6.44]，从头训练时 +1.19 不显著；2a 全部 22 个微调减从头训练 −9.15）。被追问“那为什么还推荐 16 个”：只有本人一天的数据、从头训练时，16 个就够；能用很多别人的数据预训练时，更多电极才划算。共用的 16 个是所有人同一套，不是每人自己选的。

## Q7 · Why didn't you test on a real microcontroller?

> I ran the real Arm library, CMSIS-NN, on emulated Cortex-M0, M4 and M55 boards, and its output matched my integer reference bit for bit in all builds, so the arithmetic is what a chip would compute. What emulation cannot give is time: it counts instructions, not cycles or memory waits. Timing on a physical board is the next step, and the instruction counts tell us where to look.

意思：我在模拟的 Cortex-M0、M4、M55 上运行了真正的 Arm 库 CMSIS-NN，所有构建的输出都和我的整数参考逐位一致，所以算出来的就是芯片会算出来的。模拟给不了时间：它数的是指令，不是时钟周期和内存等待。在真实的板子上计时是下一步，指令条数告诉我们该看哪里。

数字：`mcu_study/analysis.json`（136/136 个 INT8 构建逐位一致）；`mcu_kernel_study/summary.md`（补上时间卷积内核，136/136 逐位一致，指令少 1.5–2.8 倍）。

## Q8 · Are 9 people enough? Are 54?

> Nine are not enough to generalize, which is why I added a second dataset of 54 people and wrote the predictions down before scoring it. With 54 people the intervals are around plus or minus 1 to 2 points, and the main results held. I still report means with intervals and how many people lost more than 2 points, because averages hide individual losses.

意思：9 个人不够推广，所以我加了第二个 54 人的数据集，并在打分前写下预测。54 个人时区间大约正负 1–2 个百分点，主要结论都成立。我仍然报告均值、区间和“掉 2 个百分点以上的人数”，因为平均数会掩盖个体损失。

数字：`replication_study/hypotheses.csv`（H1b：24/54 人掉 2 pp 以上）；冻结方案 4/9 人掉 2 pp 以上（`results/analysis/conditions_primary.csv`）。

## Q9 · Why EEGNet and not a newer model?

> EEGNet is the standard compact baseline: about 2,400 parameters, used by the embedded work I compare with, like Q-EEGNet and MI-BMInet. My questions are about electrodes, bits and kernels, so I kept one well-known network and changed one thing at a time. Larger models would make the hardware questions harder, not different.

意思：EEGNet 是公认的小型基线：约 2,400 个参数，我对比的嵌入式工作（Q-EEGNet、MI-BMInet）都用它。我的问题是电极、位数和内核，所以只用一个大家熟悉的网络，每次只改一样东西。更大的模型会让硬件问题更难，而不是不同。

数字：参数 2,420、乘加 5.9 百万（`results/analysis/efficiency.csv`）。

## Q10 · What happens in online use?

> My demo replays recorded EEG, so it is not closed-loop control. The decoder needs well under a millisecond per decision, so the 2-second window, not the computation, sets the delay. Online use adds feedback and the user adapting, which a replay cannot show; half of every Lee 2019 test day comes from its online run, so those trials are inside every Lee 2019 score here.

意思：演示是回放录好的脑电，不是闭环控制。每次判断的计算远小于 1 毫秒，所以延迟由 2 秒的观察窗口决定，不是计算。在线使用还会有反馈和用户自己的适应，回放显示不出来；Lee 2019 每个测试日有一半试验来自在线组，所以这里每个 Lee 2019 的分数里都包含它们。

数字：冻结方案调换层顺序加内核 44 微秒（`ort_kernel_study/latency.csv`）；Lee 2019 每天 100 次离线、100 次在线试验（`src/mibci/lee2019.py`）。

## Q11 · Why is INT8 less accurate on some CPUs?

> On x86 CPUs without the VNNI instructions, ONNX Runtime's 8-bit kernel adds two 8-bit products in a 16-bit register that can saturate, while Arm and VNNI add straight into 32 bits. The same INT8 file changed predictions for 457 of 459 models there and lost about 4 points. I checked it layer by layer: the first difference is in the temporal convolution, and every output that differs is one the 16-bit pairing predicts to overflow. Using 7-bit weights in every layer, which ONNX Runtime documents, gives identical results on every CPU I tested.

意思：在没有 VNNI 指令的 x86 CPU 上，ONNX Runtime 的 8 位内核先把两个 8 位乘积加到一个 16 位寄存器里，可能饱和；Arm 和 VNNI 直接加到 32 位。同一个 INT8 文件在那里 459 个模型里有 457 个预测变了，掉约 4 个百分点。我逐层核对过：第一个不同的是时间卷积，每个不同的输出都是按 16 位配对会溢出的那种。每一层都用 7 位权重（ONNX Runtime 文档里写的办法），在我测过的每个 CPU 上结果都一样。

数字：`int8_portability/machines.csv`（−3.97 pp；7 位权重比 FP32 少 0.17 pp）；`int8_capture_study/summary.md`（Gold 6140 对 AMD 9654，18 个模型第一个不同的层都是时间卷积；该层 188 万个不同的输出 100% 在饱和预测里；单层 7 位仍有 7%–15% 的第一天预测不同，五层全改才一致）。被追问“有 VNNI 的 x86 为什么没事”：x86 上 ONNX Runtime 总把激活改成无符号 8 位，VNNI 的指令把 u8×s8 直接加到 32 位，精确。

## Q12 · Did pruning help?

> It saves bytes, not time. Pruning removes half of the last layer's filters, but that layer holds only half a percent of the multiplies, so the saving in time is about one percent. At a matched training budget it cost 1.6 points on the 16-electrode model, so I would not prune this network for speed.

意思：省字节，不省时间。剪枝去掉最后一层一半的滤波器，但这一层只占 0.5% 的乘加，时间只省约 1%。在相同的训练预算下，16 个电极的模型掉 1.6 个百分点，所以不会为了速度剪这个网络。

数字：`matched_budget/analysis.json`（−1.59 [−2.86, −0.49]）；M4 上加内核 138 → 137 微秒（`ort_kernel_study/latency.csv`）。

## Q13 · Why did the redundancy penalty not matter?

> Because the relevance scores differ between electrodes three to four times more than the penalty does, so the penalty rarely changes which electrodes are chosen. With ten seeds the difference is 0.3 points with an interval that includes zero. A different redundancy measure, checked on the training day, changed at most one electrode per person.

意思：因为电极之间相关性分数的差别比惩罚项大 3–4 倍，惩罚很少改变选出的电极。10 个种子时差 0.3 个百分点，区间包含 0。换一种冗余度量（只用训练日检查），每人最多变 1 个电极。

数字：`electrode_count/analysis.json`；`redundancy_check/`。

## Q14 · Why does Riemannian selection work better?

> The decoder uses electrodes in combination, through spatial filters, so a criterion that scores the chosen set together matches what the network actually needs. Scoring one electrode at a time keeps electrodes that look good alone but say the same thing. It found this on 9 people, and I then registered it on 54 new people: plus 3.7 points at 8 electrodes, 37 of 54 better.

意思：解码器是把电极组合起来用的（空间滤波器），所以给一组电极一起打分的标准，和网络真正需要的一致。一个一个打分会留下单独看很好、但说的是同一件事的电极。这是在 9 个人身上发现的，之后在 54 个新的人身上事先登记检验：8 个电极时高 3.7 个百分点，54 人里 37 人变好。

数字：`replication_riemann_study/hypotheses.csv`（R1 +3.66 [+2.10, +5.27]）。

## Q15 · Why does an input that carries nothing cost accuracy?

> Because recorded noise is the same in every epoch, and the network uses it to remember its training trials: with their own noise the models classify all of them correctly, with noise they have not seen only 62 percent. When I redrew the same noise in every epoch, the network switched it off by itself, and nine-tenths of the cost disappeared, from 13 points to about 1, as registered on 54 people. What costs is the part the other inputs cannot predict: copies of kept electrodes cost nothing visible.

意思：因为录下来的噪声每轮都一样，网络拿它来记住训练试验：配原来的噪声，训练试验全部答对；换一份没见过的噪声，只剩 62%。我把同样的噪声改成每一轮重新抽，网络自己就把它关小了，代价去掉九成，从 13 个点降到 1 个点左右，这在 54 人上事先登记过。有代价的是其他输入预测不了的那部分：保留电极的拷贝没有可见的代价。

数字：WP3b `fresh_noise_study/summary.md`（Lee 2019，46 个噪声，第二天：固定噪声 −13.33、每轮重抽 −1.15 [−2.12, −0.23]；F1 +12.18 [+8.77, +15.70] 成立；F2 不成立，区间下限 −2.12 低于 −2；F3 成立，噪声与电极的权重比中位数 0.19 对 1.07，噪声占空间滤波器输出 8% 对 86%；固定噪声的模型配原来的噪声训练试验 100%、配没见过的噪声 61.5%；2a 上重抽去掉 74%、70% 的代价）。被追问“剩下那 1 个点是什么”：噪声从不重复，网络也没把它关到零（权重比 0.10–0.19），留下的权重把噪声带进了特征；这部分有多少来自“脑电数据太少”，这项研究分不出来。旧的证据：训练后噪声仍占空间滤波器输出的 38%（初始化时 43%），训练后把权重设为 0 只救回 12%–19%（`noise_mechanism_study/analysis.json`）；WP1 `input_kind_study/summary.md`（Lee 2019：拷贝 +0.62 [−0.05, +1.27]、同一条噪声重复 46 次 −1.83、独立噪声 −13.33；K1、K3 都成立）。

## Q16 · Can the network choose the electrodes itself?

> Not from one person's day of data. A learned selection layer was the worst selection method; a group-lasso penalty did not switch any input off, because a batch norm makes the size of a filter's weights irrelevant; and a version that the batch norm cannot undo did not switch the noise off either, as registered on 54 people. What did help was pretraining on 45 other people: then the noise inputs cost 2 points instead of 12.

意思：只有一个人一天的数据时不行。学出来的选择层是所有选法里最差的；group lasso 惩罚没有关掉任何输入，因为 BatchNorm 让滤波器权重的整体大小变得无关紧要；改成 BatchNorm 抵消不了的版本，也没有把噪声关掉，这在 54 人上事先登记过。真正有用的是用 45 个别人的数据预训练：那时噪声输入的代价从 12 个点降到 2 个点。

数字：`selection_study/accuracy.csv`（learned）；`input_penalty_study/summary.md`；`scalefree_penalty_study/summary.md`（N1 +1.19 [−0.24, +2.63]、N2 −0.12、N3 不成立：噪声对电极的范数之比最低约 0.5，没到 1/3）；`pretrain_study/summary.md`（噪声代价 −12.08 → −2.17）。被追问“可噪声每轮重抽时网络就能关掉它”：对（WP3b，权重比 0.19），因为那时噪声对记住训练试验没用；可录下来的电极每轮都一样，没法重抽。

## Q17 · Why did day-to-day alignment hurt on Lee 2019?

> Whitening amplified dozens of near-empty directions. With 62 electrodes, 49 of the 62 directions of the covariance carry less than 1 percent of the largest one's variance, and alignment scales them all to unit variance: many inputs the others cannot predict, which is exactly what costs accuracy. Keeping those directions without amplifying them removed all of the harm, and it also hurt within one day, so it is not about the second day.

意思：白化把几十个几乎没有方差的方向放大了。62 个电极时，协方差的 62 个方向里有 49 个的方差不到最大方向的 1%，对齐把它们全部拉到单位方差，等于给网络加了很多其他输入预测不了的输入，这正是有代价的那种。保留这些方向但不放大，害处就全部消失；而且在同一天之内对齐也有害，所以和第二天无关。

数字：`alignment_mechanism_study/summary.md`（探索性；M1 同一天内 −4.02 [−6.27, −2.05]；M3 Ledoit–Wolf 平均 0.0061；去掉的害处比例：ea-floor 102%、ea-shrink10 88%、ea-top 48%、ea-lw 23%；Lee 2019 普通对齐 62.28% 对不对齐 66.86%，ea-floor 66.97%）。被追问“那 2a 上为什么有帮助”：2a 上平均 +0.8，区间很宽（9 人），ea-floor 和收缩都比普通对齐好一点，都不显著。

## Q18 · Why did channel dropout hurt so much?

> Not because of the batch norm statistics: re-estimating them with every electrode present won back only 14 percent of the loss, as registered on 54 people. The spatial filters rely on exact cancellations between electrodes, and dropping electrodes in training gives worse filters. But it buys robustness: on the first dataset, one failed electrode costs a normal model 15 points and a model trained with 5 percent dropout only 2.

意思：不是因为 BatchNorm 的统计量对不上：用全部电极重估统计量，只救回了 14% 的损失，这在 54 人上事先登记过。空间滤波器靠电极之间精确的抵消，训练时随机关掉电极，学到的滤波器就更差。但它换来了耐受：在第一个数据集上，坏掉一个电极，普通模型掉 15 个点，用 5% dropout 训练的模型只掉 2 个点。

数字：`chdrop_study/summary.md`（B1 +0.38 [−0.04, +0.81]，救回 14%；B2 +2.45；B3 +0.49 [+0.28, +0.72]；2a 坏一个电极：不加 dropout 平均 14.8、最坏 31.7，p = 0.05 平均 1.9、最坏 10.3；Lee 2019 0.81 对 0.32；p = 0.05 在电极全好时 2a 低 4.4–4.8 pp，Lee 2019 低 0.3 pp）。

## Q19 · How do you know your kernels are correct?

> Every kernel was checked bit for bit against reference arithmetic: the two ONNX Runtime kernels give identical logits for all 459 models, 132,192 predictions each, and the microcontroller kernel gives identical logits in all 136 builds. Because the outputs are identical, accuracy cannot change; only speed does.

意思：每个内核都和参考算法逐位核对过：两个 ONNX Runtime 内核在全部 459 个模型上 logits 完全相同（各 132,192 个预测）；单片机内核在全部 136 个构建上 logits 完全相同。输出完全相同，准确率就不可能变，变的只有速度。

数字：`ort_kernel_study/summary.md`；`mcu_kernel_study/summary.md`；x86 上的 uint8 版本（`int8_kernel_x86_study/summary.md`）：有 VNNI 的 AMD 9654 上 459/459 个模型与运行库逐位相同；没有 VNNI 的 Gold 6140 上只把分组卷积算准，准确率损失从 3.97 减到 1.88 个点。

## Q20 · Is the test day really untouched?

> For the main study, yes: the configuration was frozen on November 3 before the second day was scored, the code can only read the test day through a guarded loader, and a lint check fails if any other file opens it. Everything after that, on that dataset, is labeled exploratory. On the second dataset, every prediction was written into a plan before its test day was scored.

意思：主研究是的：配置在 11 月 3 日、第二天打分之前冻结；代码只能经过一个上锁的读取函数读测试日，任何别的文件去读，检查工具就会报错。之后在这个数据集上做的都标为探索性的。在第二个数据集上，每条预测都在它的测试日打分之前写进了计划。

数字：`results/frozen_config.json`（frozen_at 2026-11-03）；`tools/guard.py`（session2-access）。

## Q21 · How much compute did this take?

> About fifty-nine thousand trainings and seven and a half thousand CPU hours on the UNC Longleaf cluster, plus local work on a Mac mini. The cluster is what made ten seeds, 54 people, the exhaustive brain-region search and the registered tests of why 16 electrodes are enough possible; it turned anecdotes into intervals.

意思：在 UNC 的 Longleaf 集群上约五万九千次训练、七千五百 CPU 小时，另有在 Mac mini 上的本地工作。正是集群让 10 个种子、54 个人、脑区的全部组合，以及“为什么 16 个电极就够”的那些事先登记的检验成为可能，把个例变成了区间。

数字：`results/supplementary/compute_summary.json`（11-22 21:00 重算：Longleaf 58,626 次训练、7,452.3 CPU 小时，只算训练时间，不含打分、计时和模拟；最大的几项是脑区研究 18,396 次、1,528 CPU 小时，尺度无关惩罚 5,166 次、572 CPU 小时）。被追问“Mac 上做了什么”：10 个种子的重新训练（703 个模型，约 23 CPU 小时）和全部计时、单片机模拟。

## Q22 · Some people lose accuracy with fewer electrodes. What then?

> Yes, and I report it: with the frozen configuration 4 of 9 people lost more than 2 points, and with 16 of 62 electrodes 24 of 54 did, while about as many gained. A deployed system should check each user on their own calibration data and keep the full montage as a fallback.

意思：是的，我报告了：冻结方案 9 人里有 4 人掉 2 个百分点以上；62 个里留 16 个时 54 人里 24 人掉这么多，同时差不多同样多的人变好。实际系统应该用每个用户自己的校准数据检查，并保留全部电极作为后备。

数字：`results/analysis/conditions_primary.csv`；`replication_study/hypotheses.csv`。

## Q23 · What would you do next?

> Two things: pretraining on many users so that more electrodes pay off, with the electrode choice learned on that data, and timing on a physical microcontroller board. After that, closed-loop online sessions with real users.

意思：两件事：用很多用户的数据预训练，让更多电极变得划算，并在这样的数据上学怎么选电极；以及在真实的单片机板子上计时。再往后是和真实用户一起做闭环在线实验。

## Q24 · What is the main lesson beyond EEG?

> Measure every claimed saving on the target hardware. Fewer bytes did not mean faster until the runtime had the right kernel, and the same 8-bit file was not even equally accurate on every CPU. When a model's common layer falls to a general code path, a short dedicated kernel can matter more than further compression.

意思：每一项“节省”都要在目标硬件上量。字节少了，在运行库有合适的内核之前并不会更快；同一个 8 位文件在不同 CPU 上连准确率都不一样。当模型的常见层落到通用代码路径时，一个简短的专用内核可能比继续压缩更重要。
