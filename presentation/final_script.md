# Final 讲稿（26 页）

- **怎么用**：英文部分照着说。"意思"是中文对照，帮你理解自己在说什么，不用念。
- **时长**：完整说一遍约 18 分 15 秒（按每分钟 125 个词算，2281 个词）。去掉方括号 [ ] 里的句子约 16 分 53 秒（2111 个词）。第 24 页的视频边放边讲，不另外占时间。
- **时间限制**：期末汇报一共 25 分钟，其中至少 5 分钟留给提问，所以正式讲的部分不能超过 20 分钟。先计时完整练一遍：如果超过 19 分钟，正式讲时就固定去掉方括号里的句子。
- **计时**：标题后的时间是这一页大约开始的时刻。如果第 10 页晚于 6:00 才开始，或第 18 页晚于 11:25 才开始，就跳过后面所有方括号里的句子。
- **视频**（第 24 页）：放映时点一下画面开始播放，边放边讲，放 20 秒左右就点一下暂停，翻到下一页。
- **备注**：英文部分也写进了 PPTX 每页的备注，演讲者视图（Presenter View）里能看到。"If asked" 那一行只在被问到时用。
- **读法**：EEG 读 E-E-G；EEGNet 读 E-E-G net；INT8 读 int eight；INT16 读 int sixteen；FP32 读 F-P thirty-two；ONNX 读 "onyx"；VNNI 逐个字母读 V-N-N-I；x86 读 x eighty-six；Riemannian 读 ree-MAH-nee-un；Shapley 读 SHAP-lee；mRMR 逐个字母读；μ 读 mu（发音像 "mew"）；0.74 读 zero point seven four；CMSIS-NN 读 C-M-SIS N-N；QEMU 读 Q-E-M-U；NEON 读 neon；MOABB 读 "moab"；Cortex-M0 读 cortex M zero。
- **维护**：这份讲稿是 PPTX 备注和幻灯片标题的来源；改了以后重新生成 PPTX（命令在 `presentation/pptx/build_final_pptx.js` 的开头）。幻灯片上的数字都由生成器从 `results/` 读出；讲稿里的数字是手写的，改了结果要两边一起核对。

## 1 · Cover（0:00）

> Hi everyone, I'm Peiye Li. My project is about brain-computer interfaces that read EEG from the scalp. I asked which electrodes a decoder can lose, how many bits it needs, and whether the answers hold in new people. I tested them on two datasets, with 63 people in all.

意思：大家好，我是 Peiye Li。我的项目研究从头皮读取脑电（EEG）的脑机接口。我问了三件事：解码器可以去掉哪些电极，需要几位数字，以及这些答案换一批人是否仍然成立。我在两个数据集上做了检验，一共 63 个人。

## 2 · Why Fewer Electrodes?（0:25）

> Why remove electrodes? Every electrode is a sensor on the scalp. It needs a contact, an amplifier channel, power, and setup time before each use. [Wet electrodes also need gel, which makes the cap less comfortable.] The decoder pays for every electrode too, because in its standard design the first layer runs once per electrode. Pruning the network can only save that second cost. Removing electrodes can save both.

意思：为什么要去掉电极？每个电极都是头皮上的一个传感器。它需要一个接触点、一个放大器通道、电量，每次使用前还要花时间调。[湿电极还要涂导电膏，帽子戴着也不舒服。] 解码器也要为每个电极付出代价，因为在标准结构里，第一层对每个电极各运行一次。剪枝只能省第二种成本，去掉电极两种都能省。

If asked: A common EEG amplifier chip draws about 5 milliwatts per channel. At that rate, the 22 channels use about 10 times more energy per decision than the original network. With low-power research chips, the balance flips. This is an order-of-magnitude estimate, not a measurement.

（补充：常见的脑电放大器芯片每个通道约 5 毫瓦。按这个算，22 个通道每次判断用的电是原始网络的约 10 倍。如果换成低功耗的研究用芯片，情况就反过来。这是数量级的估算，不是测量。）

## 3 · Research Questions（0:55）

> I asked three questions. First, which electrodes can a decoder lose on a new recording day, and does avoiding redundant electrodes help? Second, how many bits does it need, and does 8-bit arithmetic make it faster? Third, do the answers hold in new people? The midterm review added four more: why 16 electrodes could beat all 22, how the electrodes are chosen, why INT8 was slower, and why pruning didn't make the model faster. I'll answer all of them today.

意思：我问了三个问题。第一，换到新的录制日，解码器可以去掉哪些电极？避开重复的电极有没有帮助？第二，它需要几位数字？8 位运算会不会更快？第三，这些答案换一批人还成立吗？期中评审又加了四个问题：为什么 16 个电极能比 22 个全用还好，电极到底是怎么选的，INT8 为什么更慢，以及剪枝为什么没有让模型变快。今天这些问题我都会回答。

If asked: The redundancy rule was the main hypothesis of my proposal. The other questions came from the midterm review and from the results.

（补充：避开重复的规则是 proposal 里的主要假设；其他问题来自期中评审和结果本身。）

## 4 · Related Work and Novelty（1:35）

> Most of the pieces already exist. EEGNet is the compact network I use. Q-EEGNet ran an 8-bit EEGNet on a microcontroller, and MI-BMInet chose electrodes from network weights before using 8 bits. My redundancy rule comes from mRMR, and the Riemannian method picks electrodes as a group. What I add is a careful comparison: one change at a time, test days locked until each plan froze, and the main tests repeated in 54 new people. [I also found the INT8 kernel that was missing, and wrote it.]

意思：这些部分大多已经有人做过。EEGNet 是我用的小型网络。Q-EEGNet 把 8 位的 EEGNet 跑在了单片机上，MI-BMInet 先按网络权重挑电极，再改用 8 位。我的“避开重复”规则来自 mRMR，黎曼方法则是把电极当成一组来挑。我新加的是一个仔细的比较：一次只改一个因素；测试日的数据在每个计划冻结之前一直锁着；主要的检验在 54 个新的人身上又做了一遍。[我还找到了缺少的那个 INT8 内核，并把它写了出来。]

If asked: The weight-based rule trains on all electrodes, then ranks each electrode by the size of its learned spatial weights. MI-BMInet also puts the spatial layer first by design. I applied that order to an EEGNet that was already trained.

（补充：按权重挑选的规则先用全部电极训练，再按每个电极学到的空间权重大小排序。MI-BMInet 在设计上就把空间层放在前面；我是把这个顺序用到已经训练好的 EEGNet 上。）

## 5 · Datasets and Evaluation（2:15）

> I use two public datasets, 63 people in all. My primary dataset, BCI Competition IV 2a, has nine people, 22 electrodes, and four imagined movements. Lee 2019 has 54 people, 62 electrodes, and left versus right hand. Everyone was recorded on two days. I make every choice with day 1 only, freeze the plan, and score day 2 once. [That's how a real system works: you calibrate on one day and use it on another.] Studies added after the first test results are exploratory, unless I registered them in advance on Lee 2019.

意思：我用两个公开数据集，一共 63 个人。主要的数据集是 BCI Competition IV 2a：9 个人，22 个电极，想象四种动作。Lee 2019 有 54 个人，62 个电极，想象左手或右手。每个人都在两个不同的日子各录一次。所有选择都只用第一天的数据做，然后冻结计划，再在第二天上打一次分。[实际系统就是这样用的：一天校准，另一天使用。] 看过第一个数据集的测试结果之后才加的研究都是探索性的，除非我事先在 Lee 2019 上登记过。

If asked: In the first dataset, each person has 288 trials per day, and I use the 2 seconds after each cue. In the second, each day has 100 offline and 100 online trials. The plans were frozen on November 3 and November 21.

（补充：第一个数据集每人每天 288 次试次，取每次提示后的 2 秒；第二个数据集每天有 100 次离线和 100 次在线试次。两个计划分别在 11 月 3 日和 11 月 21 日冻结。）

## 6 · Systems View: Model Size and Compute（3:00）

> Now the systems view. The model is tiny: 2,420 parameters, under 10 kilobytes. But one decision takes 5.9 million multiply-adds, and 96 percent of them happen in the first layer, a temporal filter that runs on every electrode. So in the exported model, compute grows with the electrode count, and going from 22 to 8 electrodes removes 63 percent. [Keep this layer order in mind, because it comes back in the INT8 results.]

意思：现在从系统角度看。模型很小：2,420 个参数，不到 10 KB。但每做一次判断要 590 万次乘加运算，其中 96% 在第一层；第一层是一个时间滤波器，对每个电极都要跑一遍。所以在导出的模型里，计算量随电极数增长：从 22 个减到 8 个，计算少了 63%。[记住这个层的顺序，后面 INT8 的结果还会用到。]

If asked: Multiply-adds are counted from the graph, not measured. Parameter bytes include the quantization scales.

（补充：乘加次数是按计算图数出来的，不是测出来的；参数字节数包括量化用的缩放系数。）

## 7 · Method: Electrode Selection（3:35）

> For each person, I score every electrode on day 1 by how well its mu and beta power separate the movements. [These rhythms weaken when you imagine moving.] The frozen rule simply keeps the top scorers. My proposed rule also subtracts a penalty for overlap with electrodes already chosen. The controls are a weight-based rule, random sets, and all electrodes. The head map shows how often each electrode was kept at 16. The front row was dropped most often.

意思：对每个人，我用第一天的数据给每个电极打分，看它的 μ 波和 β 波能量能多好地区分几种动作。[想象动作时，这两种节律会减弱。] 冻结规则就是留下分数最高的那些。我提出的规则还会减去一项惩罚：和已经选中的电极重叠多少。对照组是按权重挑选、随机组合，以及全部电极。头部图显示留 16 个时每个电极被保留了几次：最前面那一排被去掉得最多。

If asked: Relevance is a Fisher score on log power in the mu band, 8 to 13 hertz, and the beta band, 13 to 30 hertz. Redundancy is the mean absolute Spearman correlation with the electrodes already chosen, weighted by one half.

（补充：有用程度是 μ 频段（8–13 Hz）和 β 频段（13–30 Hz）对数能量的 Fisher 分数；重复程度是和已选电极的 Spearman 相关系数取绝对值后的平均，乘以 0.5。）

## 8 · Process After the Midterm（4:15）

> After the midterm, every step was a question, a tool and a result. Ten seeds tested the 16-versus-22 gap. The ONNX Runtime profiler found the slow INT8 layer. On Longleaf, a registered test in 54 new people confirmed the electrode results, and five server CPUs showed that INT8 accuracy depends on the processor. Then custom kernels and an emulator answered the speed questions, and direct tests showed why 16 electrodes are enough.

意思：期中之后，每一步都是一个问题、一个工具、一个结果。10 个种子检验了 16 对 22 的差距。ONNX Runtime 的性能分析器找到了慢的那一个 INT8 层。在 Longleaf 上，54 个新的人身上事先登记的检验证实了电极方面的结果，五种服务器 CPU 说明 INT8 准不准取决于处理器。之后，自己写的内核和模拟器回答了速度方面的问题，直接检验说明了为什么 16 个电极就够。

If asked: The ten seeds put the gap at plus 1.31 points, with a seed standard error of 0.57; a later retrain on one machine gave plus 0.32. Each step came from what the previous one showed.

（补充：10 个种子下差距是 +1.31 个点，种子标准误是 0.57；之后在同一台机器上重训，差距是 +0.32。每一步都是根据上一步的结果决定的。）

## 9 · Tools and Compute（4:50）

> Here's what each tool did. PyTorch and Braindecode trained EEGNet. ONNX Runtime ran INT8 on CPUs; its profiler found the slow layer, and two kernels I wrote in C fixed it. CMSIS-NN in the QEMU emulator ran the models on three Arm microcontroller cores. My Mac ran the ten-seed retraining and all the timing. Longleaf ran about 59,000 trainings in about 7,500 CPU hours: the region study, the 54-person replication, the tests of why 16 electrodes are enough, and INT8 on five server CPUs.

意思：每个工具做了什么：PyTorch 和 Braindecode 训练 EEGNet。ONNX Runtime 在 CPU 上运行 INT8；它的性能分析器找到了慢的那一层，我用 C 写的两个内核把它修好了。CMSIS-NN 在 QEMU 模拟器里，把模型跑在三种 Arm 单片机内核上。我的 Mac 跑了 10 个种子的重新训练和全部计时。Longleaf 跑了约 59,000 次训练，用了约 7,500 个 CPU 小时：脑区研究、54 人的复现、为什么 16 个电极就够的检验，以及五种服务器 CPU 上的 INT8。

If asked: The Mac and Longleaf use the same locked uv environment. Every INT8 microcontroller build was checked bit for bit against a NumPy integer reference.

（补充：Mac 和 Longleaf 用的是同一个锁定的 uv 环境。每个单片机上的 INT8 版本都和一个 NumPy 整数参考实现逐位核对过。）

## 10 · Result 1: Accuracy vs. Electrode Count（5:30）

> Here are the day-2 results on the first dataset, with ten seeds per point. All 22 electrodes give 64 percent, where chance is 25. Accuracy stays level or rises slightly down to 16 electrodes, then falls: 8 electrodes lose 4 points. At 16 the curve is 1.3 points higher, but its interval reaches zero, and a retrain on one machine gave only 0.3. So 16 well-chosen electrodes are as good as 22, not better. [INT8 changed these numbers by half a point or less.]

意思：这是第一个数据集第二天的结果，每个点训练了 10 个随机种子。22 个电极全用是 64%，瞎猜是 25%。电极减到 16 个之前，准确率持平或略升，之后开始下降：只剩 8 个时掉 4 个点。16 个时曲线高 1.3 个点，但按人算的区间碰到 0，而且之后在同一台机器上重训只高 0.3 个点。所以挑选过的 16 个电极和 22 个一样好，并不更好。[换成 INT8 后，这些数字的变化都不超过半个点。]

If asked: For 16 minus 22, the interval across people is minus 0.1 to plus 2.8 points, and the seed standard error is 0.57. At 8 electrodes, the loss is 4.2 points, with an interval from minus 7.1 to minus 1.0.

（补充：16 减 22 按人算的区间是 −0.1 到 +2.8 个点，种子标准误是 0.57；8 个电极时掉 4.2 个点，区间 −7.1 到 −1.0。）

## 11 · Result 2: Redundancy-Aware Selection（6:10）

> My original hypothesis was that avoiding redundant electrodes would help. It mostly didn't. My rule beat random sets by 2 to 3 points, but it gained less than one point over relevance alone, which is within noise. The reason is that these rhythms rise and fall across the whole scalp together, with an average correlation of 0.74. So every candidate looks equally redundant, and for 6 or 7 of the 9 people, both rules picked exactly the same electrodes.

意思：我原来的假设是：避开重复的电极会有帮助。结果基本没有。我的规则比随机组合高 2 到 3 个点，但比只看有用程度只高不到 1 个点，在误差范围内。原因是这些节律在整个头皮上一起增强或减弱，电极之间平均相关系数是 0.74。所以每个候选电极看起来都一样"重复"；9 个人里有 6 到 7 个，两种规则选出的电极完全相同。

If asked: With ten seeds, my rule is 0.3 points above relevance alone at 16 electrodes and 0.5 at 8, and both intervals include zero. Removing the shared scalp power before measuring redundancy would change at most two electrodes per person, so I didn't retrain for it.

（补充：10 个种子时，我的规则比只看有用程度在 16 个电极时高 0.3、8 个时高 0.5，两个区间都包含 0。先去掉全头皮共有的能量再算重复程度，每人最多只换两个电极，所以没有为它重新训练。）

## 12 · Result 3: Why 16 Electrodes Are Enough（6:50）

> This answers a midterm question: why can 16 electrodes do as well as 22? Every extra input costs accuracy, about a point per noise input here, and more data doesn't shrink that. What costs is the part of an input the others cannot predict: copies of kept electrodes cost almost nothing, independent noise the most. Registered on Lee 2019, it held. The dropped electrodes are mostly predictable; measured on Lee 2019, their unpredictable part costs 4 points and adds 4, so it cancels. Most of that cost is memorization: noise redrawn every epoch loses nine-tenths of it.

意思：这一页回答期中的一个问题：为什么 16 个电极能和 22 个一样好？每多一个输入都要付出准确率的代价，在这里每多一个噪声输入约掉 1 个点，训练数据再多也不变小。有代价的是这个输入里其他输入预测不了的那部分：保留电极的拷贝几乎没有代价，独立的噪声代价最大。这在 Lee 2019 上事先登记过，结果成立。被去掉的电极大部分能被预测；在 Lee 2019 上直接量，它们预测不了的那部分代价 4 个点、带来的信息也是 4 个点，正好抵消。这个代价大部分是记答案：噪声改成每一轮重新抽，代价就少了九成。

If asked: Mixing just a quarter noise into a copy already costs 5 of the 7 points that pure noise costs. With the unpredictable part as input, the networks fit 99 percent of their training trials, 96 without it. Across people, how unpredictable someone's dropped electrodes are did not predict their cost; that registered test failed. Earlier I compared the real electrodes with noise in their place and called the 6-point difference their information; that also credited them for being predictable, so it overstated both. Zeroing the noise weights after training wins back at most 30 percent of the loss. A group-lasso penalty did not help: batch norm makes the weights' size irrelevant, so it shrank all weights together.

（补充：拷贝里只混 25% 的噪声，就已经付出纯噪声代价 7 个点里的 5 个。输入里有预测不了的那部分时，网络在训练试验上 99% 答对，没有时 96%。在不同的人之间，一个人被去掉的电极有多难预测，并不能预测这个人的代价；这条事先登记的预测没有成立。之前我拿同位置的噪声和真实电极比，把相差的 6 个点叫作它们的信息；这样把“容易预测、所以代价小”也算进去了，两项都高估了。训练后把噪声输入的权重设为 0，最多只救回 30% 的损失。给每个输入的权重加 group lasso 惩罚也没有用：BatchNorm 让权重的整体大小变得无关紧要，所以惩罚把所有权重一起缩小。）

## 13 · Result 4: Accuracy Levels Off After 16 Electrodes（7:35）

> Is 16 a lucky peak? 4,824 more models traced the curve with less of the training day. On both datasets, accuracy rises and then stays flat: from about 16 electrodes on the first dataset, and 16 to 20 on the second. Halving the training data did not move that point, so two registered predictions did not hold. Past that point, each new electrode adds about as much as it costs, as on the previous slide. More data changes that: pretrained on 45 other people, all 62 electrodes beat a shared 16 by 4.5 points, as registered.

意思：16 是不是碰巧的尖峰？我又训练了 4,824 个模型，只用第一天的一部分数据重画曲线。两个数据集上，准确率都是先升后平：第一个数据集从约 16 个电极开始，第二个从 16 到 20 个开始。训练数据减半，这个位置也没有移动，所以两条事先登记的预测没有成立。过了这个点，每多一个电极带来的信息和它的代价差不多，就像上一页说的那样。数据多了就不一样：用 45 个别人的数据预训练后，全部 62 个电极比共用的 16 个高 4.5 个点，和事先登记的预测一致。

If asked: With all training data, the first dataset gives 65.2 percent at 16 electrodes and 64.8 at 22; Lee 2019 gives 67.0 at 16, 67.5 at 20 and 66.9 at 62. The registered predictions were that less data would favor fewer electrodes and move the best count lower. The shared 16 are the same for everyone, the best on the pretraining people. Pretraining also cut the cost of 46 noise inputs from 12 points to 2; on the first dataset, eight other people were too few, and pretraining hurt.

（补充：用全部训练数据时，第一个数据集 16 个电极 65.2%，22 个 64.8%；Lee 2019 16 个 67.0%，20 个 67.5%，62 个 66.9%。事先登记的预测是：数据越少，少电极越占优，最好的电极数会变小。共用的 16 个是所有人用同一套，按预训练的人选出来的。预训练还把 46 个噪声输入的代价从 12 个点降到 2 个点；在第一个数据集上，只有 8 个别人，太少，预训练反而有害。）

## 14 · Result 5: Brain Regions（8:20）

> Another midterm question was whether brain regions could guide the choice. I grouped the electrodes into nine regions and trained a model on all 511 combinations, for every person. That gives each region's exact contribution, its Shapley value. The data recover the textbook motor map: the left hand comes from the right hemisphere, the right hand from the left, and the feet from the midline. But as a selection rule, whole regions did not beat single electrodes.

意思：期中的另一个问题是：能不能按脑区来选电极。我把电极分成 9 个脑区，对每个人都把 511 种组合各训练了一个模型。这样就能算出每个脑区准确的贡献，也就是它的 Shapley 值。数据重现了教科书上的运动皮层分布：左手主要靠右半球，右手靠左半球，双脚靠头顶中线。但作为选电极的规则，按整个脑区选并没有比按单个电极选更好。

If asked: A Shapley value averages what a region adds over every combination it can join, so the nine values add up exactly to the full accuracy minus chance. Region sets chosen on day 1 scored between 3.6 points below and 2.0 points above the single-electrode rule.

（补充：Shapley 值是把一个脑区加入每一种可能组合时带来的提升取平均，所以 9 个值加起来正好等于全部电极的准确率减去瞎猜的水平。按第一天选出的脑区组合，比单电极规则低 3.6 到高 2.0 个点。）

## 15 · Result 6: Smarter Selection（8:55）

> Scoring one electrode at a time was weak when few remained, so I tested rules that judge electrodes together. The frozen rule falls fastest below 16. The Riemannian rule keeps the electrodes whose covariance patterns differ most between the movements. At 8 electrodes it reached 64 percent, 4.4 points above the frozen rule, and 8 of the 9 people improved. [A learned selector did worst.] But I picked this method after seeing the test results, so it needed a fresh test.

意思：电极剩得少时，一个一个打分的方法效果差，所以我试了把电极放在一起评价的规则。冻结规则在少于 16 个电极时掉得最快。黎曼规则保留的是：几种动作之间协方差模式差别最大的那组电极。8 个电极时它达到 64%，比冻结规则高 4.4 个点，9 个人里有 8 个变好。[学出来的选择层最差。] 但这个方法是我看过测试结果之后才挑出来的，所以需要一次新的检验。

If asked: Against references retrained on the same hardware, the Riemannian rule is 5.4 points better at 8 electrodes, and all nine people improved. With 48 comparisons, about one interval in twenty would exclude zero by chance, which is why I needed the second dataset.

（补充：和在同一种硬件上重新训练的参照模型相比，黎曼规则在 8 个电极时高 5.4 个点，9 个人全部变好。一共做了 48 个比较，纯靠运气大约每 20 个区间就有一个不包含 0，所以需要第二个数据集来检验。）

## 16 · Result 7: Replication in 54 New People（9:35）

> So for the second dataset, I wrote my hypotheses into a plan before scoring any of its test days. Most of them held. Fewer electrodes were not worse on average, relevance beat random choices, and 8 bits lost nothing. Most importantly, the Riemannian rule beat the frozen rule at 8, 16 and 20 electrodes, by 3.7 points at 8. Two things did not hold. The people decoded worst did not gain the most, and aligning the two days actually hurt, by 5.5 points.

意思：所以在第二个数据集上，我先把假设写进计划，然后才给任何一天的测试数据打分。大部分假设成立：少电极平均不比全部差，按有用程度选比随机好，8 位没有损失。最重要的是，黎曼规则在 8、16、20 个电极时都赢了冻结规则，8 个电极时高 3.7 个点。有两条没有成立：解得最差的人并没有受益最多；把两天的信号对齐反而有害，掉了 5.5 个点。

If asked: The plan was frozen at 9:54 on November 21, and the Riemannian plan at 10:32, while the scoring was on hold. With 8 Riemannian electrodes, accuracy was 68.1 percent, above the 66.9 percent with all 62, but the plan didn't test that comparison.

（补充：计划在 11 月 21 日 9:54 冻结，黎曼方法的计划在 10:32 冻结，期间打分一直暂停。8 个黎曼电极的准确率是 68.1%，高于 62 个全用的 66.9%，但计划里没有检验这个比较。）

## 17 · Result 8: Bit Width（10:15）

> Now compression. I was asked whether 16-bit integers would do better than 8. To measure bits without depending on any machine, I simulated integer rounding for all 459 models. INT16 and INT8 both lose almost nothing, 0.1 points or less, so 16 bits buy nothing. Below 8 bits, accuracy breaks: 4-bit weights lose 5 points. Choosing the bits per layer from day-1 data saves 41 percent of the weight bytes for half a point. [Everything below 8 bits here is simulated, not run on real low-bit hardware.]

意思：接下来是压缩。期中有人问 16 位整数会不会比 8 位好。为了不依赖任何一台机器来衡量位数，我对全部 459 个模型模拟了整数取整。INT16 和 INT8 几乎都不掉，最多 0.1 个点，所以 16 位没有好处。低于 8 位就不行了：4 位权重会掉 5 个点。用第一天的数据给每一层定位数，可以少 41% 的权重字节，只掉半个点。[这里低于 8 位的结果都是模拟的，没有在真正的低位硬件上跑。]

If asked: The simulated 8-bit network matches the real ONNX Runtime INT8 predictions on 99.3 percent of trials. Below 8 bits, the input and the temporal filter outputs break first.

（补充：模拟的 8 位网络和真实 ONNX Runtime INT8 的预测在 99.3% 的试次上一致。低于 8 位时，最先坏掉的是输入和时间滤波器的输出。）

## 18 · Result 9: Why INT8 Was Slower（11:00）

> Now the INT8 question. The slow part is one small layer, the grouped spatial convolution. It does only 3 percent of the multiply-adds, yet in INT8 it took 189 microseconds, against 13 in FP32. ONNX Runtime has fast INT8 kernels for ordinary convolutions and for depthwise ones, with one filter per channel. EEGNet's layer is neither: it has 8 groups of 2 filters. So it takes a generic path: for each group, copy the input into a buffer, then multiply by a matrix only two columns wide. [The chip is fine: the first layer is 1.6 times faster in INT8.] One kernel is missing, not an accelerator.

意思：现在回答 INT8 的问题。慢的是很小的一层：分组空间卷积。它只占 3% 的乘加运算，但在 INT8 下花了 189 微秒，FP32 下只要 13 微秒。ONNX Runtime 对普通卷积和逐通道（depthwise，每个通道一个滤波器）卷积都有快速的 INT8 内核，但 EEGNet 的这一层两种都不是：它有 8 组、每组 2 个滤波器。所以只能走通用路径：对每一组，先把输入复制到一个缓冲区，再乘以一个只有两列宽的矩阵。[芯片本身没有问题：第一层在 INT8 下快了 1.6 倍。] 缺的是一个内核，不是加速器。

If asked: That's 1.1 nanoseconds per multiply-add in INT8, against 0.074 in FP32. Each filter spans all 22 electrodes, which is why neither fast path applies. The M4 reports FEAT_DotProd and FEAT_I8MM, its 8-bit dot-product instructions.

（补充：INT8 下每次乘加约 1.1 纳秒，FP32 下是 0.074 纳秒。每个滤波器覆盖全部 22 个电极，所以两条快路径都用不上。M4 报告支持 FEAT_DotProd 和 FEAT_I8MM，也就是它的 8 位点积指令。）

## 19 · Result 10: The Missing INT8 Kernel（11:50）

> So I wrote that kernel. It's an ONNX Runtime custom operator in C with ARM NEON instructions. [It rounds exactly like ONNX Runtime's own library.] It reads each input once and computes all 16 filter outputs in registers, instead of copying the input for every group. The spatial layer dropped from 189 to 10 microseconds, faster than FP32. The whole 22-electrode model now takes 180 microseconds, against 273 in FP32 and 357 in the original INT8. So INT8 is finally 1.5 times faster than FP32. The outputs are bit-identical on all 459 models, so accuracy can't change.

意思：所以我写了这个内核。它是一个 ONNX Runtime 的自定义算子，用 C 和 ARM NEON 指令写成。[取整方式和 ONNX Runtime 自己的库完全一样。]它对每个输入只读一次，在寄存器里同时算出全部 16 个滤波器的输出，不再为每一组复制一遍输入。空间层从 189 微秒降到 10 微秒，比 FP32 还快。整个 22 电极模型现在只要 180 微秒，FP32 是 273，原来的 INT8 是 357。所以 INT8 终于比 FP32 快了 1.5 倍。在全部 459 个模型上，输出逐位相同，所以准确率不可能变。

If asked: The timing used 15 interleaved rounds of 500 runs, one thread, batch 1, with the load average recorded. The frozen final model takes 137 microseconds, against 203 in FP32. A second kernel of the same kind fixes the first layer of the swapped model, which comes next. The kernels cover these layer shapes on ARM64; they are custom operators, not a change to ONNX Runtime itself.

（补充：计时用的是 15 轮交替、每轮 500 次，单线程，每次一个样本，并记录了系统负载。冻结的最终模型要 137 微秒，FP32 是 203。同样的第二个内核修好了调换顺序后模型的第一层，下一页会讲。这些内核只覆盖 ARM64 上这种形状的层；它们是自定义算子，没有改动 ONNX Runtime 本身。）

## 20 · Result 11: Swapping the First Two Layers（12:35）

> A second fix changes the model instead of the runtime, with no retraining. EEGNet's first two layers are both linear, so I can swap them and get the same predictions. Then the expensive temporal filter runs on 16 mixtures instead of on every electrode. INT8 takes 66 microseconds, four times faster than FP32, [and 47 with a second kernel for the new first layer]. This corrects my midterm claim: after the swap, going from 22 to 8 electrodes saves only 15 percent of the compute, so fewer electrodes mainly save hardware.

意思：第二种解决办法改的是模型本身，而不是运行库，也不需要重新训练。EEGNet 的前两层都是线性的，所以可以调换顺序，预测结果不变。调换以后，昂贵的时间滤波器只对 16 个混合信号运行，而不是对每个电极都跑一遍。INT8 只要 66 微秒，比 FP32 快 4 倍，[再给新的第一层配上第二个内核，就只要 47 微秒]。这也修正了我期中的说法：调换以后，从 22 个电极减到 8 个只省 15% 的计算，所以少电极主要省的是硬件。

If asked: In FP32, the swapped graph reproduces the predictions of all 459 models, with a largest logit difference of 4 millionths. MI-BMInet uses this order by design. The swap stores the temporal filters twice, which adds 504 values.

（补充：FP32 下，调换后的计算图对全部 459 个模型给出相同的预测，输出最大相差百万分之四。MI-BMInet 在设计上就用这个顺序。调换后时间滤波器要存两份，多出 504 个数。）

## 21 · Result 12: INT8 Across Processors（13:20）

> I also scored the same INT8 files on six kinds of processors. On the M4 and on an AMD chip with VNNI, every prediction matched. On x86 chips without VNNI, 457 of the 459 models changed, and accuracy dropped 4 points. Those chips add pairs of 8-bit products in 16 bits, which can overflow; reading every layer's integers, each output that differed was one that overflows, starting in the first layer. Storing the weights in 7 bits fixes it: results become identical everywhere, 0.17 points below FP32. [So INT8 accuracy depends on the processor, not only on the model.]

意思：我还把同样的 INT8 文件放到六种处理器上打分。在 M4 和一款带 VNNI 的 AMD 芯片上，每个预测都一样。在没有 VNNI 的 x86 芯片上，459 个模型里有 457 个变了，准确率掉了 4 个点。这些芯片把两个 8 位乘积加在一个 16 位数里，可能溢出；我逐层读出整数后发现，每个不同的输出都是会溢出的那种，从第一层就开始。把权重存成 7 位就解决了：在每种处理器上结果都相同，只比 FP32 低 0.17 个点。[所以 INT8 准不准取决于处理器，不只取决于模型。]

If asked: The instruction is VPMADDUBSW, which adds two 8-bit products into a 16-bit result. On x86, ONNX Runtime always shifts the activations to unsigned 8 bits; VNNI makes that exact. ONNX Runtime documents 7-bit weights, called reduce range, as the remedy; 7 bits in one layer is not enough, all five are needed. FP32 gave identical predictions on every processor.

（补充：这条指令是 VPMADDUBSW，它把两个 8 位乘积加进一个 16 位结果。在 x86 上，ONNX Runtime 总是把激活改成无符号 8 位；有 VNNI 时这样算是精确的。ONNX Runtime 的文档把 7 位权重（reduce range）列为解决办法；只改一层不够，五层都要改。FP32 在每种处理器上的预测都相同。）

## 22 · Result 13: INT8 on Microcontrollers（14:05）

> Next, microcontrollers. I built the trained models with CMSIS-NN and ran them in QEMU on three emulated Arm cores. On the Cortex-M0, which has no floating-point unit, INT8 needs 16 times fewer instructions than FP32. On the M4 it's 1.6 times, and on the M55 only 1.2. With the layers swapped, the temporal filters still do most of the work, because CMSIS-NN runs them as a general layer: about 11 instructions per multiply-add on the M4. So I wrote a filter-bank kernel for them. On the M4 the whole network drops from 6.9 to 2.7 million instructions per window. [Every INT8 build, with and without my kernel, matches my integer reference bit for bit.] [Accuracy stays within a third of a point of FP32.]

意思：接下来是单片机。我用 CMSIS-NN 编译训练好的模型，在 QEMU 里的三种模拟 Arm 内核上运行。Cortex-M0 没有浮点运算单元，在它上面 INT8 比 FP32 少用 16 倍的指令；M4 上是 1.6 倍，M55 上只有 1.2 倍。调换层顺序后，大部分工作仍然在时间滤波器上，因为 CMSIS-NN 把它们当成通用的层来算：在 M4 上每次乘加约 11 条指令。所以我给它们写了一个滤波器组内核。在 M4 上，整个网络每个窗口从 690 万条指令降到 270 万条。[每个 INT8 版本，不管加没加我的内核，都和我的整数参考实现逐位一致。] [准确率和 FP32 相差不到三分之一个点。]

If asked: QEMU counts instructions, not cycles, so these are not times or energy. Across all 136 builds my kernel cuts the whole network's instructions 1.5 to 2.8 times; on the M4 the temporal layer drops from 10.7 to 2.5 instructions per multiply-add, where the hardware's best case is 0.5. In INT8 with the swap, the 22-electrode model needs about 22 kilobytes of RAM on the M4, against about 400 in FP32 as exported.

（补充：QEMU 统计的是指令条数，不是时钟周期，所以这些不是时间，也不是能耗。在全部 136 个构建上，我的内核让整个网络的指令少 1.5 到 2.8 倍；M4 上时间卷积每次乘加从 10.7 条降到 2.5 条，硬件的上限是 0.5 条。INT8 加调换后，22 电极模型在 M4 上需要约 22 KB 内存，导出的 FP32 版本约 400 KB。）

## 23 · Result 14: Pruning（15:05）

> At the midterm, I was asked why pruning didn't make the model faster. It removed half the filters of the last convolution, a layer with under 1 percent of the multiply-adds, or under 5 percent after the swap. So little can be saved: 6.74 to 6.66 million instructions on the emulated M4, and 138 to 137 microseconds on the Mac. With equal training, pruning also cost 1.6 points. My frozen final model still scored 65.0 percent, the same mean as all 22, but 4 of the 9 people lost more than 2 points.

意思：期中时有人问：为什么剪枝没有让模型变快？剪枝删掉的是最后一个卷积层一半的滤波器，这一层只占不到 1% 的乘加运算，调换层顺序后也不到 5%。所以能省的很少：模拟的 M4 上指令从 674 万条降到 666 万条，Mac 上从 138 微秒降到 137 微秒。训练量相同时，剪枝还会损失 1.6 个点。冻结的最终模型仍然得了 65.0%，平均值和 22 个全用一样，不过 9 个人里有 4 个掉了 2 个点以上。

If asked: The 95 percent interval for the pruning cost is minus 2.86 to minus 0.42 points. The original comparison gave only the pruned models 100 extra epochs, which hid the cost.

（补充：剪枝损失的 95% 区间是 −2.86 到 −0.42 个点。原来的比较只给剪过的模型多训练了 100 轮，把损失掩盖了。）

## 24 · Demo: Real-Time Replay（15:50）

> Here is the decoder running. This replays participant 3's recorded day-2 EEG at 250 samples per second, through the deployed 16-electrode INT8 model. [The shaded band is the 2-second window the decoder reads after each cue.] Each decision takes about a third of a millisecond to compute, and the cursor moves toward the decoded movement. It's a replay, so the person couldn't react to it: this is not closed-loop control.

意思：这是解码器在运行。这段视频把第 3 位参与者第二天录下的脑电，按每秒 250 个采样点重放，送进部署用的 16 电极 INT8 模型。[阴影部分是解码器在每次提示后读取的 2 秒窗口。] 每次判断的计算大约三分之一毫秒，光标会向解码出的动作方向移动。这是重放，参与者看不到反馈，所以不是闭环控制。

If asked: The video is 62 seconds long, and I stop it after about 20. A decision is ready 2.5 seconds after the cue, plus the compute time.

（补充：视频长 62 秒，我放 20 秒左右就停。每次判断在提示后 2.5 秒加上计算时间时给出。）

## 25 · What Did Not Work（16:25）

> Several ideas didn't work. The redundancy penalty barely changed the choices. A learned selector did worst. Whole brain regions were weaker than single electrodes. Aligning the two days hurt the second dataset, because whitening blows up dozens of near-empty directions; not amplifying them removes the harm. Channel dropout cost 11 points, though with a little of it a failed electrode costs 2 points instead of 15. Registered predictions about extra inputs failed: less data did not make them costlier, four times longer training did not make them cheaper, and neither of two sparsity penalties switched them off. INT16 and CoreML made nothing faster. And pruning cost accuracy at equal training.

意思：有几个想法没有成功。重复度惩罚几乎没有改变选出的电极。学出来的选择层最差。按整个脑区选不如按单个电极选。两天对齐在第二个数据集上有害，因为白化把几十个几乎没有方差的方向放大了；不放大它们，害处就没了。通道 dropout 掉了 11 个点，不过加一点 dropout，坏掉一个电极只掉 2 个点，而不是 15 个点。关于多出的输入、事先登记的预测没有成立：训练数据更少，并没有让它们代价更大；训练 4 倍长，也没有让它们代价变小；两种稀疏惩罚，都没有把它们关掉。INT16 和 CoreML 都没有带来加速。训练量相同时，剪枝会损失准确率。

If asked: On the second dataset, alignment cost 5.5 points with 62 electrodes, and 42 of the 54 people got worse. It also hurt within the training day, and the covariance was estimated well; 49 of 62 directions had under 1 percent of the largest one's variance. With half the training data, 17 extra inputs cost 2.5 points less, and they removed 45 percent of the accuracy above chance either way.

（补充：在第二个数据集上，62 个电极时对齐掉了 5.5 个点，54 人中 42 人变差。在同一天之内对齐也有害，协方差也估计得准；62 个方向里有 49 个的方差不到最大方向的 1%。训练数据减半时，多出 17 个输入的代价反而少 2.5 个点；按高于瞎猜水平的那部分算，两种训练量下都掉了 45%。）

## 26 · Takeaways（17:15）

> To sum up. First, with one day of data, choose 16 to 20 electrodes before the network: EEGNet uses useless inputs to memorize its training trials, even with sparsity penalties; pretraining on many people changed that. Choosing them as a group beat one by one, in 54 new people. Second, 8 bits are enough. INT8 was slower only because a kernel was missing; with it, INT8 is 1.5 times faster than FP32, and almost 6 times faster with the layer swap and a second kernel. Third, compression pays only where the compute is: INT8 helps most without floating point, and pruning a layer with 1 percent of the work saves about 1 percent. Thank you, I'm happy to take questions.

意思：总结一下。第一，只有一天数据时，在网络之前先选好 16 到 20 个电极：EEGNet 会拿没用的输入去记训练试验，加了稀疏惩罚也一样；用很多人的数据预训练才改变了这一点。把电极当成一组来选比一个一个选更好，这在 54 个新的人身上也成立。第二，8 位就够了。INT8 之前变慢，只是因为缺一个内核；补上以后，INT8 比 FP32 快 1.5 倍；再调换两层、配上第二个内核，就快将近 6 倍。第三，压缩只在计算集中的地方才有用：INT8 在没有浮点单元的芯片上帮助最大；剪掉一个只占 1% 计算量的层，也只能省 1% 左右。谢谢，欢迎提问。
