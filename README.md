# stencil_CPU
Stencil computation on Arm architecture (Kunpeng 920)

For detail implementation, please refer to my blog: https://blog.csdn.net/weixin_43614211/article/details/122108753

# 以下为《Stencil计算：ARM-based CPU》博客的内容
（由于上述的CSDN需要付费才能看到完整博客内容，违背本人技术分享的初衷，此处附载博客全文）

机器：单节点内2片鲲鹏920 CPU（64x2=128核），这里最高扩展到8节点

## 串行程序优化
对于naïve版本的代码，不妨先“无脑”地加上`-O3 -fomit-frame-pointer -march=armv8-a -ffast-math`等编译选项来让编译器尽可能提供些自动向量化的效果。仅仅是如此，在不同规模的算例上性能就已经有3~5倍的提升，如下图中的VEC图例所示。

再修改benchmark.c内容对开辟的内存加上内存对齐的声明，如下图中ALIGNED图例所示，并无什么变化。从[StackOverflow](https://stackoverflow.com/questions/55232526/mpi-malloc-vs-mpi-alloc-mem-when-to-use)上了解到分配内存时使用的MPI_Alloc_mem函数（在一些实现中）可能已经做了内存对齐，故效果没有提升也合理。

进一步根据Gabriel Rivera等人写的[Tiling Optimizations for 3D Scientiﬁc Computations](https://dblp.uni-trier.de/rec/conf/sc/RiveraT00.html)，实行分块策略。按照Tiling的方法，逻辑和伪代码如左图所示，在固定的的x-y分区上逐层向上计算，每次先将该x-y分区内的Stencil计算完毕，再移动至下一个x-y分区，目的是每次换层的时候只需将3层a0中的一层替换出L1 cache，在有限的cache容量内尽量提高数据的可复用性。经过简单实验，得到最优的分块大小为$X$=256, $Y$=8。

除此以外，还可利用指针定位读写的位置，避免计算指标`INDEX(…)`时相互类似的大量计算。如下图所示

<table>
    <tr>
        <td ><center><img src="https://i-blog.csdnimg.cn/blog_migrate/db5a48867384e8b544b9c53c6c6f1560.png" >原有大量index计算</center></td>
        <td ><center><img src="https://i-blog.csdnimg.cn/blog_migrate/571bdfcbc6fb1a1e15c5f475ab956d44.png"  >规避冗余计算</center></td>
    </tr>
</table>

此部分整体的性能提升如下图所示。此部分的代码见stencil-naïve.c文件，和benchmark.sh, test.sh文件。

<table>
    <tr>
        <td ><center><img src="https://i-blog.csdnimg.cn/blog_migrate/4af0233650fa13a1fe2e16b06e80a8fc.png" >串行优化结果</center></td>
        <td ><center><img src="https://i-blog.csdnimg.cn/blog_migrate/e9ce2ad89bfc4ee853224b6bbf17c9ae.png"  >2D Tiling 示意</center></td>
    </tr>
</table>

## MPI并行
MPI并行模型使用分离的地址空间，因此每个进程做的计算互不干扰，主要需考虑通信带来的开销。由于有3个维度，对进程进行计算任务划分时有多种选择，因此首先从一维划分开始考虑。综合考虑实现复杂性和性能表现，使用MPI的Subarray type来组织和管理halo区的通信。说明，为使负载均衡，以下所有的划分都力求每个进程负责计算的区域大小相等，因此不能整除算例规模的划分方式不予考虑。在跨节点测试时，性能有一定波动，结果取多次测试中的最高值。此部分测试文件见mpi-benchmark.sh和mpi-test.sh文件。

### 一维z轴划分
将z轴等距划分给$np$个进程，每个进程负责$nx*ny*(nz/np)$的任务量。强可扩展性测试结果如下图所示，进程数一直增长到用满4个节点的共计512核。可见计算访存比更高的27点stencil的强可扩展性强于7点stencil。此部分代码见stencil-mpi-1dz.c文件。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/46135846cf5650b3e27df57eb1d17597.png#pic_center)

### 一维y轴划分
将y轴等距划分给$np$个进程，每个进程负责$nx*(ny/np)*nz$的任务量。强可扩展性测试结果如下图所示。性能结果与一维z轴划分类似。此部分代码见stencil-mpi-1dy.c文件。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3fb03641bb306513cb0c78fcf62398c5.png#pic_center)

### 一维x轴划分
将x轴等距划分给$np$个进程，每个进程负责$(nx/np)*ny*nz$的任务量。强可扩展性测试结果如下图所示。性能结果相比一维z或y轴划分有较大下降。这可能是因为x轴为三维数组的最内维，在x轴做划分会导致y和z变量变动时内存地址的跳跃较大。此部分代码见stencil-mpi-1dx.c文件。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d509dd0f552291655a0b34f7d258cecd.png#pic_center)

### 二维zy轴划分
综合上述一维划分的结果，在二维划分时考虑采用z和y轴联合划分。此部分代码见stencil-mpi-2dzy.c文件。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/09db94e45076ff4ef8daa7556e87cbe7.png#pic_center)

可见，使用二维划分时，由于划分有更大自由度的选择，能够使用更多的核数，并能获得更好的可扩展性。

### 计算通信重叠和非阻塞通信
基于上一节的二维zy轴划分，考虑计算通信重叠的实现，即每个进程先算自己的内halo区（邻居进程的外halo区），然后用非阻塞通信将内halo区数据通信。在此通信过程中，各进程计算自己真正的内部区域（不与其他进程有依赖关系的区域）。此部分代码见stencil-mpi-nb.c文件。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3dbf4c1aea10bfd194d78e1010d3adc2.png#pic_center)

值得一提的是，当使用8个节点（1024核）时，会出现执行程序非常慢，甚至有时提交任务太久没执行完而被作业系统杀掉的情况。但执行后输出的结果却显示时间仍然只是零点几秒，这大概是由于MPI-IO读取数据时非常耗时。具体原因我没有深究，但由于等待时间实在太久，所以只进行了test.sh中的测试，即跑了16个时间步的循环。而跨节点时本来性能就会有较大波动。

实际上，应用计算通信重叠会导致在某些并行度下性能有较明显的下降。这可能是因为刨去内halo区剩下的区域并不能对齐，导致后续在计算真正的内部区域时，会有更长的计算时间。所以在此只是尝试了一下，后续的优化没有应用计算通信重叠。

但**非阻塞通信仍然是很有意义的**。因为如果使用阻塞的`MPI_Sendrecv`进行进程对间的通信，将不得不写成如下图的形式，这里实际上引入了“序”的概念：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fd37d0ea566dec0e6039ae04d5dba2c0.png#pic_center)

为什么一定要以从1到4的方向顺序完成通信（发送、接受）呢？通信请求的发起是随机的，因此有可能某个时刻，序号为4的邻居进程向“我”发起了通信，但“我”还在等2号邻居，2号邻居因为算得慢，一直没有给“我”发信息，“我”只好在这干耗着，于是顺便也耗掉了4号邻居进程的时间。而这段“我”处于空闲的时间是可以用来与4号邻居完成通信的，这样显然更高效！因此可以用非阻塞的`MPI_Waitall`来去掉这个“序”的限制，通信可按任意顺序发生：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/cb662aef346f27503f1b948a2b0936f5.png#pic_center)

经此简单的修改，程序性能有明显提升，尤其是对于邻居数较多的27点程序（8邻居），它们的性能关切比只需4邻居通信的7点更严重。最后，应用上图中非阻塞通信、计算通信分离的性能如下图和下表所示：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f4f8ef5868b6bbbfafd8da0c90108a57.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c7e70467417259a7b2630c203840d154.png#pic_center)

### 节点内进程映射优化
经过进程映射的优化（进程尽可能均匀散布于整个节点，核与核之间距离尽可能远）可以得到单节点内（进程数较小时）更高的可扩展性！这有两个原因。

一方面是L1和L2 cache是各个cpu独有的，而L3 cache整个numa-region内的32个核共享。MPI程序是分离的地址空间，一个进程计算时所需访存的地址肯定与别的进程不一样，不怕伪共享，反而是多个进程共用一个numa-region内的核时会导致L3 cache共用而产生的capacity miss或conflict miss增多！所以应尽量让进程分布距离远一些，避免过度聚集而致共享的L3 cache过热（在进程数较少时可以独享或尽可能多占L3 cache），使整个节点的负载均衡。

另一方面，我认为更重要的是，内存总线一般是几个核共用一条的（具体的排线方式不同机器有差异，只是一般情况），比如在该节点128核内，0-3核（核组0），4-7核（核组1）等是以核组为单位共享内存总线的。所以当进程分布得更散落时，有利于提高机器的内存带宽利用率。这对于stencil这种memory-bounded、严重吃带宽的程序而言，应该是相比于cache更重要的因素。

该映射优化可以通过计算给定进程数$np$时均匀分布于整个节点的步长$stride$，和`mpirun`的命令行参数`--map-by slot:PE=$stride --bind-to core`来实现，可见mpi-benchmark.sh文件和mpi-test.sh文件。

下图为单节点内，基于上一节的优化上，使用进程映射优化和没有使用的对比。可见对比还是很明显的。尤其是对于计算访存比（AI：Arithmetic-Memory）更低的7点stencil，提升更加明显，印证了第二点的分析。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5d42fa85ec7e3c0bd863373ac77acc3e.png#pic_center)

### MPI小结
综合以上二维划分、非阻塞通信、进程映射的优化手段，最后汇总一下各种优化手段所能在纯MPI程序上获得的最高性能。对数坐标图不太容易看出具体数值，故列出如下表所示，性能单位GFlop/s，对应加速比见下表。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/28e5788b4ac68f4c520eef4786ac1dce.png#pic_center)

加速比如下图所示。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e5a44eda24ae65718be628de0fdec63b.png#pic_center)

## OpenMP并行
OpenMP只能限于单节点内的线程并行。考虑到串行最优实现的Tiling分块为$X$=256，$Y$=8，那么对于$n$=256的算例，最多只能有256/8=32的任务数，这对于有128个核的单节点而言，显然是不太合理的。

因此选用collapse(2)增加线程可分配的任务数，并且改小Tiling的分块为$X$=128，$Y$=8，以让更多的线程有任务可分。性能和加速比如下图所示。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6714de4743ec5ff1384cfed900277fd1.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3d1cee1dea77d5e661e67c888a9b9a83.png)

此部分代码见stencil-omp.c文件，omp-benchmark.sh和omp-test.sh文件。

值得一提的是，与MPI程序相反，OpenMP程序在线程映射（核绑定）时越靠近越好，这大概是因为**OpenMP各个线程之间是可以相互“影响”的**（例如，一个线程在计算时需要的数据可能之前已经被某个线程读取而放到了共享的L3cache上）因此它们挤在一起有利于相互利用。当然线程之间也可能存在伪共享而致性能降低，为避免此，不能将线程负责的分块设置过小，此处的Y=8足够避免了。

## MPI+OpenMP混合并行
综合上述第二、三节的优化技术手段，考虑MPI+OpenMP的混合并行，MPI进程间对计算区域做z、y轴的二维划分，每个进程得到一个子区域SubReigon_zy，而在进程内迸发出多个OpenMP线程对该子区域SubReigon_zy做进一步的任务划分和计算。

考虑到机器架构，每个节点内有2个socket，每个socket内有2个numa-region，每个numa-region内有32个核且它们共享L3 cache。因此一个自然的想法是，利用每个numa-region内放置一个MPI进程，并迸发出32个线程用满该numa-region的所有核，共同完成计算任务。值得注意的是，由`mpirun --map-by slot:PE=$stride --bind-to core`得到的进程所绑定的核号在一定范围内时随机的，所以在根据MPI进程所处核号来设置OpenMP线程所用的核号时，需要判断是否越界，如下图代码所示。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a0e501f5dff12d1e2a1a6b84bb8457ce.png#pic_center)

当然，后来助教解决此自动绑核问题了，不用这么麻烦手动操作绑核了。

分别用核数为1、4、128（1个节点）、256（2个节点）、512（4个节点）、1024（8个节点）测试，最后获得的性能和加速比如下图所示。该部分代码见stencil-hyb.c文件，测试脚本用hyb-benchmark.sh文件和hyb-test.sh文件。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/77a56e1044a5d995656337a4a99df62f.png#pic_center)

具体性能数值和加速比见下表。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ed4f9068974b347239b43ebd846e6ab8.png#pic_center)
