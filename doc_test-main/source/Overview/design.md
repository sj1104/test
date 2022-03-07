系统设计
=======
深度学习（DL）算法的发展和新兴的DL模型给底层系统带来了巨大的挑战。传统的DL系统，如TensorFlow和PyTorch，由于其一般特征和丰富的生态系统，在各种深度学习工作负载上表现出了优异的性能。然而，随着DL模型和数据集规模的爆炸式增长，分布式可扩展性正成为DL系统的核心竞争力。尽管现有的DL系统提供了一些定制的分布式训练接口，但它们仍然面临着严峻的挑战和障碍：

+ 功能性：
支持的通信体系结构、并行策略和一致性协议有限。

+ 复杂性：
通信和计算的实现高度耦合，难以追踪和优化。

+ 实用性：
分布式训练模式的部署需要人类专家知识来提高效率。

此外，DL系统还面临着大规模分布式训练的效率和可扩展性瓶颈。这些观察结果促使我们打破当前的系统抽象，进行一种新颖的设计来处理所有上述问题，并构建一个高性能的分布式DL系统。

Hetu从现有的深度学习框架中继承了数据流图（DFG）的概念，操作是顶点，数据依赖是边。操作顶点不仅代表计算核，还包含通信算子。此外，我们还提供了DFG的三级表示，用于描述和优化分布式训练计划。

针对上述挑战，Hetu具有以下特征：

+ 功能性：
Hetu支持各种通信架构、并行策略和一致性协议。

<table style="margin: 0px auto;" cellspacing="0" cellpadding="3" border="1">
	<colgroup span="6" width="107"></colgroup>
	<tr>
		<td colspan=3 height="21" align="center" valign=middle><font color="#000000">分布式DL系统</font></td>
		<td align="left" valign=middle><font color="#000000">TensorFlow</font></td>
		<td align="left" valign=middle><font color="#000000">PyTorch</font></td>
		<td align="left" valign=middle><font color="#000000">Hetu</font></td>
	</tr>
	<tr>
		<td rowspan=3 height="64" align="center" style="vertical-align:middle;" valign=middle><font color="#000000">通信架构</font></td>
		<td colspan=2 align="center" valign=middle><font color="#000000">参数服务器</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td colspan=2 align="center" valign=middle><font color="#000000">All-Reduce</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td colspan=2 align="center" valign=middle><font color="#000000">混合架构</font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td rowspan=5 height="107" align="center" style="vertical-align:middle;" valign=middle><font color="#000000">并行策略</font></td>
		<td colspan=2 align="center" valign=middle><font color="#000000">数据并行</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td rowspan=4 align="center" style="vertical-align:middle;" valign=middle><font color="#000000">模型并行</font></td>
		<td align="left" valign=middle><font color="#000000">参数</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td align="left" valign=middle><font color="#000000">算子</font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td align="left" valign=middle><font color="#000000">流水</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td align="left" valign=middle><font color="#000000">子图</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td rowspan=3 height="64" align="center" style="vertical-align:middle;" valign=middle><font color="#000000">一致性协议</font></td>
		<td colspan=2 align="center" valign=middle><font color="#000000">完全同步</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td colspan=2 align="center" valign=middle><font color="#000000">完全异步</font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td colspan=2 align="center" valign=middle><font color="#000000">宽松同步</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
</table>

+ 复杂性：
Hetu将通信和计算过程解耦为单独的运算符，并应用可选依赖项和异步执行逻辑以实现高性能。

+ 实用性：
Hetu为专家提供半自动并行训练接口，为零知识用户实现全自动并行训练功能。