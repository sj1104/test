Design
=======

The development of deep learning (DL) algorithms and emerging DL models bring great challenges to underlying systems. Traditional DL systems, such as TensorFlow and PyTorch, have shown superior performance on various deep learning workloads due to their general characteristics and rich ecosystems. However, since the explosive growth of the scale of DL models and datasets, the distributed scalability is becoming the core competitiveness of DL systems. Although existing DL systems have provided some customized distributed training interfaces, they are still facing severe challenges and obstacles:

+ Functionality:
The supported communication architecture, parallel strategy and consistency protocal are limited.

+ Complexity:
The implementation of communication and compuation is highly coupled and hard to follow and optimize.

+ Usability:
The deployment of distributed training paradigms requires human expert knowledge for efficiency.

Besides, they are also suffering from the efficiency and scalability bottlenecks for large-scale distributed training. These observations motivate us to break the current system abstraction, make a novel design to handle all the above concerns and build a high-performance distributed DL system.

Hetu inherits the concept of data-flow graph (DFG) from existing deep learning frameworks, with operations as vertices and data dependencies as edges. The operation vertices not only represent computation kernels, but also consist of communication operators. Moreover, we provide a three-level representation of the DFG to describe and optimize distributed training programs.

Targeting on these above challenges, Hetu has the following advanced features:

+ Functionality:
Hetu supports various communication architectures, parallel strategies and consistency protocals.

<table style="margin: 0px auto;" cellspacing="0" cellpadding="3" border="1">
	<colgroup span="6" width="107"></colgroup>
	<tr>
		<td colspan=3 height="21" align="center" valign=middle><font color="#000000">Distributed DL System</font></td>
		<td align="left" valign=middle><font color="#000000">TensorFlow</font></td>
		<td align="left" valign=middle><font color="#000000">PyTorch</font></td>
		<td align="left" valign=middle><font color="#000000">Hetu</font></td>
	</tr>
	<tr>
		<td rowspan=3 height="64" align="center" style="vertical-align:middle;" valign=middle><font color="#000000">Communication Architecture</font></td>
		<td colspan=2 align="center" valign=middle><font color="#000000">Parameter Server</font></td>
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
		<td colspan=2 align="center" valign=middle><font color="#000000">Hybrid architecture</font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td rowspan=5 height="107" align="center" style="vertical-align:middle;" valign=middle><font color="#000000">Parallel Strategy</font></td>
		<td colspan=2 align="center" valign=middle><font color="#000000">Data Parallel</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td rowspan=4 align="center" style="vertical-align:middle;" valign=middle><font color="#000000">Model Parallel</font></td>
		<td align="left" valign=middle><font color="#000000">Parameter</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td align="left" valign=middle><font color="#000000">Operator</font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td align="left" valign=middle><font color="#000000">Pipeline</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td align="left" valign=middle><font color="#000000">Subgraph</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td rowspan=3 height="64" align="center" style="vertical-align:middle;" valign=middle><font color="#000000">Consistency Protocal</font></td>
		<td colspan=2 align="center" valign=middle><font color="#000000">Fully Synchronous</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td colspan=2 align="center" valign=middle><font color="#000000">Asynchronous</font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
	<tr>
		<td colspan=2 align="center" valign=middle><font color="#000000">Staleness synchronous</font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
		<td align="left" valign=middle><font color="#000000"><br></font></td>
		<td align="left" valign=middle><font color="#000000"><span>&#10004;</span></font></td>
	</tr>
</table>

+ Complexity:
Hetu decouples the communication and compuation procedures into separate operators and applies optional dependencies and asynchronous execution logic for high performance.

+ Usability:
Hetu provides semi-automatical parallel training interfaces for human experts and fully-automatical parallel training functionality for zero-knowledge users.